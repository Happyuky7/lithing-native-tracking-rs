//! The web tracker's low / high light modes ("Mejorar cámara con poca luz" /
//! "Mejorar cámara sobreexpuesta"): the face models see a filtered copy of
//! the camera frame, drawn into a 640x480 canvas with a CSS filter
//! (src/app/face-tracker.ts `update`, detection-worker.ts `processLight`):
//!
//! * low light: `blur(b) brightness(B) contrast(C) saturate(0.78)` where `B`
//!   follows a light meter (the frame drawn into a 32x24 canvas, centre
//!   24x20 averaged) with an exponential moving average, and `C`, `b` follow
//!   `B` and the measured luminance;
//! * high light: `brightness(0.72) contrast(1.1) saturate(0.9)`.
//!
//! Hands and the tongue detector keep the unfiltered frame, as on the web.
//!
//! The filter reproduces Chromium's 2D canvas semantics (Filter Effects
//! shorthands, in sRGB), measured in headless Chrome (see ARCHITECTURE.md,
//! "Light filters"):
//!
//! * `brightness(s)` and `contrast(c)` are FEComponentTransfer linear
//!   functions, evaluated by Blink into 8-bit lookup tables in f32 and
//!   truncated: `table[i] = clamp(trunc(slope * i + intercept * 255))`,
//!   brightness `slope = s, intercept = 0`, contrast `slope = c,
//!   intercept = 0.5 - 0.5c`; consecutive tables compose.
//! * `saturate(s)` is the feColorMatrix saturate matrix
//!   (`0.213 + 0.787s, 0.715 - 0.715s, 0.072 - 0.072s, ...`) rounded to the
//!   nearest 8-bit value. It equals `s * c + (1 - s) * luma`, which is how it
//!   is computed here (two table lookups per channel).
//! * `blur(σ)` on the app's accelerated canvas is a separable Gaussian with
//!   radius `ceil(3σ)`, weights normalised; pixels outside the image are
//!   transparent and the result is unpremultiplied, i.e. the weights are
//!   renormalised over the pixels inside. The blurred image is stored as
//!   8-bit before the colour filters run. (Chromium's software canvas skips
//!   blurs this small; the app's canvas is the GPU one.)
//! * `drawImage` of a frame of another size into the 640x480 canvas, and of
//!   the frame into the 32x24 light meter, is bilinear (no mipmaps).

use crate::frame::Frame;
use std::time::Instant;

/// The processing canvas of the web tracker.
pub const PROCESSING_WIDTH: u32 = 640;
pub const PROCESSING_HEIGHT: u32 = 480;
/// `lowLightBrightness` before the first measurement.
pub const INITIAL_LOW_LIGHT_BRIGHTNESS: f64 = 1.38;
/// Chromium's accelerated canvas writes its blur and colour-matrix results
/// about 0.035 of a level low (measured on D3D11 / NVIDIA, the app's WebView2
/// path: `round(v - 0.035)` halves the mismatches of plain rounding, 3.1% to
/// 1.1% for the blur and 3% to 1.5% for saturate). Exact .5 ties of a plain
/// rescale still round up, so the rescale does not use it.
pub const GPU_ROUNDING_BIAS: f64 = 0.035;

/// Which camera fix is on. Low light wins when both flags are set, as on the web.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum LightMode {
    #[default]
    None,
    Low,
    High,
}

impl LightMode {
    pub fn from_flags(low_light: bool, high_light: bool) -> Self {
        if low_light {
            Self::Low
        } else if high_light {
            Self::High
        } else {
            Self::None
        }
    }
}

/// MediaPipe face landmarker thresholds per mode (face-tracker.ts `faceConfidenceOptions`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceConfidence {
    pub detection: f32,
    pub presence: f32,
    /// MediaPipe's `minTrackingConfidence`: the IoU above which a new face
    /// detection is merged with a tracked face. With one face the detector
    /// only runs while no face is tracked, so it has no effect on results;
    /// kept for completeness.
    pub tracking: f32,
}

pub fn face_confidence(mode: LightMode) -> FaceConfidence {
    match mode {
        LightMode::Low => FaceConfidence { detection: 0.28, presence: 0.34, tracking: 0.38 },
        LightMode::High => FaceConfidence { detection: 0.38, presence: 0.44, tracking: 0.5 },
        LightMode::None => FaceConfidence { detection: 0.45, presence: 0.5, tracking: 0.58 },
    }
}

/// One CSS filter chain `blur(blur px) brightness(..) contrast(..) saturate(..)`;
/// `blur == 0` leaves the blur out.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct CssFilter {
    pub blur: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub saturate: f64,
}

impl CssFilter {
    /// The high light filter.
    pub const HIGH_LIGHT: Self = Self { blur: 0.0, brightness: 0.72, contrast: 1.1, saturate: 0.9 };

    /// The same string the web tracker assigns to `context.filter`.
    pub fn css(&self) -> String {
        if *self == Self::HIGH_LIGHT {
            return "brightness(0.72) contrast(1.1) saturate(0.9)".into();
        }
        // Low light: every value written with toFixed(2).
        let colour = format!("brightness({:.2}) contrast({:.2}) saturate({})", self.brightness, self.contrast, self.saturate);
        if self.blur > 0.0 { format!("blur({:.2}px) {colour}", self.blur) } else { colour }
    }
}

/// JavaScript's `Number.prototype.toFixed(2)`, parsed back: the nearest
/// multiple of 0.01 to the exact value of `x`, ties to the larger one.
pub fn to_fixed2(x: f64) -> f64 {
    // Rust prints the exact decimal expansion of a double; look at the digits past the second decimal.
    let exact = format!("{:.60}", x.abs());
    let (whole, fraction) = exact.split_once('.').unwrap_or((&exact, "0"));
    let hundredths: u64 = whole.parse::<u64>().unwrap_or(0) * 100 + fraction[..2].parse::<u64>().unwrap_or(0);
    let rest = &fraction[2..];
    let round_up = match rest.as_bytes()[0] {
        b'5' => true, // above or exactly half: toFixed picks the larger n (for x >= 0)
        d => d > b'5',
    };
    let n = hundredths + round_up as u64;
    let value = n as f64 / 100.0;
    if x < 0.0 {
        // Ties go to the larger n, i.e. towards +inf: for negatives an exact half rounds down in magnitude.
        let tie = rest.as_bytes()[0] == b'5' && rest[1..].bytes().all(|b| b == b'0');
        -(if tie { hundredths as f64 / 100.0 } else { value })
    } else {
        value
    }
}

/// What the light processing did for one face frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct LightState {
    pub mode: LightMode,
    /// Centre luminance 0..1 from the light meter (low light only).
    pub luminance: f64,
    /// The filter drawn with.
    pub filter: CssFilter,
    /// Cost of metering and filtering, milliseconds.
    pub ms: f32,
}

/// Colour stage tables for one (brightness, contrast, saturate).
struct ColourTables {
    key: (u64, u64, u64),
    /// `s * table[v]` in Q16 (the channel's own term of the saturate matrix).
    own: [u32; 256],
    /// `(1 - s) * 0.213 / 0.715 / 0.072 * table[v]` in Q16 (the luma term).
    luma: [[u32; 256]; 3],
}

/// Gaussian kernels for one sigma and image size, Q14 fixed point.
struct BlurKernel {
    sigma: f32,
    size: (usize, usize),
    radius: usize,
    /// `[2 * radius + 1]` weights for pixels at least `radius` from both edges.
    interior: Vec<u32>,
    /// Columns / rows within `radius` of an edge and their renormalised weights.
    edge_columns: Vec<(usize, Vec<u32>)>,
    edge_rows: Vec<(usize, Vec<u32>)>,
}

/// The light processing of the web tracker, with its state (the low light
/// brightness moving average). One per face pipeline.
pub struct LightFilter {
    brightness: f64,
    scaled: Vec<u8>,
    horizontal: Vec<u16>,
    row: Vec<u8>,
    acc: Vec<u32>,
    tables: Option<ColourTables>,
    kernel: Option<BlurKernel>,
    output: Frame,
    pub last: LightState,
}

impl Default for LightFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl LightFilter {
    pub fn new() -> Self {
        let (w, h) = (PROCESSING_WIDTH as usize, PROCESSING_HEIGHT as usize);
        Self {
            brightness: INITIAL_LOW_LIGHT_BRIGHTNESS,
            scaled: Vec::new(),
            horizontal: Vec::new(),
            row: Vec::new(),
            acc: Vec::new(),
            tables: None,
            kernel: None,
            output: Frame::new(0, PROCESSING_WIDTH, PROCESSING_HEIGHT, vec![0; w * h * 3]),
            last: LightState::default(),
        }
    }

    /// The low light brightness moving average (unrounded).
    pub fn brightness(&self) -> f64 {
        self.brightness
    }

    /// The filter for this face frame; in low light this measures `frame`
    /// and advances the brightness average (once per face detection, as on the web).
    pub fn next_filter(&mut self, frame: &Frame, mode: LightMode) -> Option<(CssFilter, f64)> {
        match mode {
            LightMode::None => None,
            LightMode::High => Some((CssFilter::HIGH_LIGHT, 0.0)),
            LightMode::Low => {
                let luminance = measure_center_luminance(frame);
                let target = (0.42 / luminance.max(0.08)).clamp(1.12, 2.45);
                self.brightness += (target - self.brightness) * 0.16;
                let contrast = (1.04 + (0.34 - luminance).max(0.0) * 0.48).clamp(1.04, 1.2);
                let blur = if self.brightness > 1.8 { 0.65 } else { 0.42 };
                let filter = CssFilter { blur: to_fixed2(blur), brightness: to_fixed2(self.brightness), contrast: to_fixed2(contrast), saturate: 0.78 };
                Some((filter, luminance))
            }
        }
    }

    /// The frame the face models should see: `frame` itself without a
    /// light mode, else the filtered 640x480 copy (valid until the next call),
    /// and what was done (also kept in `last`).
    pub fn process<'a>(&'a mut self, frame: &'a Frame, mode: LightMode) -> (&'a Frame, LightState) {
        let start = Instant::now();
        let Some((filter, luminance)) = self.next_filter(frame, mode) else {
            self.last = LightState::default();
            return (frame, self.last);
        };
        self.output.id = frame.id;
        self.output.captured = frame.captured;
        self.output.pts_ms = frame.pts_ms;
        self.render(frame, &filter);
        self.last = LightState { mode, luminance, filter, ms: start.elapsed().as_secs_f32() * 1000.0 };
        (&self.output, self.last)
    }

    /// Draws `frame` into the 640x480 output with `filter` (no metering, no state change).
    pub fn render(&mut self, frame: &Frame, filter: &CssFilter) -> &Frame {
        let (w, h) = (PROCESSING_WIDTH as usize, PROCESSING_HEIGHT as usize);
        let source: &[u8] = if frame.width == PROCESSING_WIDTH && frame.height == PROCESSING_HEIGHT {
            &frame.rgb
        } else {
            self.scaled.resize(w * h * 3, 0);
            resize_bilinear(&frame.rgb, frame.width as usize, frame.height as usize, &mut self.scaled, w, h, &mut self.acc);
            &self.scaled
        };
        let key = (filter.brightness.to_bits(), filter.contrast.to_bits(), filter.saturate.to_bits());
        if self.tables.as_ref().is_none_or(|t| t.key != key) {
            self.tables = Some(ColourTables::new(filter));
        }
        let tables = self.tables.as_ref().expect("tables");
        let kernel = if filter.blur > 0.0 {
            let sigma = filter.blur as f32;
            if self.kernel.as_ref().is_none_or(|k| k.sigma != sigma || k.size != (w, h)) {
                self.kernel = Some(BlurKernel::new(sigma, w, h));
            }
            self.horizontal.resize(w * h * 3, 0);
            self.row.resize(w * 3, 0);
            self.kernel.as_ref()
        } else {
            None
        };
        let mut job = FilterJob { source, w, h, kernel, tables, horizontal: &mut self.horizontal, row: &mut self.row, acc: &mut self.acc, out: &mut self.output.rgb };
        job.run();
        &self.output
    }
}

/// One filtered image: the blur passes and the colour stage, row by row.
struct FilterJob<'a> {
    source: &'a [u8],
    w: usize,
    h: usize,
    kernel: Option<&'a BlurKernel>,
    tables: &'a ColourTables,
    horizontal: &'a mut Vec<u16>,
    row: &'a mut Vec<u8>,
    acc: &'a mut Vec<u32>,
    out: &'a mut [u8],
}

impl FilterJob<'_> {
    fn run(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2 (checked above).
            unsafe { self.run_avx2() };
            return;
        }
        self.run_generic();
    }

    /// The same code compiled for AVX2 (8 lanes for the u32 blur accumulators).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn run_avx2(&mut self) {
        self.run_generic();
    }

    #[inline(always)]
    fn run_generic(&mut self) {
        let stride = self.w * 3;
        match self.kernel {
            Some(kernel) => {
                blur_horizontal(self.source, self.w, self.h, kernel, self.acc, self.horizontal);
                for y in 0..self.h {
                    blur_vertical_row(self.horizontal, self.w, y, kernel, self.acc, self.row);
                    self.tables.apply(self.row, &mut self.out[y * stride..(y + 1) * stride]);
                }
            }
            None => self.tables.apply(self.source, self.out),
        }
    }
}

/// The web light meter: the frame drawn into a 32x24 canvas (bilinear), the
/// centre 24x20 pixels' Rec. 709 luma averaged, 0..1.
pub fn measure_center_luminance(frame: &Frame) -> f64 {
    const MW: usize = 32;
    const MH: usize = 24;
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w == 0 || h == 0 {
        return 0.3;
    }
    // Sample positions as Skia maps them: through the inverse of the f32 scale (exact at
    // 640x480 and 1280x960 against Chrome, within 4e-5 of its luminance at 1280x720).
    let (inv_x, inv_y) = (1.0 / (MW as f32 / w as f32), 1.0 / (MH as f32 / h as f32));
    let mut sum = 0.0f64;
    for y in 2..22 {
        for x in 4..28 {
            // Chromium's software canvas truncates the interpolated value (measured: 0 of 2304 differ).
            let [r, g, b] = sample_bilinear(&frame.rgb, w, h, (x as f32 + 0.5) * inv_x - 0.5, (y as f32 + 0.5) * inv_y - 0.5, 0.0);
            sum += r as f64 * 0.2126 + g as f64 * 0.7152 + b as f64 * 0.0722;
        }
    }
    sum / (24.0 * 20.0) / 255.0
}

#[inline]
fn sample_bilinear(rgb: &[u8], w: usize, h: usize, sx: f32, sy: f32, rounding: f32) -> [u8; 3] {
    let sx = sx.clamp(0.0, (w - 1) as f32);
    let sy = sy.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (sx as usize, sy as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let p = |x: usize, y: usize| rgb[(y * w + x) * 3 + c] as f32;
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
        let bottom = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
        out[c] = (top + (bottom - top) * fy + rounding) as u8;
    }
    out
}

/// `drawImage` of a `sw`x`sh` frame into a `dw`x`dh` canvas: bilinear,
/// pixel centres aligned, weights in 1/256 (GPU filtering precision),
/// separable. Against Chrome's accelerated canvas: an exact 2x downscale
/// rounds half up (0 values differ); other ratios come out GPU_ROUNDING_BIAS
/// low (1280x720: 0.6% of values one level off).
pub fn resize_bilinear(src: &[u8], sw: usize, sh: usize, dst: &mut [u8], dw: usize, dh: usize, column: &mut Vec<u32>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2 (checked above).
        unsafe { resize_bilinear_avx2(src, sw, sh, dst, dw, dh, column) };
        return;
    }
    resize_bilinear_generic(src, sw, sh, dst, dw, dh, column);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn resize_bilinear_avx2(src: &[u8], sw: usize, sh: usize, dst: &mut [u8], dw: usize, dh: usize, column: &mut Vec<u32>) {
    resize_bilinear_generic(src, sw, sh, dst, dw, dh, column);
}

#[inline(always)]
fn resize_bilinear_generic(src: &[u8], sw: usize, sh: usize, dst: &mut [u8], dw: usize, dh: usize, column: &mut Vec<u32>) {
    let position = |i: usize, scale: f32, max: usize| {
        let s = ((i as f32 + 0.5) * scale - 0.5).clamp(0.0, (max - 1) as f32);
        let i0 = s as usize;
        (i0, (i0 + 1).min(max - 1), ((s - i0 as f32) * 256.0).round() as u32)
    };
    let columns: Vec<(usize, usize, u32)> = (0..dw).map(|x| position(x, sw as f32 / dw as f32, sw)).collect();
    let stride = sw * 3;
    let round = if sw == 2 * dw && sh == 2 * dh { 32768 } else { 32768 - (GPU_ROUNDING_BIAS * 65536.0 + 0.5) as u32 };
    column.resize(stride, 0);
    for y in 0..dh {
        let (y0, y1, fy) = position(y, sh as f32 / dh as f32, sh);
        // Vertical lerp of the two source rows (Q8), then horizontal (Q16).
        let (top, bottom) = (&src[y0 * stride..(y0 + 1) * stride], &src[y1 * stride..(y1 + 1) * stride]);
        for ((c, &a), &b) in column.iter_mut().zip(top).zip(bottom) {
            *c = a as u32 * (256 - fy) + b as u32 * fy;
        }
        let out = &mut dst[y * dw * 3..(y + 1) * dw * 3];
        let pixels = column.as_chunks::<3>().0;
        for (pixel, &(x0, x1, fx)) in out.as_chunks_mut::<3>().0.iter_mut().zip(&columns) {
            let (a, b) = (pixels[x0], pixels[x1]);
            for c in 0..3 {
                pixel[c] = ((a[c] * (256 - fx) + b[c] * fx + round) >> 16) as u8;
            }
        }
    }
}

/// One FEComponentTransfer linear table as Blink builds it (f32, truncated).
fn linear_table(slope: f32, intercept: f32) -> [u8; 256] {
    let mut table = [0u8; 256];
    let offset = intercept * 255.0;
    for (i, slot) in table.iter_mut().enumerate() {
        let value = slope * i as f32 + offset;
        *slot = value.clamp(0.0, 255.0) as u8;
    }
    table
}

/// The composed brightness then contrast table of `filter`.
pub fn transfer_table(filter: &CssFilter) -> [u8; 256] {
    let brightness = linear_table(filter.brightness as f32, 0.0);
    let contrast_slope = filter.contrast as f32;
    let contrast = linear_table(contrast_slope, -0.5 * contrast_slope + 0.5);
    std::array::from_fn(|i| contrast[brightness[i] as usize])
}

impl ColourTables {
    fn new(filter: &CssFilter) -> Self {
        let table = transfer_table(filter);
        let s = filter.saturate;
        let q = 65536.0;
        let own = std::array::from_fn(|i| (s * table[i] as f64 * q).round() as u32);
        let weights = [0.213, 0.715, 0.072];
        let luma = std::array::from_fn(|c| std::array::from_fn(|i| ((1.0 - s) * weights[c] * table[i] as f64 * q).round() as u32));
        Self { key: (filter.brightness.to_bits(), filter.contrast.to_bits(), filter.saturate.to_bits()), own, luma }
    }

    /// Brightness, contrast and saturate on packed RGB.
    #[inline(always)]
    fn apply(&self, src: &[u8], dst: &mut [u8]) {
        let (own, [lr, lg, lb]) = (&self.own, &self.luma);
        // Round to nearest, GPU_ROUNDING_BIAS low.
        const ROUND: u32 = 0x8000 - (GPU_ROUNDING_BIAS * 65536.0 + 0.5) as u32;
        for (s, d) in src.as_chunks::<3>().0.iter().zip(dst.as_chunks_mut::<3>().0.iter_mut()) {
            let (r, g, b) = (s[0] as usize, s[1] as usize, s[2] as usize);
            let luma = lr[r] + lg[g] + lb[b] + ROUND;
            d[0] = ((own[r] + luma) >> 16).min(255) as u8;
            d[1] = ((own[g] + luma) >> 16).min(255) as u8;
            d[2] = ((own[b] + luma) >> 16).min(255) as u8;
        }
    }
}

/// Q14 weights of the taps `-radius..=radius` around `position` in a line of
/// `len` pixels, renormalised over the taps inside it (transparent outside, unpremultiplied).
fn edge_weights(weights: &[f64], radius: usize, position: usize, len: usize) -> Vec<u32> {
    let inside = |k: usize| position + k >= radius && position + k - radius < len;
    let total: f64 = (0..weights.len()).filter(|&k| inside(k)).map(|k| weights[k]).sum();
    quantize((0..weights.len()).map(|k| if inside(k) { weights[k] / total } else { 0.0 }).collect())
}

/// Q14 weights that add up to exactly 16384 (the remainder goes to the largest).
fn quantize(weights: Vec<f64>) -> Vec<u32> {
    let mut q: Vec<u32> = weights.iter().map(|w| (w * 16384.0).round() as u32).collect();
    let sum: i64 = q.iter().map(|&v| v as i64).sum();
    let largest = (0..q.len()).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).unwrap_or(0);
    q[largest] = (q[largest] as i64 + 16384 - sum) as u32;
    q
}

impl BlurKernel {
    fn new(sigma: f32, w: usize, h: usize) -> Self {
        let radius = (3.0 * sigma).ceil().max(1.0) as usize;
        let s = sigma as f64;
        let raw: Vec<f64> = (0..=2 * radius).map(|k| (-((k as f64 - radius as f64).powi(2)) / (2.0 * s * s)).exp()).collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        // Near an edge: renormalised over the taps inside the image.
        let edges = |len: usize| (0..len).filter(|&i| i < radius || i + radius >= len).map(|i| (i, edge_weights(&raw, radius, i, len))).collect();
        Self { sigma, size: (w, h), radius, interior: quantize(weights), edge_columns: edges(w), edge_rows: edges(h) }
    }

    fn row_weights(&self, y: usize) -> &[u32] {
        self.edge_rows.iter().find(|(row, _)| *row == y).map_or(&self.interior, |(_, weights)| weights)
    }
}

/// Horizontal pass: u8 RGB rows to Q8 (value * 256) u16.
#[inline(always)]
fn blur_horizontal(src: &[u8], w: usize, h: usize, kernel: &BlurKernel, acc: &mut Vec<u32>, dst: &mut [u16]) {
    let r = kernel.radius;
    let stride = w * 3;
    let weights = &kernel.interior;
    let span = stride.saturating_sub(6 * r);
    acc.resize(span.max(stride), 0);
    for y in 0..h {
        let row = &src[y * stride..(y + 1) * stride];
        let out = &mut dst[y * stride..(y + 1) * stride];
        if w > 2 * r {
            // Interior: a plain 1-D convolution over bytes with a 3-byte tap stride (vectorises).
            let acc = &mut acc[..span];
            acc.fill(0);
            for (k, &weight) in weights.iter().enumerate() {
                for (a, &v) in acc.iter_mut().zip(&row[3 * k..3 * k + span]) {
                    *a += weight * v as u32;
                }
            }
            for (o, a) in out[3 * r..3 * r + span].iter_mut().zip(acc.iter()) {
                *o = ((a + 32) >> 6) as u16;
            }
        }
        for (x, w_edge) in &kernel.edge_columns {
            for c in 0..3 {
                let mut sum = 0u32;
                for (k, &weight) in w_edge.iter().enumerate() {
                    if weight != 0 {
                        sum += weight * row[(x + k - r) * 3 + c] as u32;
                    }
                }
                out[x * 3 + c] = ((sum + 32) >> 6) as u16;
            }
        }
    }
}

/// Vertical pass of output row `y`: Q8 u16 rows to u8 (rounded, GPU_ROUNDING_BIAS low).
#[inline(always)]
fn blur_vertical_row(src: &[u16], w: usize, y: usize, kernel: &BlurKernel, acc: &mut Vec<u32>, out: &mut [u8]) {
    const ROUND: u32 = (1 << 21) - (GPU_ROUNDING_BIAS * (1u32 << 22) as f64 + 0.5) as u32;
    let r = kernel.radius;
    let stride = w * 3;
    let weights = kernel.row_weights(y);
    acc.resize(stride, 0);
    let acc = &mut acc[..stride];
    acc.fill(0);
    for (k, &weight) in weights.iter().enumerate() {
        if weight == 0 {
            continue;
        }
        let row = &src[(y + k - r) * stride..(y + k - r + 1) * stride];
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += weight * v as u32;
        }
    }
    for (o, a) in out.iter_mut().zip(acc.iter()) {
        *o = ((a + ROUND) >> 22).min(255) as u8;
    }
}

/// Reference implementation in f64 of the whole filter on a 640x480 (or any
/// size) RGB image, for tests: the Gaussian, the tables and the saturate matrix
/// computed directly.
pub fn reference_filter(src: &[u8], w: usize, h: usize, filter: &CssFilter) -> Vec<u8> {
    let mut blurred = src.to_vec();
    if filter.blur > 0.0 {
        let s = filter.blur as f32 as f64;
        let r = (3.0 * s).ceil().max(1.0) as isize;
        let g: Vec<f64> = (-r..=r).map(|k| (-(k * k) as f64 / (2.0 * s * s)).exp()).collect();
        let mut tmp = vec![0f64; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let (mut acc, mut total) = (0.0, 0.0);
                    for k in -r..=r {
                        let sx = x as isize + k;
                        if sx >= 0 && (sx as usize) < w {
                            acc += g[(k + r) as usize] * src[(y * w + sx as usize) * 3 + c] as f64;
                            total += g[(k + r) as usize];
                        }
                    }
                    tmp[(y * w + x) * 3 + c] = acc / total;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let (mut acc, mut total) = (0.0, 0.0);
                    for k in -r..=r {
                        let sy = y as isize + k;
                        if sy >= 0 && (sy as usize) < h {
                            acc += g[(k + r) as usize] * tmp[(sy as usize * w + x) * 3 + c];
                            total += g[(k + r) as usize];
                        }
                    }
                    blurred[(y * w + x) * 3 + c] = (acc / total - GPU_ROUNDING_BIAS).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    reference_colour(&blurred, filter)
}

/// Reference of the colour stage alone (tables, then the saturate matrix in f32).
pub fn reference_colour(src: &[u8], filter: &CssFilter) -> Vec<u8> {
    let table = transfer_table(filter);
    let s = filter.saturate as f32;
    let m = [
        [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
    ];
    let mut out = vec![0u8; src.len()];
    for (p, o) in src.as_chunks::<3>().0.iter().zip(out.as_chunks_mut::<3>().0.iter_mut()) {
        let v = [table[p[0] as usize] as f32 / 255.0, table[p[1] as usize] as f32 / 255.0, table[p[2] as usize] as f32 / 255.0];
        for c in 0..3 {
            let value = m[c][0] * v[0] + m[c][1] * v[1] + m[c][2] * v[2];
            o[c] = (value.clamp(0.0, 1.0) * 255.0 - GPU_ROUNDING_BIAS as f32 + 0.5).max(0.0) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_and_thresholds_match_the_web() {
        // Low light wins when both are set (face-tracker.ts checks lowLightMode first).
        assert_eq!(LightMode::from_flags(true, true), LightMode::Low);
        assert_eq!(LightMode::from_flags(false, true), LightMode::High);
        assert_eq!(face_confidence(LightMode::Low), FaceConfidence { detection: 0.28, presence: 0.34, tracking: 0.38 });
        assert_eq!(face_confidence(LightMode::High), FaceConfidence { detection: 0.38, presence: 0.44, tracking: 0.5 });
        assert_eq!(face_confidence(LightMode::None), FaceConfidence { detection: 0.45, presence: 0.5, tracking: 0.58 });
        assert_eq!(CssFilter::HIGH_LIGHT.css(), "brightness(0.72) contrast(1.1) saturate(0.9)");
        assert_eq!(CssFilter { blur: 0.42, brightness: 1.7, contrast: 1.04, saturate: 0.78 }.css(), "blur(0.42px) brightness(1.70) contrast(1.04) saturate(0.78)");
    }

    #[test]
    fn to_fixed_matches_javascript() {
        // Values checked against node: (x).toFixed(2).
        for (x, expected) in [(1.38, 1.38), (1.005, 1.0), (1.125, 1.13), (2.4449999, 2.44), (1.8150000001, 1.82), (0.65, 0.65), (1.0400000000000003, 1.04), (1.115, 1.11), (1.375, 1.38)] {
            assert_eq!(to_fixed2(x), expected, "{x}");
        }
    }

    #[test]
    fn tables_match_chromium() {
        // Measured in headless Chrome 156 (gray ramp through the canvas filter).
        let b = linear_table(1.38, 0.0);
        assert_eq!((b[50], b[100], b[127], b[128], b[129], b[150], b[184], b[185]), (69, 138, 175, 176, 178, 207, 253, 255));
        let c = transfer_table(&CssFilter { blur: 0.0, brightness: 1.0, contrast: 1.1, saturate: 1.0 });
        assert_eq!((c[1], c[50], c[100], c[127], c[128], c[150], c[184], c[200], c[254]), (0, 42, 97, 126, 128, 152, 189, 207, 255));
        let h = transfer_table(&CssFilter::HIGH_LIGHT);
        assert_eq!((h[50], h[100], h[127], h[150], h[200], h[255]), (26, 66, 87, 106, 145, 188));
    }

    fn test_image(w: usize, h: usize) -> Vec<u8> {
        let mut seed = 7u32;
        (0..w * h * 3)
            .map(|i| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                // Smooth gradients plus noise, like a camera frame.
                (((i / 3) % w) as u32 / 4 + (i / (3 * w)) as u32 / 8 + (seed >> 26)) as u8
            })
            .collect()
    }

    fn compare(a: &[u8], b: &[u8]) -> (f64, i32) {
        let (mut diff, mut max) = (0usize, 0i32);
        for (x, y) in a.iter().zip(b) {
            let d = (*x as i32 - *y as i32).abs();
            diff += (d > 0) as usize;
            max = max.max(d);
        }
        (diff as f64 / a.len() as f64, max)
    }

    #[test]
    fn colour_stage_is_exact() {
        // Every 8-bit colour through the fixed-point tables vs the f32 matrix.
        let rgb: Vec<u8> = (0..1u32 << 24).flat_map(|v| [(v >> 16) as u8, (v >> 8) as u8, v as u8]).collect();
        for filter in [CssFilter::HIGH_LIGHT, CssFilter { blur: 0.0, brightness: 1.38, contrast: 1.12, saturate: 0.78 }] {
            let tables = ColourTables::new(&filter);
            let mut fast = vec![0u8; rgb.len()];
            tables.apply(&rgb, &mut fast);
            let (rate, max) = compare(&fast, &reference_colour(&rgb, &filter));
            assert!(max <= 1 && rate < 1e-4, "{filter:?}: {rate} differ, max {max}");
        }
    }

    #[test]
    fn blur_matches_reference() {
        let (w, h) = (640usize, 480usize);
        let rgb = test_image(w, h);
        let frame = Frame::new(0, w as u32, h as u32, rgb.clone());
        for sigma in [0.42, 0.65] {
            // brightness 1 contrast 1 saturate 1: the colour stage is the identity, leaving the blur.
            let filter = CssFilter { blur: sigma, brightness: 1.0, contrast: 1.0, saturate: 1.0 };
            let mut light = LightFilter::new();
            let fast = light.render(&frame, &filter).rgb.clone();
            let (rate, max) = compare(&fast, &reference_filter(&rgb, w, h, &filter));
            assert!(max <= 1 && rate < 5e-3, "blur {sigma}: {rate} differ, max {max}");
        }
    }

    #[test]
    fn low_light_brightness_follows_the_meter() {
        let dark = Frame::new(0, 64, 48, vec![20; 64 * 48 * 3]);
        let mut light = LightFilter::new();
        let mut last = 0.0;
        for _ in 0..60 {
            let (filter, luminance) = light.next_filter(&dark, LightMode::Low).unwrap();
            assert!((luminance - 20.0 / 255.0).abs() < 1e-9);
            last = filter.brightness;
        }
        // 0.42 / max(0.08, 0.078) = 5.25, capped at 2.45.
        assert_eq!(last, 2.45);
        assert_eq!(light.next_filter(&dark, LightMode::Low).unwrap().0.blur, 0.65);
    }
}
