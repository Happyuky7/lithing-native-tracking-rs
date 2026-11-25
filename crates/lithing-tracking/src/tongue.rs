//! Port of src/app/tongue-detector.ts: tongue detection from the mouth
//! pixels, since MediaPipe has no usable tongue score. An upright 56x56 crop
//! around the mouth is sampled straight from the camera frame (no canvas),
//! white balanced on the user's own skin (blurred first in dim light), split
//! per column by the lip contours and judged by redness against the lips. A
//! tongue counts when one continuous run of red climbs from the lower lip
//! with red in the opening, or reaches past the lips; a wide open mouth (a
//! yawn) raises the bar. See the web file for the full rationale.

use crate::frame::Frame;
use crate::types::TongueReading;

pub const TONGUE_CROP: usize = 56;
const HALF_WIDTH: f32 = 0.95;
const ABOVE: f32 = 0.7;
const SCALE: f32 = TONGUE_CROP as f32 / (HALF_WIDTH * 2.0);
/// Below this crop lightness (skin mean, 0-1) the crop is blurred against sensor noise.
const DIM_LIGHT: f32 = 0.3;

pub const MASK_NONE: u8 = 0;
pub const MASK_SKIN: u8 = 1;
pub const MASK_LIP: u8 = 2;
pub const MASK_OPENING: u8 = 3;
pub const MASK_TONGUE: u8 = 4;
pub const MASK_TONGUE_INSIDE: u8 = 5;

const UPPER_OUTER: [usize; 11] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
const UPPER_INNER: [usize; 13] = [61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 291];
const LOWER_INNER: [usize; 13] = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291];
const LOWER_OUTER: [usize; 11] = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct TongueTuning {
    pub sensitivity: f32,
    pub in_level: f32,
    pub out_level: f32,
}

impl Default for TongueTuning {
    fn default() -> Self {
        Self { sensitivity: 1.0, in_level: 0.08, out_level: 0.46 }
    }
}

pub struct TongueDetector {
    pub reading: TongueReading,
    /// The upright mouth crop of the last detection (RGB, 0..255) and what each pixel was taken for (MASK_*).
    pub crop: Vec<f32>,
    pub mask: Vec<u8>,
    pub frame: u64,
    /// The extension of the last detection before smoothing and the onset check (calibration captures average it).
    pub last_raw: f32,
    blurred: Vec<f32>,
    redness: Vec<f32>,
    lightness: Vec<f32>,
    upper_outer: [f32; TONGUE_CROP],
    upper_inner: [f32; TONGUE_CROP],
    lower_inner: [f32; TONGUE_CROP],
    lower_outer: [f32; TONGUE_CROP],
    extents: [f32; TONGUE_CROP],
    smoothed_raw: f32,
    lip_base: f32,
    smoothed_x: f32,
    smoothed_y: f32,
    streak: u32,
}

impl Default for TongueDetector {
    fn default() -> Self {
        let pixels = TONGUE_CROP * TONGUE_CROP;
        Self {
            reading: TongueReading::default(),
            crop: vec![0.0; pixels * 3],
            mask: vec![0; pixels],
            frame: 0,
            last_raw: 0.0,
            blurred: vec![0.0; pixels * 3],
            redness: vec![0.0; pixels],
            lightness: vec![0.0; pixels],
            upper_outer: [0.0; TONGUE_CROP],
            upper_inner: [0.0; TONGUE_CROP],
            lower_inner: [0.0; TONGUE_CROP],
            lower_outer: [0.0; TONGUE_CROP],
            extents: [0.0; TONGUE_CROP],
            smoothed_raw: 0.0,
            lip_base: 0.2,
            smoothed_x: 0.0,
            smoothed_y: 0.0,
            streak: 0,
        }
    }
}

/// What one crop measured (see `evaluate`).
struct Evaluation {
    contrast: f32,
    light: f32,
    noise: f32,
    run: f32,
    filled: f32,
    total: u32,
    sum_x: f32,
    sum_y: f32,
    upward: u32,
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn trace_contour(indices: &[usize], to_x: &impl Fn(usize) -> f32, to_y: &impl Fn(usize) -> f32, out: &mut [f32; TONGUE_CROP]) {
    let count = indices.len();
    let mut contour = [(0f32, 0f32); 13];
    for (k, &i) in indices.iter().enumerate() {
        contour[k] = (to_x(i), to_y(i));
    }
    let mut segment = 0;
    for (x, slot) in out.iter_mut().enumerate() {
        let x = x as f32;
        while segment < count - 2 && contour[segment + 1].0 < x {
            segment += 1;
        }
        let (x0, y0) = contour[segment];
        let (x1, y1) = contour[segment + 1];
        let t = if x1 - x0 > 1e-3 { ((x - x0) / (x1 - x0)).clamp(0.0, 1.0) } else { 0.0 };
        *slot = y0 + (y1 - y0) * t;
    }
}

/// 3x3 box blur of an RGB crop (edges clamped).
fn box_blur(source: &[f32], out: &mut [f32]) {
    let n = TONGUE_CROP as i32;
    for y in 0..n {
        for x in 0..n {
            let mut rgb = [0f32; 3];
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let o = (((y + dy).clamp(0, n - 1) * n + (x + dx).clamp(0, n - 1)) * 3) as usize;
                    for c in 0..3 {
                        rgb[c] += source[o + c];
                    }
                }
            }
            let o = ((y * n + x) * 3) as usize;
            for c in 0..3 {
                out[o + c] = rgb[c] / 9.0;
            }
        }
    }
}

impl TongueDetector {
    pub fn reset(&mut self) {
        self.smoothed_raw = 0.0;
        self.smoothed_x = 0.0;
        self.smoothed_y = 0.0;
        self.streak = 0;
        let r = &mut self.reading;
        r.out = 0.0;
        r.up = 0.0;
        r.down = 0.0;
        r.left = 0.0;
        r.right = 0.0;
        r.raw = 0.0;
        r.valid = false;
    }

    /// Samples the upright crop: crop (cx, cy) = scale * (u·P, v·P) + (e, f) for image point P.
    #[allow(clippy::too_many_arguments)]
    fn sample_crop(&mut self, frame: &Frame, u: (f32, f32), v: (f32, f32), scale: f32, e: f32, f: f32) {
        let (width, height) = (frame.width as i32, frame.height as i32);
        let stride = frame.width as usize * 3;
        // Supersample when the crop shrinks the image (a canvas draw averages similarly).
        let taps = (1.0 / scale).ceil().clamp(1.0, 4.0) as usize;
        let weight = 1.0 / (taps * taps) as f32;
        for j in 0..TONGUE_CROP {
            for i in 0..TONGUE_CROP {
                let mut rgb = [0f32; 3];
                for sj in 0..taps {
                    for si in 0..taps {
                        let cx = i as f32 + (si as f32 + 0.5) / taps as f32;
                        let cy = j as f32 + (sj as f32 + 0.5) / taps as f32;
                        let a = (cx - e) / scale;
                        let b = (cy - f) / scale;
                        let x = u.0 * a + v.0 * b - 0.5;
                        let y = u.1 * a + v.1 * b - 0.5;
                        let (x0, y0) = (x.floor(), y.floor());
                        let (fx, fy) = (x - x0, y - y0);
                        let (xi, yi) = (x0 as i32, y0 as i32);
                        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                            let px = (xi + dx).clamp(0, width - 1) as usize;
                            let py = (yi + dy).clamp(0, height - 1) as usize;
                            let p = py * stride + px * 3;
                            for c in 0..3 {
                                rgb[c] += frame.rgb[p + c] as f32 * w * weight;
                            }
                        }
                    }
                }
                let o = (j * TONGUE_CROP + i) * 3;
                self.crop[o..o + 3].copy_from_slice(&rgb);
            }
        }
    }

    /// Looks at the mouth in `frame` with the face mesh `points` (normalized).
    /// `mirror` names sides the way the mirrored avatar has them.
    pub fn detect(&mut self, frame: &Frame, points: &[[f32; 3]], mirror: bool, tuning: &TongueTuning) -> TongueReading {
        let (width, height) = (frame.width as f32, frame.height as f32);
        if points.len() < 468 || width <= 0.0 || height <= 0.0 {
            self.reading.valid = false;
            return self.reading;
        }
        let (ax, ay) = (points[61][0] * width, points[61][1] * height);
        let (bx, by) = (points[291][0] * width, points[291][1] * height);
        let mouth_width = (bx - ax).hypot(by - ay);
        if mouth_width < 6.0 {
            self.reading.valid = false;
            return self.reading;
        }
        let (ux, uy) = ((bx - ax) / mouth_width, (by - ay) / mouth_width);
        let (vx, vy) = (-uy, ux);
        let cx = (points[13][0] + points[14][0]) / 2.0 * width;
        let cy = (points[13][1] + points[14][1]) / 2.0 * height;
        let scale = SCALE / mouth_width;
        let offset_x = HALF_WIDTH * SCALE;
        let offset_y = ABOVE * SCALE;
        let e = offset_x - scale * (ux * cx + uy * cy);
        let f = offset_y - scale * (vx * cx + vy * cy);
        self.sample_crop(frame, (ux, uy), (vx, vy), scale, e, f);
        let to_x = |i: usize| scale * (ux * points[i][0] * width + uy * points[i][1] * height) + e;
        let to_y = |i: usize| scale * (vx * points[i][0] * width + vy * points[i][1] * height) + f;
        trace_contour(&UPPER_OUTER, &to_x, &to_y, &mut self.upper_outer);
        trace_contour(&UPPER_INNER, &to_x, &to_y, &mut self.upper_inner);
        trace_contour(&LOWER_INNER, &to_x, &to_y, &mut self.lower_inner);
        trace_contour(&LOWER_OUTER, &to_x, &to_y, &mut self.lower_outer);
        let left = to_x(61).ceil().max(0.0) as i32;
        let right = (to_x(291).floor() as i32).min(TONGUE_CROP as i32 - 1);
        if right - left < 8 {
            self.reading.valid = false;
            return self.reading;
        }
        // Head turned sideways: nose tip (1) off the middle of the face (234 / 454).
        let face_half = (points[454][0] - points[234][0]).abs() * width / 2.0;
        let yaw = if face_half > 1.0 { (points[1][0] * width - (points[454][0] + points[234][0]) * width / 2.0).abs() / face_half } else { 0.0 };

        let Some(evaluation) = self.evaluate(left, right, offset_y) else {
            self.reading.valid = false;
            return self.reading;
        };
        self.reading.contrast = evaluation.contrast;
        self.reading.light = evaluation.light;
        self.reading.noise = evaluation.noise;
        // Without lip/skin contrast, in the dark, through noise, with the head turned far
        // or on a mouth a few pixels wide, the colour test means little: lower the confidence.
        let confidence = clamp01((evaluation.contrast - 0.01 - evaluation.noise * 0.5) / 0.03)
            * clamp01((evaluation.light - 0.012) / 0.03)
            * clamp01((0.9 - yaw) / 0.4)
            * clamp01((mouth_width - 18.0) / 18.0);
        self.reading.confidence = confidence;
        let measured = self.extension(&evaluation) * confidence;
        self.last_raw = measured;
        let mut raw = measured;
        // A tongue stays out for more than one detection: a single frame over the in-level does not count yet.
        if raw > tuning.in_level {
            self.streak += 1;
        } else {
            self.streak = 0;
        }
        if self.streak < 2 {
            raw = raw.min(tuning.in_level);
        }
        self.smoothed_raw += (raw - self.smoothed_raw) * 0.55;
        if evaluation.total > 4 {
            self.smoothed_x += ((evaluation.sum_x / evaluation.total as f32 - offset_x) / SCALE - self.smoothed_x) * 0.5;
            self.smoothed_y += ((evaluation.sum_y / evaluation.total as f32 - offset_y) / SCALE - self.smoothed_y) * 0.5;
        }
        let span01 = (tuning.out_level - tuning.in_level).max(0.03);
        let out = clamp01((self.smoothed_raw - tuning.in_level) / span01 * tuning.sensitivity);
        let reading = &mut self.reading;
        reading.raw = self.smoothed_raw;
        reading.out = out;
        reading.valid = true;
        let side = (self.smoothed_x / 0.35).clamp(-1.0, 1.0) * if mirror { 1.0 } else { -1.0 };
        reading.right = out * clamp01(side);
        reading.left = out * clamp01(-side);
        let up_share = if evaluation.total > 0 { evaluation.upward as f32 / evaluation.total as f32 } else { 0.0 };
        reading.up = out * clamp01((up_share - 0.3) / 0.4);
        reading.down = out * clamp01((self.smoothed_y - 0.55) / 0.4) * (1.0 - up_share);
        self.frame += 1;
        *reading
    }

    /// The colour references and the per-column measurements of one crop; fills the mask.
    /// None when the crop has too little skin or lip.
    fn evaluate(&mut self, left: i32, right: i32, offset_y: f32) -> Option<Evaluation> {
        let n = TONGUE_CROP;
        let span = right - left;
        let (left_f, right_f, span_f) = (left as f32, right as f32, span as f32);
        let (upper_outer, upper_inner, lower_inner, lower_outer) = (self.upper_outer, self.upper_inner, self.lower_inner, self.lower_outer);
        let is_skin = |x: usize, y: usize| {
            let (xf, yf) = (x as f32, y as f32);
            if x as i32 >= left && x as i32 <= right {
                let top = upper_outer[x];
                return yf < top - 0.08 * SCALE && yf > top - 0.3 * SCALE;
            }
            (yf - offset_y).abs() < 0.15 * SCALE && (xf < left_f - 0.12 * SCALE || xf > right_f + 0.12 * SCALE)
        };
        let is_lip = |x: usize, y: usize| {
            let (xf, yf) = (x as f32, y as f32);
            xf > left_f + span_f * 0.2 && xf < right_f - span_f * 0.2 && yf > upper_outer[x] + 1.0 && yf < upper_inner[x] - 1.0
        };
        let (mut skin_light, mut skin_count) = (0f32, 0u32);
        for y in 0..n {
            for x in 0..n {
                if is_skin(x, y) {
                    let o = (y * n + x) * 3;
                    skin_light += self.crop[o] + self.crop[o + 1] + self.crop[o + 2];
                    skin_count += 1;
                }
            }
        }
        if skin_count < 10 {
            return None;
        }
        // Dim light: sensor noise dominates single pixels, so average 3x3 first.
        let light = skin_light / skin_count as f32 / 765.0;
        let rgb: &[f32] = if light < DIM_LIGHT {
            box_blur(&self.crop, &mut self.blurred);
            &self.blurred
        } else {
            &self.crop
        };
        let mask = &mut self.mask;
        mask.fill(MASK_NONE);
        let (mut skin_r, mut skin_g, mut skin_b) = (0f32, 0f32, 0f32);
        for y in 0..n {
            for x in 0..n {
                if is_skin(x, y) {
                    let o = (y * n + x) * 3;
                    skin_r += rgb[o];
                    skin_g += rgb[o + 1];
                    skin_b += rgb[o + 2];
                    mask[y * n + x] = MASK_SKIN;
                }
            }
        }
        let count = skin_count as f32;
        let (skin_r, skin_g, skin_b) = (skin_r / count + 1.0, skin_g / count + 1.0, skin_b / count + 1.0);
        // Balanced on the skin: skin comes out grey (redness 0, lightness 1) under any light colour.
        let (redness, lightness) = (&mut self.redness, &mut self.lightness);
        for index in 0..n * n {
            let o = index * 3;
            let (r, g, b) = (rgb[o] / skin_r, rgb[o + 1] / skin_g, rgb[o + 2] / skin_b);
            let sum = r + g + b + 0.01;
            redness[index] = (r - g) / sum;
            lightness[index] = sum / 3.0;
        }
        let (mut lip_red, mut lip_count, mut skin_variance) = (0f32, 0u32, 0f32);
        for y in 0..n {
            for x in 0..n {
                let index = y * n + x;
                if mask[index] == MASK_SKIN {
                    skin_variance += redness[index] * redness[index];
                } else if x as i32 >= left && x as i32 <= right && is_lip(x, y) {
                    lip_red += redness[index];
                    lip_count += 1;
                    mask[index] = MASK_LIP;
                }
            }
        }
        if lip_count < 6 {
            return None;
        }
        let contrast = lip_red / lip_count as f32;
        let noise = (skin_variance / count).sqrt();
        // Halfway from skin to lip, never closer to skin than this, and above the noise.
        let threshold = 0.025f32.max(contrast * 0.5).max(noise * 1.6);
        // The inside of the mouth is unlit in dim light: only the darkest of it is taken for dark.
        let dark_limit = if light < DIM_LIGHT { 0.28 } else { 0.38 };
        let redness = &*redness;
        let lightness = &*lightness;
        let is_tongue = |index: usize| lightness[index] > dark_limit && redness[index] > threshold;
        // Past the lips the tongue covers skin: nearly as red as the lips, or a warm chin reads as a tongue.
        let outside_threshold = threshold.max(contrast * 0.7);
        let is_tongue_outside = |index: usize| lightness[index] > dark_limit && redness[index] > outside_threshold;
        // Teeth: as light as skin or lighter and not red (balanced on skin, they come out bluish).
        let teeth_limit = (threshold * 0.4).min(contrast * 0.15);
        let is_teeth = |index: usize| lightness[index] > 0.8 && redness[index] < teeth_limit;

        let (mut total, mut sum_x, mut sum_y, mut opening_red, mut opening, mut upward) = (0u32, 0f32, 0f32, 0u32, 0u32, 0u32);
        let (mut central_opening, mut dark, mut teeth, mut run_sum, mut run_columns, mut gap_sum) = (0u32, 0u32, 0u32, 0f32, 0u32, 0f32);
        let margin = (span_f * 0.12).round() as i32;
        let last = n as i32 - 1;
        for x in left..=right {
            let xu = x as usize;
            let xf = x as f32;
            self.extents[xu] = 0.0;
            let central = x >= left + margin && x <= right - margin;
            let middle = xf >= left_f + span_f * 0.3 && xf <= right_f - span_f * 0.3;
            for y in (upper_inner[xu].ceil() as i32).max(0)..=(lower_inner[xu].floor() as i32).min(last) {
                opening += 1;
                let index = y as usize * n + xu;
                let red = is_tongue(index);
                if red {
                    opening_red += 1;
                    mask[index] = MASK_TONGUE_INSIDE;
                } else {
                    mask[index] = MASK_OPENING;
                }
                if middle {
                    central_opening += 1;
                    if lightness[index] <= dark_limit {
                        dark += 1;
                    } else if !red && is_teeth(index) {
                        teeth += 1;
                    }
                }
            }
            if !central {
                continue;
            }
            if middle {
                // One continuous run of tongue colour up from the lower outer lip, broken by dark or teeth.
                let (mut top, mut misses) = (-1i32, 0);
                let mut y = (lower_outer[xu].floor() as i32).min(last);
                while y >= (upper_inner[xu].floor() as i32 - 1).max(0) {
                    let index = y as usize * n + xu;
                    if is_tongue(index) {
                        misses = 0;
                        top = y;
                    } else {
                        misses += 1;
                        if lightness[index] <= dark_limit || is_teeth(index) || misses > 1 {
                            break;
                        }
                    }
                    y -= 1;
                }
                run_sum += if top >= 0 { (lower_outer[xu] - top as f32).max(0.0) } else { 0.0 };
                gap_sum += (lower_inner[xu] - upper_inner[xu]).max(0.0);
                run_columns += 1;
            }
            let lower = lower_outer[xu];
            let (mut reach, mut misses) = (0f32, 0);
            for y in (lower.floor() as i32 - 1).max(0)..n as i32 {
                if is_tongue_outside(y as usize * n + xu) {
                    misses = 0;
                    reach = y as f32 - lower;
                    if y as f32 > lower + 1.0 {
                        mask[y as usize * n + xu] = MASK_TONGUE;
                        total += 1;
                        sum_x += xf;
                        sum_y += y as f32;
                    }
                } else {
                    misses += 1;
                    if misses > 1 {
                        break;
                    }
                }
            }
            let upper = upper_outer[xu];
            let (mut reach_up, mut misses) = (0f32, 0);
            let mut y = (upper.ceil() as i32 + 1).min(last);
            while y >= 0 {
                if is_tongue_outside(y as usize * n + xu) {
                    misses = 0;
                    reach_up = upper - y as f32;
                    if (y as f32) < upper - 1.0 {
                        mask[y as usize * n + xu] = MASK_TONGUE;
                        total += 1;
                        sum_x += xf;
                        sum_y += y as f32;
                        upward += 1;
                    }
                } else {
                    misses += 1;
                    if misses > 1 {
                        break;
                    }
                }
                y -= 1;
            }
            self.extents[xu] = 0f32.max(reach).max(reach_up * 0.9);
        }
        let (mut best, mut second, mut third) = (0f32, 0f32, 0f32);
        for x in (left + margin)..=(right - margin) {
            let value = self.extents[x as usize];
            if value > best {
                third = second;
                second = best;
                best = value;
            } else if value > second {
                third = second;
                second = value;
            } else if value > third {
                third = value;
            }
        }
        let filled = if opening > 0 { opening_red as f32 / opening as f32 } else { 0.0 };
        let open_height = opening as f32 / span_f.max(1.0) / SCALE;
        let run = if run_columns > 0 { run_sum / run_columns as f32 / SCALE } else { 0.0 };
        let reading = &mut self.reading;
        reading.reach = third / SCALE;
        reading.band = run;
        reading.fill = filled * open_height.min(0.4);
        reading.open = if run_columns > 0 { gap_sum / run_columns as f32 / SCALE } else { 0.0 };
        reading.dark = if central_opening > 0 { dark as f32 / central_opening as f32 } else { 0.0 };
        reading.teeth = if central_opening > 0 { teeth as f32 / central_opening as f32 } else { 0.0 };
        Some(Evaluation { contrast, light, noise, run, filled, total, sum_x, sum_y, upward })
    }

    /// The uncalibrated extension of one evaluated crop (mouth widths); see the web file.
    fn extension(&mut self, evaluation: &Evaluation) -> f32 {
        // The continuous red height counts above this user's own lower lip (a slow floor).
        let run = evaluation.run;
        if run < self.lip_base {
            self.lip_base += (run - self.lip_base) * 0.3;
        } else {
            self.lip_base += (run - self.lip_base) * 0.004;
        }
        let reading = &self.reading;
        // Only with red in the opening too (a pout makes a tall red lower lip alone).
        let between = (run - self.lip_base).max(0.0) * clamp01((reading.fill - 0.03) / 0.07);
        // The wider the mouth opens, the less the red inside counts (a yawn shows its own tongue).
        let wide = clamp01((reading.open - 0.5) / 0.3);
        let mut raw = reading.reach + (between + reading.fill * 0.35) * (1.0 - 0.8 * wide);
        // And from half open on, the more of the opening must be tongue rather than dark or teeth.
        let open = clamp01((reading.open - 0.4) / 0.3);
        if open > 0.0 {
            raw *= 1.0 - open * clamp01((reading.dark - 0.15 + reading.teeth * 6.0 - (evaluation.filled - 0.6)) * 2.0);
        }
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_or_tiny_mouths() {
        let frame = Frame::new(0, 64, 64, vec![128; 64 * 64 * 3]);
        let mut detector = TongueDetector::default();
        assert!(!detector.detect(&frame, &[[0.5, 0.5, 0.0]; 10], true, &TongueTuning::default()).valid);
        assert!(!detector.detect(&frame, &vec![[0.5, 0.5, 0.0]; 478], true, &TongueTuning::default()).valid);
    }
}
