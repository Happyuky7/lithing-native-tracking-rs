//! Frame sources: a live camera (nokhwa: Media Foundation on Windows, V4L2
//! on Linux) or a video file (.y4m read directly, anything else decoded by
//! an `ffmpeg` child process into a y4m pipe).

use crate::frame::{Frame, FramePool};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("video format: {0}")]
    Format(String),
    #[error("camera: {0}")]
    Camera(String),
    /// A live source had no new frame in time (a pushed stream between frames); the
    /// capture loop simply asks again. Not an error worth counting.
    #[error("no frame yet")]
    NoFrame,
}

pub type Result<T> = std::result::Result<T, SourceError>;

/// What a source actually delivers (a camera may not honour the request).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    /// Nominal frame rate (the camera's negotiated rate, the file's rate).
    pub fps: f32,
    pub description: String,
    /// Pixel format the device delivers (camera) or the container's (file).
    pub format: String,
    /// Why the source differs from what was asked (a camera mode the device
    /// rejected, opened in Auto instead). None when it opened as requested.
    #[cfg_attr(feature = "serde", serde(default))]
    pub warning: Option<String>,
}

/// One camera mode: pixel format, resolution and frame rate (`MJPEG:1280x720@30`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct CameraMode {
    /// "MJPEG", "YUYV", "NV12", "GRAY", "RAWRGB", "RAWBGR"; None = any format with this size and rate.
    #[cfg_attr(feature = "serde", serde(default))]
    pub format: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl CameraMode {
    /// Parses `MJPEG:1280x720@30`, `1280x720@30` or `yuyv:640x480@60`.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (format, rest) = match text.split_once(':') {
            Some((format, rest)) => (Some(format.trim().to_ascii_uppercase()), rest),
            None => (None, text),
        };
        let (size, fps) = rest.split_once('@')?;
        let (width, height) = size.trim().split_once(['x', 'X'])?;
        let mode = Self { format: format.filter(|f| !f.is_empty()), width: width.trim().parse().ok()?, height: height.trim().parse().ok()?, fps: fps.trim().parse().ok()? };
        (mode.width > 0 && mode.height > 0 && mode.fps > 0).then_some(mode)
    }
}

impl std::fmt::Display for CameraMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(format) = &self.format {
            write!(f, "{format}:")?;
        }
        write!(f, "{}x{}@{}", self.width, self.height, self.fps)
    }
}

pub trait FrameSource: Send {
    fn info(&self) -> SourceInfo;
    /// Reads the next frame into a pooled buffer; `None` at the end of a file.
    fn next_frame(&mut self, pool: &FramePool, id: u64) -> Result<Option<Frame>>;
    /// Whether frames arrive at their own pace (a camera) rather than on demand (a file).
    fn is_live(&self) -> bool;
    /// The "Keep frame rate" state of a camera (None for files). Take it
    /// before handing the source to the tracker.
    fn camera_controls(&self) -> Option<CameraControlsHandle> {
        None
    }
    /// The modes the camera listed when it was opened (None for files).
    fn camera_modes(&self) -> Option<Vec<CameraMode>> {
        None
    }
}

/// Shared view of a camera's controls while it streams: the report (controls
/// found, changes, delivered fps before / after) and the live on/off switch.
#[derive(Clone, Debug)]
pub struct CameraControlsHandle {
    report: std::sync::Arc<std::sync::Mutex<lithing_camera_controls::CameraControlReport>>,
    keep: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CameraControlsHandle {
    pub fn new(keep_frame_rate: bool) -> Self {
        Self { report: Default::default(), keep: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(keep_frame_rate)) }
    }

    pub fn report(&self) -> lithing_camera_controls::CameraControlReport {
        self.report.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn publish(&self, report: &lithing_camera_controls::CameraControlReport) {
        *self.report.lock().unwrap_or_else(|e| e.into_inner()) = report.clone();
    }

    /// Turns "Keep frame rate" on or off; the camera thread applies or restores within a second.
    pub fn set_keep_frame_rate(&self, enabled: bool) {
        self.keep.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn keep_frame_rate(&self) -> bool {
        self.keep.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// YUV 4:2:0 to RGB8, BT.601, limited or full range (fixed point).
pub fn yuv420_to_rgb(y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], width: usize, height: usize, full_range: bool, out: &mut [u8]) {
    let chroma_width = width.div_ceil(2);
    let (y_offset, y_scale) = if full_range { (0, 256) } else { (16, 298) };
    // Coefficients * 256 (limited: 409, -100, -208, 516; full: 359, -88, -183, 454).
    let (rv, gu, gv, bu) = if full_range { (359, -88, -183, 454) } else { (409, -100, -208, 516) };
    for row in 0..height {
        let y_row = &y_plane[row * width..(row + 1) * width];
        let chroma_row = (row / 2) * chroma_width;
        let out_row = &mut out[row * width * 3..(row + 1) * width * 3];
        for col in 0..width {
            let c = y_scale * (y_row[col] as i32 - y_offset);
            let d = u_plane[chroma_row + col / 2] as i32 - 128;
            let e = v_plane[chroma_row + col / 2] as i32 - 128;
            let o = col * 3;
            out_row[o] = ((c + rv * e + 128) >> 8).clamp(0, 255) as u8;
            out_row[o + 1] = ((c + gu * d + gv * e + 128) >> 8).clamp(0, 255) as u8;
            out_row[o + 2] = ((c + bu * d + 128) >> 8).clamp(0, 255) as u8;
        }
    }
}

/// A YUV4MPEG2 (4:2:0) stream: a .y4m file or ffmpeg's `-f yuv4mpegpipe` output.
pub struct Y4mSource {
    reader: Box<dyn BufRead + Send>,
    child: Option<Child>,
    width: u32,
    height: u32,
    fps: f32,
    full_range: bool,
    planes: Vec<u8>,
    description: String,
    /// Loop the file when it ends (benchmarks of a fixed length).
    pub looping: Option<PathBuf>,
}

impl Y4mSource {
    /// Opens a video file: .y4m directly, other formats through `ffmpeg` (which must be on PATH).
    pub fn open(path: &Path) -> Result<Self> {
        let is_y4m = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("y4m"));
        if is_y4m {
            let file = std::fs::File::open(path)?;
            Self::from_reader(Box::new(BufReader::with_capacity(1 << 20, file)), None, path.display().to_string())
        } else {
            let mut child = Command::new("ffmpeg")
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(path)
                .args(["-an", "-f", "yuv4mpegpipe", "-pix_fmt", "yuv420p", "-strict", "-1", "-"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| SourceError::Format(format!("could not start ffmpeg for {}: {e}", path.display())))?;
            let stdout = child.stdout.take().ok_or_else(|| SourceError::Format("ffmpeg has no stdout".into()))?;
            Self::from_reader(Box::new(BufReader::with_capacity(1 << 20, stdout)), Some(child), format!("{} (ffmpeg)", path.display()))
        }
    }

    pub fn from_reader(mut reader: Box<dyn BufRead + Send>, child: Option<Child>, description: String) -> Result<Self> {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if !header.starts_with("YUV4MPEG2") {
            return Err(SourceError::Format("not a YUV4MPEG2 stream".into()));
        }
        let (mut width, mut height, mut fps, mut full_range) = (0, 0, 30.0, false);
        for token in header.split_whitespace().skip(1) {
            let (tag, value) = token.split_at(1);
            match tag {
                "W" => width = value.parse().unwrap_or(0),
                "H" => height = value.parse().unwrap_or(0),
                "F" => {
                    if let Some((n, d)) = value.split_once(':') {
                        let (n, d): (f32, f32) = (n.parse().unwrap_or(30.0), d.parse().unwrap_or(1.0));
                        if d > 0.0 {
                            fps = n / d;
                        }
                    }
                }
                "C" if !value.starts_with("420") && !value.starts_with("mono") => return Err(SourceError::Format(format!("unsupported chroma {value} (need 4:2:0)"))),
                "X" if value.eq_ignore_ascii_case("COLORRANGE=FULL") => full_range = true,
                _ => {}
            }
        }
        if width == 0 || height == 0 {
            return Err(SourceError::Format("missing size in the y4m header".into()));
        }
        let planes = vec![0; width as usize * height as usize + 2 * (width as usize).div_ceil(2) * (height as usize).div_ceil(2)];
        Ok(Self { reader, child, width, height, fps, full_range, planes, description, looping: None })
    }

    fn reopen(&mut self) -> Result<bool> {
        let Some(path) = self.looping.clone() else { return Ok(false) };
        let mut again = Self::open(&path)?;
        again.looping = Some(path);
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *self = again;
        Ok(true)
    }
}

impl Drop for Y4mSource {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl FrameSource for Y4mSource {
    fn info(&self) -> SourceInfo {
        SourceInfo { width: self.width, height: self.height, fps: self.fps, description: self.description.clone(), format: if self.full_range { "yuv420p (full)".into() } else { "yuv420p".into() }, warning: None }
    }

    fn is_live(&self) -> bool {
        false
    }

    fn next_frame(&mut self, pool: &FramePool, id: u64) -> Result<Option<Frame>> {
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            if self.reopen()? {
                return self.next_frame(pool, id);
            }
            return Ok(None);
        }
        if !line.starts_with("FRAME") {
            return Err(SourceError::Format(format!("expected FRAME, got {:?}", line.chars().take(16).collect::<String>())));
        }
        match self.reader.read_exact(&mut self.planes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let luma = w * h;
        let chroma = w.div_ceil(2) * h.div_ceil(2);
        let mut frame = pool.take(id, self.width, self.height);
        yuv420_to_rgb(&self.planes[..luma], &self.planes[luma..luma + chroma], &self.planes[luma + chroma..], w, h, self.full_range, &mut frame.rgb);
        Ok(Some(frame))
    }
}

/// Packed RGB24 frames back to back (`ffmpeg -f rawvideo -pix_fmt rgb24`):
/// exactly the pixels a PNG sequence decodes to, for comparisons with the
/// web reference that must not go through YUV.
pub struct RawRgbSource {
    reader: Box<dyn Read + Send>,
    width: u32,
    height: u32,
    fps: f32,
    description: String,
}

impl RawRgbSource {
    pub fn open(path: &Path, width: u32, height: u32, fps: f32) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self { reader: Box::new(BufReader::with_capacity(1 << 20, file)), width, height, fps, description: format!("{} (rgb24 {width}x{height})", path.display()) })
    }
}

impl FrameSource for RawRgbSource {
    fn info(&self) -> SourceInfo {
        SourceInfo { width: self.width, height: self.height, fps: self.fps, description: self.description.clone(), format: "rgb24".into(), warning: None }
    }

    fn is_live(&self) -> bool {
        false
    }

    fn next_frame(&mut self, pool: &FramePool, id: u64) -> Result<Option<Frame>> {
        let mut frame = pool.take(id, self.width, self.height);
        match self.reader.read_exact(&mut frame.rgb) {
            Ok(()) => Ok(Some(frame)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Camera request: the web app asks for 640x480 (1280x720 for precise hands) at 60 fps ideal.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct CameraRequest {
    pub index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// "Keep frame rate" (default on): stop the camera from lowering its
    /// frame rate in low light (auto-exposure priority off, else an exposure
    /// cap with more gain); every control is restored when the camera closes.
    pub keep_frame_rate: bool,
    /// An explicit camera mode (the app's Camera mode list); None = Auto: the
    /// best of the device's formats for width / height / fps (best_camera_format).
    /// A mode the device rejects falls back to Auto with `SourceInfo::warning`.
    pub mode: Option<CameraMode>,
}

impl Default for CameraRequest {
    fn default() -> Self {
        Self { index: 0, width: 640, height: 480, fps: 60, keep_frame_rate: true, mode: None }
    }
}

/// Names of the cameras (does not open any of them).
#[cfg(feature = "camera")]
pub fn list_cameras() -> Result<Vec<String>> {
    let backend = nokhwa::native_api_backend().ok_or_else(|| SourceError::Camera("no camera backend on this platform".into()))?;
    let cameras = nokhwa::query(backend).map_err(|e| SourceError::Camera(e.to_string()))?;
    Ok(cameras.iter().map(|c| format!("{}: {} ({})", c.index(), c.human_name(), c.description())).collect())
}

/// The formats each camera offers (format, resolution, fps), read by opening the device
/// without starting a stream. Sorted by format, resolution and fps.
#[cfg(feature = "camera")]
pub fn list_camera_formats() -> Result<Vec<(String, Vec<String>)>> {
    let backend = nokhwa::native_api_backend().ok_or_else(|| SourceError::Camera("no camera backend on this platform".into()))?;
    let cameras = nokhwa::query(backend).map_err(|e| SourceError::Camera(e.to_string()))?;
    let mut out = Vec::new();
    for info in cameras {
        let name = format!("{}: {}", info.index(), info.human_name());
        let formats = match device_formats(info.index()) {
            Ok(list) => to_modes(&list).iter().map(|m| format!("{} {}x{} @{}", m.format.as_deref().unwrap_or("?"), m.width, m.height, m.fps)).collect(),
            Err(e) => vec![format!("could not read formats: {e}")],
        };
        out.push((name, formats));
    }
    Ok(out)
}

/// The modes camera `index` offers (format, resolution, fps; sorted, no duplicates), read by
/// opening the device without starting a stream.
#[cfg(feature = "camera")]
pub fn camera_modes(index: u32) -> Result<Vec<CameraMode>> {
    Ok(to_modes(&device_formats(&nokhwa::utils::CameraIndex::Index(index))?))
}

#[cfg(feature = "camera")]
fn device_formats(index: &nokhwa::utils::CameraIndex) -> Result<Vec<nokhwa::utils::CameraFormat>> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{RequestedFormat, RequestedFormatType};
    let error = |e: nokhwa::NokhwaError| SourceError::Camera(e.to_string());
    let mut camera = nokhwa::Camera::new(index.clone(), RequestedFormat::new::<RgbFormat>(RequestedFormatType::None)).map_err(error)?;
    camera.compatible_camera_formats().map_err(error)
}

#[cfg(feature = "camera")]
fn to_modes(formats: &[nokhwa::utils::CameraFormat]) -> Vec<CameraMode> {
    let mut modes: Vec<CameraMode> = formats
        .iter()
        .map(|f| CameraMode { format: Some(f.format().to_string()), width: f.resolution().width(), height: f.resolution().height(), fps: f.frame_rate() })
        .collect();
    modes.sort_by(|a, b| (&a.format, a.width, a.height, a.fps).cmp(&(&b.format, b.width, b.height, b.fps)));
    modes.dedup();
    modes
}

/// A camera read on its own thread (nokhwa's camera handle is not `Send`,
/// so it lives and dies on that thread); frames arrive through a channel.
#[cfg(feature = "camera")]
pub struct CameraSource {
    frames: crossbeam_channel::Receiver<Result<Frame>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    info: SourceInfo,
    controls: CameraControlsHandle,
    modes: Vec<CameraMode>,
}

/// The format to ask for in Auto (see open_camera), from the formats the device lists.
#[cfg(feature = "camera")]
fn best_camera_format(formats: &[nokhwa::utils::CameraFormat], request: &CameraRequest) -> Option<nokhwa::utils::CameraFormat> {
    use nokhwa::utils::FrameFormat;
    let rank = |format: FrameFormat| match format { FrameFormat::MJPEG => 0, FrameFormat::YUYV => 1, FrameFormat::NV12 => 2, _ => 3 };
    let wanted = (request.width as i64, request.height as i64);
    formats
        .iter()
        .copied()
        // A usable live rate: below 15 fps is a slideshow, whatever the format.
        .filter(|f| f.frame_rate() >= 15 && rank(f.format()) < 3)
        .min_by_key(|f| {
            let (w, h) = (f.resolution().width() as i64, f.resolution().height() as i64);
            let distance = (w - wanted.0).pow(2) + (h - wanted.1).pow(2);
            let fps = f.frame_rate().min(request.fps) as i64;
            (rank(f.format()), distance, -fps)
        })
}

/// The device format for an explicit mode: its exact format, else (no format given) the
/// listed format of that size and rate that decodes best (MJPEG, YUYV, NV12, ...).
#[cfg(feature = "camera")]
fn explicit_camera_format(formats: &[nokhwa::utils::CameraFormat], mode: &CameraMode) -> Option<nokhwa::utils::CameraFormat> {
    use nokhwa::utils::{CameraFormat, FrameFormat, Resolution};
    let resolution = Resolution::new(mode.width, mode.height);
    if let Some(name) = &mode.format {
        let format: FrameFormat = name.parse().ok()?;
        return Some(CameraFormat::new(resolution, format, mode.fps));
    }
    let rank = |format: FrameFormat| match format { FrameFormat::MJPEG => 0, FrameFormat::YUYV => 1, FrameFormat::NV12 => 2, _ => 3 };
    formats.iter().copied().filter(|f| f.resolution() == resolution && f.frame_rate() == mode.fps).min_by_key(|f| rank(f.format()))
}

#[cfg(feature = "camera")]
fn open_camera(request: &CameraRequest) -> Result<(nokhwa::Camera, SourceInfo, String, Vec<CameraMode>)> {
    use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormatType, Resolution};
    let index = CameraIndex::Index(request.index);
    let formats = device_formats(&index).unwrap_or_default();
    // An explicit mode first, exactly as listed; if the device rejects it, Auto below with
    // a warning in SourceInfo (the app shows it; the camera still opens).
    let mut warning = None;
    if let Some(mode) = &request.mode {
        match explicit_camera_format(&formats, mode) {
            Some(exact) => match open_stream(&index, RequestedFormatType::Exact(exact)) {
                Ok((camera, mut info, id)) => {
                    let got = camera.camera_format();
                    if got != exact {
                        info.warning = Some(format!("camera mode {mode} not applied, the camera opened {}:{}x{}@{}", got.format(), got.resolution().width(), got.resolution().height(), got.frame_rate()));
                    }
                    return Ok((camera, info, id, to_modes(&formats)));
                }
                Err(e) => warning = Some(format!("camera mode {mode} rejected ({e}); using Auto")),
            },
            None => warning = Some(format!("camera mode {mode} is not offered by this camera; using Auto")),
        }
    }
    // Pick the format ourselves: nokhwa's `Closest` only looks at frame rates of the exact
    // requested resolution and, on some Media Foundation drivers, settled on a 1 fps mode
    // (a webcam then delivered ~11 fps and keep-fps saw nothing to fix). Prefer MJPEG (what
    // webcams offer at 30/60 fps), then YUYV, then anything decodable; within a format the
    // resolution closest to the request, then the highest frame rate up to the request.
    let mut attempts = Vec::new();
    if let Some(best) = best_camera_format(&formats, request) {
        attempts.push(RequestedFormatType::Exact(best));
    }
    attempts.extend([
        RequestedFormatType::Closest(CameraFormat::new(Resolution::new(request.width, request.height), FrameFormat::MJPEG, request.fps)),
        RequestedFormatType::Closest(CameraFormat::new(Resolution::new(request.width, request.height), FrameFormat::YUYV, request.fps)),
        RequestedFormatType::HighestFrameRate(request.fps),
        RequestedFormatType::None,
    ]);
    let mut last = None;
    for kind in attempts {
        match open_stream(&index, kind) {
            Ok((camera, mut info, id)) => {
                info.warning = warning;
                return Ok((camera, info, id, to_modes(&formats)));
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| SourceError::Camera("no camera".into())))
}

/// Opens camera `index` with one format request and starts its stream.
#[cfg(feature = "camera")]
fn open_stream(index: &nokhwa::utils::CameraIndex, kind: nokhwa::utils::RequestedFormatType) -> Result<(nokhwa::Camera, SourceInfo, String)> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::RequestedFormat;
    let error = |e: nokhwa::NokhwaError| SourceError::Camera(e.to_string());
    let mut camera = nokhwa::Camera::new(index.clone(), RequestedFormat::new::<RgbFormat>(kind)).map_err(error)?;
    camera.open_stream().map_err(error)?;
    let format = camera.camera_format();
    let info = SourceInfo {
        width: format.resolution().width(),
        height: format.resolution().height(),
        fps: format.frame_rate() as f32,
        description: camera.info().human_name(),
        format: format!("{:?}", format.format()),
        warning: None,
    };
    // Symbolic link on Windows (Media Foundation), device description elsewhere.
    let id = camera.info().misc();
    Ok((camera, info, id))
}

#[cfg(feature = "camera")]
impl CameraSource {
    /// Opens a camera asking for the closest to `request` (60 fps ideal, with
    /// fallbacks); `info()` reports what the device actually negotiated.
    pub fn open(request: &CameraRequest) -> Result<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let request = request.clone();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let (info_tx, info_rx) = crossbeam_channel::bounded::<Result<(SourceInfo, Vec<CameraMode>)>>(1);
        // Two frames in flight at most: the tracker keeps only the newest anyway.
        let (frame_tx, frame_rx) = crossbeam_channel::bounded::<Result<Frame>>(2);
        let thread_stop = stop.clone();
        let controls = CameraControlsHandle::new(request.keep_frame_rate);
        let thread_controls = controls.clone();
        let thread = std::thread::Builder::new()
            .name("lithing-camera".into())
            .spawn(move || {
                use nokhwa::pixel_format::RgbFormat;
                let (mut camera, info, id, modes) = match open_camera(&request) {
                    Ok(opened) => opened,
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return;
                    }
                };
                let mut keep = KeepFps::start(request.index, &info, &id, thread_controls);
                let _ = info_tx.send(Ok((info, modes)));
                let pool = FramePool::new(4);
                let mut id = 0;
                while !thread_stop.load(Ordering::Relaxed) {
                    let frame = camera.frame().map_err(|e| SourceError::Camera(e.to_string())).and_then(|buffer| {
                        let resolution = buffer.resolution();
                        let mut frame = pool.take(id, resolution.width(), resolution.height());
                        buffer.decode_image_to_buffer::<RgbFormat>(&mut frame.rgb).map_err(|e| SourceError::Camera(e.to_string()))?;
                        Ok(frame)
                    });
                    if let Ok(frame) = &frame {
                        keep.frame(frame);
                    }
                    id += 1;
                    // Full: the reader is behind; drop this frame rather than queue it.
                    if let Err(crossbeam_channel::TrySendError::Disconnected(_)) = frame_tx.try_send(frame) {
                        break;
                    }
                }
                // Controls back to what the camera had, while the stream still runs.
                keep.finish();
                let _ = camera.stop_stream();
            })
            .map_err(SourceError::Io)?;
        let (info, modes) = info_rx.recv().map_err(|_| SourceError::Camera("camera thread ended".into()))??;
        Ok(Self { frames: frame_rx, stop, thread: Some(thread), info, controls, modes })
    }
}

/// The camera thread's side of "Keep frame rate": one-second delivered-rate
/// windows and the frame luma feed [`lithing_camera_controls::KeepFrameRate`].
#[cfg(feature = "camera")]
struct KeepFps {
    controls: Option<Box<dyn lithing_camera_controls::CameraControls>>,
    policy: lithing_camera_controls::KeepFrameRate,
    handle: CameraControlsHandle,
    opened: std::time::Instant,
    window_start: Option<std::time::Instant>,
    frames: u32,
    luma_sum: f64,
    luma_frames: u32,
}

#[cfg(feature = "camera")]
impl KeepFps {
    fn start(index: u32, info: &SourceInfo, id: &str, handle: CameraControlsHandle) -> Self {
        use lithing_camera_controls as cc;
        let enabled = handle.keep_frame_rate();
        let mut policy = cc::KeepFrameRate::new(&info.description, info.fps, enabled, cc::KeepFrameRateConfig::default());
        let mut controls = None;
        if cc::is_virtual(&info.description, id) {
            policy.report.virtual_camera = true;
            policy.attach(None, Some("virtual camera: no hardware controls, nothing changed".into()));
        } else {
            match cc::open(index, id) {
                Ok(opened) => controls = Some(opened),
                Err(e) => policy.attach(None, Some(e)),
            }
            if let Some(c) = controls.as_deref_mut() {
                policy.attach(Some(c), None);
            }
        }
        handle.publish(&policy.report);
        Self { controls, policy, handle, opened: std::time::Instant::now(), window_start: None, frames: 0, luma_sum: 0.0, luma_frames: 0 }
    }

    fn frame(&mut self, frame: &Frame) {
        let now = std::time::Instant::now();
        let start = *self.window_start.get_or_insert(now);
        self.frames += 1;
        // Every 4th frame, a sparse grid of the green channel (luma enough for steering gain).
        if self.frames % 4 == 1 {
            self.luma_sum += sparse_luma(&frame.rgb, frame.width as usize, frame.height as usize) as f64;
            self.luma_frames += 1;
        }
        let elapsed = now.duration_since(start).as_secs_f32();
        if elapsed < 1.0 {
            return;
        }
        // The first frame opens the window: frames - 1 intervals in `elapsed`.
        let fps = (self.frames - 1).max(1) as f32 / elapsed;
        let luma = (self.luma_frames > 0).then(|| (self.luma_sum / self.luma_frames as f64) as f32);
        let seconds = now.duration_since(self.opened).as_secs_f64();
        let wanted = self.handle.keep_frame_rate();
        if wanted != self.policy.enabled() {
            self.policy.set_enabled(self.controls.as_deref_mut().map(|c| c as &mut dyn lithing_camera_controls::CameraControls), wanted, seconds);
        }
        self.policy.window(self.controls.as_deref_mut().map(|c| c as &mut dyn lithing_camera_controls::CameraControls), seconds, fps, luma);
        self.handle.publish(&self.policy.report);
        self.window_start = Some(now);
        self.frames = 1;
        self.luma_sum = 0.0;
        self.luma_frames = 0;
    }

    fn finish(&mut self) {
        if let Some(c) = self.controls.as_deref_mut() {
            self.policy.restore(c);
        }
        self.handle.publish(&self.policy.report);
        self.controls = None;
    }
}

/// Mean green value of every 8th pixel of every 8th row.
#[cfg(feature = "camera")]
fn sparse_luma(rgb: &[u8], width: usize, height: usize) -> f32 {
    let (mut sum, mut count) = (0u64, 0u64);
    for row in (0..height).step_by(8) {
        let line = &rgb[row * width * 3..(row + 1) * width * 3];
        for col in (0..width).step_by(8) {
            sum += line[col * 3 + 1] as u64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum as f32 / count as f32 }
}

#[cfg(feature = "camera")]
impl FrameSource for CameraSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn is_live(&self) -> bool {
        true
    }

    fn camera_controls(&self) -> Option<CameraControlsHandle> {
        Some(self.controls.clone())
    }

    fn camera_modes(&self) -> Option<Vec<CameraMode>> {
        Some(self.modes.clone())
    }

    fn next_frame(&mut self, _pool: &FramePool, id: u64) -> Result<Option<Frame>> {
        match self.frames.recv() {
            Ok(Ok(mut frame)) => {
                frame.id = id;
                Ok(Some(frame))
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(feature = "camera")]
impl Drop for CameraSource {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        // Unblock a full channel, then wait for the camera to close.
        while self.frames.try_recv().is_ok() {}
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "camera")]
    #[test]
    fn sparse_luma_of_a_flat_frame() {
        let (w, h) = (64, 48);
        let rgb: Vec<u8> = (0..w * h).flat_map(|_| [10u8, 100, 200]).collect();
        assert_eq!(sparse_luma(&rgb, w, h), 100.0);
        assert_eq!(sparse_luma(&[], 0, 0), 0.0);
    }

    #[test]
    fn camera_request_keeps_the_frame_rate_by_default() {
        assert!(CameraRequest::default().keep_frame_rate);
        assert!(CameraRequest::default().mode.is_none());
    }

    #[test]
    fn camera_mode_parse_and_display() {
        let mode = CameraMode::parse("MJPEG:1280x720@30").unwrap();
        assert_eq!(mode, CameraMode { format: Some("MJPEG".into()), width: 1280, height: 720, fps: 30 });
        assert_eq!(mode.to_string(), "MJPEG:1280x720@30");
        assert_eq!(CameraMode::parse("yuyv:640X480@60").unwrap().format.as_deref(), Some("YUYV"));
        let any = CameraMode::parse("640x480@15").unwrap();
        assert_eq!((any.format, any.width, any.height, any.fps), (None, 640, 480, 15));
        assert!(CameraMode::parse("640x480").is_none());
        assert!(CameraMode::parse("MJPEG:0x480@30").is_none());
    }

    #[test]
    fn y4m_header_and_frame() {
        let mut data = b"YUV4MPEG2 W4 H2 F30:1 Ip C420jpeg\nFRAME\n".to_vec();
        data.extend([235u8; 8]); // white luma (limited range)
        data.extend([128u8; 2]);
        data.extend([128u8; 2]);
        let mut source = Y4mSource::from_reader(Box::new(std::io::Cursor::new(data)), None, "test".into()).unwrap();
        assert_eq!((source.info().width, source.info().height, source.info().fps), (4, 2, 30.0));
        let pool = FramePool::new(1);
        let frame = source.next_frame(&pool, 0).unwrap().unwrap();
        assert!(frame.rgb.iter().all(|&v| v >= 254));
        assert!(source.next_frame(&pool, 1).unwrap().is_none());
    }
}
