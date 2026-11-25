//! A frame source fed from outside: the host pushes camera frames (Android's Camera2
//! through JNI, a test, any capture API nokhwa does not cover) and the tracker pulls
//! them like any other live source.
//!
//! ```
//! use lithing_tracking::push::{push_source, PushInfo};
//! let (source, pusher) = push_source(PushInfo { width: 2, height: 2, fps: 30.0, ..Default::default() });
//! pusher.push_rgb(2, 2, &[0; 12]).unwrap();
//! // Tracker::start(Box::new(source), config) ... pusher.close() ends it.
//! # drop(source);
//! ```
//!
//! The newest pushed frame wins (one slot, like the camera mailbox): a frame not yet
//! taken by the tracker is replaced, never queued. Buffers circulate between the pusher,
//! the source and the tracker's frame pool, so steady-state pushing allocates nothing.

use crate::frame::{Frame, FramePool};
use crate::source::{FrameSource, Result, SourceError, SourceInfo};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// How long `next_frame` waits for a push before handing control back to the capture loop.
const WAIT: Duration = Duration::from_millis(100);

/// What the pushed stream is (the source's [`SourceInfo`]).
#[derive(Clone, Debug, Default)]
pub struct PushInfo {
    /// Size of the frames as the tracker sees them (after rotation).
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub description: String,
    /// Pixel format the host captures ("YUV_420_888", "RGB"...), for status reports.
    pub format: String,
}

/// One YUV 4:2:0 image with arbitrary strides: Android's `YUV_420_888` (I420, NV12 and
/// NV21 layouts all fit: `uv_pixel_stride` 1 or 2, U and V planes may interleave).
#[derive(Clone, Copy, Debug)]
pub struct YuvPlanes<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub y_row_stride: usize,
    pub uv_row_stride: usize,
    pub uv_pixel_stride: usize,
    /// Size of the image as captured (before rotation).
    pub width: u32,
    pub height: u32,
    /// BT.601 full range (JFIF, what Android cameras deliver) or limited range.
    pub full_range: bool,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PushError {
    #[error("the tracker stopped")]
    Closed,
    #[error("frame: {0}")]
    Invalid(String),
}

struct Pending {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Default)]
struct State {
    pending: Option<Pending>,
    /// Buffers to convert into (returned by the tracker's pool or a replaced frame).
    spare: Vec<Vec<u8>>,
    closed: bool,
    pushed: u64,
    replaced: u64,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    ready: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A source and the pusher that feeds it.
pub fn push_source(info: PushInfo) -> (PushSource, FramePusher) {
    let shared = Arc::new(Shared::default());
    (PushSource { shared: shared.clone(), info }, FramePusher { shared })
}

/// The tracker's side: a live [`FrameSource`] that hands out the newest pushed frame.
/// It ends (`next_frame` returns `None`) once the pusher is closed and the last frame taken.
pub struct PushSource {
    shared: Arc<Shared>,
    info: PushInfo,
}

impl FrameSource for PushSource {
    fn info(&self) -> SourceInfo {
        SourceInfo { width: self.info.width, height: self.info.height, fps: self.info.fps, description: self.info.description.clone(), format: self.info.format.clone(), warning: None }
    }

    fn next_frame(&mut self, pool: &FramePool, id: u64) -> Result<Option<Frame>> {
        let mut state = self.shared.lock();
        if state.pending.is_none() && !state.closed {
            state = self.shared.ready.wait_timeout_while(state, WAIT, |s| s.pending.is_none() && !s.closed).unwrap_or_else(|e| e.into_inner()).0;
        }
        let Some(mut pending) = state.pending.take() else {
            return if state.closed { Ok(None) } else { Err(SourceError::NoFrame) };
        };
        drop(state);
        // The pool's buffer goes back to the pusher; the pushed pixels become the frame without a copy.
        let mut frame = pool.take(id, 0, 0);
        std::mem::swap(&mut frame.rgb, &mut pending.rgb);
        frame.width = pending.width;
        frame.height = pending.height;
        let mut state = self.shared.lock();
        if state.spare.len() < 3 {
            state.spare.push(pending.rgb);
        }
        Ok(Some(frame))
    }

    fn is_live(&self) -> bool {
        true
    }
}

impl Drop for PushSource {
    fn drop(&mut self) {
        // The tracker is gone: further pushes fail with `Closed` so the host can stop its camera.
        let mut state = self.shared.lock();
        state.closed = true;
        state.pending = None;
    }
}

/// The host's side. Cheap to clone; every clone feeds the same source.
#[derive(Clone)]
pub struct FramePusher {
    shared: Arc<Shared>,
}

impl FramePusher {
    /// Pushes a tightly packed RGB8 frame.
    pub fn push_rgb(&self, width: u32, height: u32, rgb: &[u8]) -> std::result::Result<(), PushError> {
        let size = width as usize * height as usize * 3;
        if width == 0 || height == 0 || rgb.len() < size {
            return Err(PushError::Invalid(format!("{width}x{height} needs {size} bytes, got {}", rgb.len())));
        }
        self.push_with(width, height, |out| out.copy_from_slice(&rgb[..size]))
    }

    /// Converts a YUV 4:2:0 image to RGB, rotated clockwise by `rotation` degrees (0, 90,
    /// 180, 270: what makes the image upright), and pushes it.
    pub fn push_yuv420(&self, planes: &YuvPlanes<'_>, rotation: u32) -> std::result::Result<(), PushError> {
        check_planes(planes)?;
        let rotation = rotation % 360;
        if rotation % 90 != 0 {
            return Err(PushError::Invalid(format!("rotation {rotation}")));
        }
        let (width, height) = if rotation % 180 == 90 { (planes.height, planes.width) } else { (planes.width, planes.height) };
        self.push_with(width, height, |out| yuv420_to_rgb_rotated(planes, rotation, out))
    }

    fn push_with(&self, width: u32, height: u32, fill: impl FnOnce(&mut [u8])) -> std::result::Result<(), PushError> {
        let mut buffer = {
            let mut state = self.shared.lock();
            if state.closed {
                return Err(PushError::Closed);
            }
            state.spare.pop().unwrap_or_default()
        };
        buffer.resize(width as usize * height as usize * 3, 0);
        // Converted without the lock: the tracker keeps taking frames meanwhile.
        fill(&mut buffer);
        let mut state = self.shared.lock();
        if state.closed {
            return Err(PushError::Closed);
        }
        state.pushed += 1;
        if let Some(old) = state.pending.replace(Pending { rgb: buffer, width, height }) {
            state.replaced += 1;
            if state.spare.len() < 3 {
                state.spare.push(old.rgb);
            }
        }
        drop(state);
        self.shared.ready.notify_one();
        Ok(())
    }

    /// Ends the stream: the tracker processes what it has and its capture stops.
    pub fn close(&self) {
        self.shared.lock().closed = true;
        self.shared.ready.notify_all();
    }

    /// True once closed, or once the tracker dropped its source.
    pub fn is_closed(&self) -> bool {
        self.shared.lock().closed
    }

    /// Frames pushed, and how many of them were replaced before the tracker took them.
    pub fn counts(&self) -> (u64, u64) {
        let state = self.shared.lock();
        (state.pushed, state.replaced)
    }
}

fn check_planes(p: &YuvPlanes<'_>) -> std::result::Result<(), PushError> {
    let (w, h) = (p.width as usize, p.height as usize);
    if w == 0 || h == 0 {
        return Err(PushError::Invalid("empty image".into()));
    }
    if p.y_row_stride < w || p.uv_pixel_stride == 0 {
        return Err(PushError::Invalid(format!("strides y {} uv {}x{} for width {w}", p.y_row_stride, p.uv_row_stride, p.uv_pixel_stride)));
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // Android's plane buffers end right after the last pixel, not after a full row stride.
    let y_needed = p.y_row_stride * (h - 1) + w;
    let uv_needed = p.uv_row_stride * (ch - 1) + p.uv_pixel_stride * (cw - 1) + 1;
    if p.y.len() < y_needed || p.u.len() < uv_needed || p.v.len() < uv_needed {
        return Err(PushError::Invalid(format!("planes too small: y {} (needs {y_needed}), u {}, v {} (need {uv_needed})", p.y.len(), p.u.len(), p.v.len())));
    }
    Ok(())
}

/// YUV 4:2:0 (any strides) to RGB8, BT.601, rotated clockwise by 0 / 90 / 180 / 270 degrees.
/// `out` holds the rotated image (width and height swapped for 90 and 270). The planes
/// must have passed `check_planes`.
fn yuv420_to_rgb_rotated(p: &YuvPlanes<'_>, rotation: u32, out: &mut [u8]) {
    let (w, h) = (p.width as usize, p.height as usize);
    let out_width = if rotation % 180 == 90 { h } else { w };
    let (y_offset, y_scale) = if p.full_range { (0, 256) } else { (16, 298) };
    let (rv, gu, gv, bu) = if p.full_range { (359, -88, -183, 454) } else { (409, -100, -208, 516) };
    let row_step = out_width as isize * 3;
    for row in 0..h {
        // Where source pixel (0, row) lands and how far each next pixel of the row moves.
        let (start, step) = match rotation {
            90 => (((h - 1 - row) * 3) as isize, row_step),
            180 => ((((h - 1 - row) * out_width + (w - 1)) * 3) as isize, -3),
            270 => ((((w - 1) * out_width + row) * 3) as isize, -row_step),
            _ => (((row * out_width) * 3) as isize, 3),
        };
        let y_row = &p.y[row * p.y_row_stride..row * p.y_row_stride + w];
        let uv_row = (row / 2) * p.uv_row_stride;
        let mut o = start;
        for col in 0..w {
            let uv = uv_row + (col / 2) * p.uv_pixel_stride;
            let c = y_scale * (y_row[col] as i32 - y_offset);
            let d = p.u[uv] as i32 - 128;
            let e = p.v[uv] as i32 - 128;
            let i = o as usize;
            out[i] = ((c + rv * e + 128) >> 8).clamp(0, 255) as u8;
            out[i + 1] = ((c + gu * d + gv * e + 128) >> 8).clamp(0, 255) as u8;
            out[i + 2] = ((c + bu * d + 128) >> 8).clamp(0, 255) as u8;
            o += step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::yuv420_to_rgb;

    fn info(width: u32, height: u32) -> PushInfo {
        PushInfo { width, height, fps: 30.0, description: "test".into(), format: "YUV_420_888".into() }
    }

    /// A 4x2 gray-ramp image as I420 planes (stride = width, pixel stride 1).
    fn ramp(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = (0..w * h).map(|i| (i * 255 / (w * h)) as u8).collect();
        let c = w.div_ceil(2) * h.div_ceil(2);
        let u = (0..c).map(|i| (100 + i * 7) as u8).collect();
        let v = (0..c).map(|i| (150 - i * 5) as u8).collect();
        (y, u, v)
    }

    fn pixel(rgb: &[u8], width: usize, x: usize, y: usize) -> [u8; 3] {
        let o = (y * width + x) * 3;
        [rgb[o], rgb[o + 1], rgb[o + 2]]
    }

    #[test]
    fn unrotated_matches_the_y4m_converter() {
        let (w, h) = (6, 4);
        let (y, u, v) = ramp(w, h);
        let mut expected = vec![0; w * h * 3];
        yuv420_to_rgb(&y, &u, &v, w, h, true, &mut expected);
        let planes = YuvPlanes { y: &y, u: &u, v: &v, y_row_stride: w, uv_row_stride: w / 2, uv_pixel_stride: 1, width: w as u32, height: h as u32, full_range: true };
        let mut out = vec![0; w * h * 3];
        yuv420_to_rgb_rotated(&planes, 0, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn rotations_move_pixels_clockwise() {
        let (w, h) = (6, 4);
        let (y, u, v) = ramp(w, h);
        let planes = YuvPlanes { y: &y, u: &u, v: &v, y_row_stride: w, uv_row_stride: w / 2, uv_pixel_stride: 1, width: w as u32, height: h as u32, full_range: true };
        let mut upright = vec![0; w * h * 3];
        yuv420_to_rgb_rotated(&planes, 0, &mut upright);
        for rotation in [90, 180, 270] {
            let mut out = vec![0; w * h * 3];
            yuv420_to_rgb_rotated(&planes, rotation, &mut out);
            for sy in 0..h {
                for sx in 0..w {
                    // Clockwise: 90 sends the top row to the right column.
                    let (dx, dy, dw) = match rotation {
                        90 => (h - 1 - sy, sx, h),
                        180 => (w - 1 - sx, h - 1 - sy, w),
                        _ => (sy, w - 1 - sx, h),
                    };
                    assert_eq!(pixel(&out, dw, dx, dy), pixel(&upright, w, sx, sy), "rotation {rotation} at {sx},{sy}");
                }
            }
        }
    }

    #[test]
    fn padded_and_interleaved_planes() {
        // NV21-like layout as Android delivers it: row stride 8 for a 6-pixel row,
        // V and U interleaved with pixel stride 2, buffers ending after the last pixel.
        let (w, h) = (6, 4);
        let (y, u, v) = ramp(w, h);
        let stride = 8;
        let mut y_padded = vec![0u8; stride * (h - 1) + w];
        for row in 0..h {
            y_padded[row * stride..row * stride + w].copy_from_slice(&y[row * w..(row + 1) * w]);
        }
        let (cw, ch) = (w / 2, h / 2);
        let mut vu = vec![0u8; stride * (ch - 1) + 2 * cw];
        for row in 0..ch {
            for col in 0..cw {
                vu[row * stride + col * 2] = v[row * cw + col];
                vu[row * stride + col * 2 + 1] = u[row * cw + col];
            }
        }
        let planes = YuvPlanes { y: &y_padded, u: &vu[1..], v: &vu[..vu.len() - 1], y_row_stride: stride, uv_row_stride: stride, uv_pixel_stride: 2, width: w as u32, height: h as u32, full_range: true };
        check_planes(&planes).unwrap();
        let mut out = vec![0; w * h * 3];
        yuv420_to_rgb_rotated(&planes, 0, &mut out);
        let mut expected = vec![0; w * h * 3];
        yuv420_to_rgb(&y, &u, &v, w, h, true, &mut expected);
        assert_eq!(out, expected);
    }

    #[test]
    fn short_planes_are_rejected_not_read() {
        let y = vec![0u8; 10];
        let planes = YuvPlanes { y: &y, u: &y, v: &y, y_row_stride: 6, uv_row_stride: 3, uv_pixel_stride: 1, width: 6, height: 4, full_range: true };
        let (_source, pusher) = push_source(info(4, 6));
        assert!(matches!(pusher.push_yuv420(&planes, 90), Err(PushError::Invalid(_))));
    }

    #[test]
    fn newest_frame_wins_and_close_ends_the_source() {
        let (mut source, pusher) = push_source(info(2, 1));
        let pool = FramePool::new(2);
        assert!(matches!(source.next_frame(&pool, 0), Err(SourceError::NoFrame)));
        pusher.push_rgb(2, 1, &[1, 1, 1, 1, 1, 1]).unwrap();
        pusher.push_rgb(2, 1, &[2, 2, 2, 2, 2, 2]).unwrap();
        let frame = source.next_frame(&pool, 7).unwrap().unwrap();
        assert_eq!((frame.id, frame.width, frame.height), (7, 2, 1));
        assert_eq!(frame.rgb, vec![2; 6]);
        assert_eq!(pusher.counts(), (2, 1));
        pusher.close();
        assert!(source.next_frame(&pool, 8).unwrap().is_none());
        assert_eq!(pusher.push_rgb(2, 1, &[0; 6]), Err(PushError::Closed));
    }

    #[test]
    fn dropping_the_source_closes_the_pusher() {
        let (source, pusher) = push_source(info(1, 1));
        drop(source);
        assert!(pusher.is_closed());
    }

    /// The whole pipeline on pushed frames: a still image pushed from another thread
    /// reaches results with its frame ids, and closing the pusher ends the tracker.
    #[test]
    fn tracker_runs_on_pushed_frames() {
        let models = crate::models::default_model_dir();
        if !models.join(crate::models::FACE_LANDMARKS).is_file() {
            eprintln!("skipped: no models in {}", models.display());
            return;
        }
        let (w, h) = (320u32, 240u32);
        let (source, pusher) = push_source(info(w, h));
        let config = crate::TrackerConfig { models_dir: models, inference: crate::InferenceOptions { provider: crate::ExecutionProvider::Cpu, ..Default::default() }, ..Default::default() };
        let tracker = crate::Tracker::start(Box::new(source), config).unwrap();
        let feeder = {
            let pusher = pusher.clone();
            std::thread::spawn(move || {
                let y = vec![128u8; (w * h) as usize];
                let c = vec![128u8; (w * h / 4) as usize];
                // Pushed as a 240x320 portrait sensor image, rotated 90 degrees to 320x240.
                let planes = YuvPlanes { y: &y, u: &c, v: &c, y_row_stride: h as usize, uv_row_stride: h as usize / 2, uv_pixel_stride: 1, width: h, height: w, full_range: true };
                for _ in 0..30 {
                    if pusher.push_yuv420(&planes, 90).is_err() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(33));
                }
                pusher.close();
            })
        };
        feeder.join().unwrap();
        let start = std::time::Instant::now();
        while !tracker.finished() && start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(tracker.finished(), "closing the pusher ends the tracker");
        let captured = tracker.stats.captured.load(std::sync::atomic::Ordering::Relaxed);
        assert!(captured > 0, "frames reached the tracker");
        assert_eq!(tracker.stats.errors.load(std::sync::atomic::Ordering::Relaxed), 0, "waiting for frames is not an error");
    }
}
