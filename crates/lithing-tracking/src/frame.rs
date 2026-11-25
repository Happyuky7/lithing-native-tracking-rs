//! Camera frames: RGB8 pixels with the id and capture time every result carries.

use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One camera frame, tightly packed RGB8 (row stride = `width * 3`).
///
/// Frames are shared as `Arc<Frame>` between the capture thread and the face
/// and hand workers, so a frame is never copied after capture. The pixel
/// buffer goes back to its [`FramePool`] when the last reference drops.
pub struct Frame {
    /// Increases by one per captured frame (a dropped frame leaves a gap).
    pub id: u64,
    /// When the frame was captured (or decoded, for files): latency is measured from here.
    pub captured: Instant,
    /// Presentation time in milliseconds: since capture started for a camera,
    /// frame index / fps for a file (so file runs are deterministic).
    pub pts_ms: f64,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pool: Option<Arc<FramePoolInner>>,
}

impl Frame {
    /// A frame that owns its buffer (tests, one-off images).
    pub fn new(id: u64, width: u32, height: u32, rgb: Vec<u8>) -> Self {
        assert_eq!(rgb.len(), width as usize * height as usize * 3, "RGB buffer size");
        Self { id, captured: Instant::now(), pts_ms: 0.0, width, height, rgb, pool: None }
    }

    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let offset = (y * self.width as usize + x) * 3;
        [self.rgb[offset], self.rgb[offset + 1], self.rgb[offset + 2]]
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let buffer = std::mem::take(&mut self.rgb);
            let mut free = pool.free.lock().unwrap_or_else(|e| e.into_inner());
            if free.len() < pool.capacity {
                free.push(buffer);
            }
        }
    }
}

struct FramePoolInner {
    free: Mutex<Vec<Vec<u8>>>,
    capacity: usize,
}

/// Recycles frame buffers so steady-state capture allocates nothing.
#[derive(Clone)]
pub struct FramePool {
    inner: Arc<FramePoolInner>,
}

impl FramePool {
    pub fn new(capacity: usize) -> Self {
        Self { inner: Arc::new(FramePoolInner { free: Mutex::new(Vec::with_capacity(capacity)), capacity }) }
    }

    /// A frame of the given size whose pixels the caller fills in.
    pub fn take(&self, id: u64, width: u32, height: u32) -> Frame {
        let size = width as usize * height as usize * 3;
        let mut buffer = self.inner.free.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_default();
        buffer.resize(size, 0);
        Frame { id, captured: Instant::now(), pts_ms: 0.0, width, height, rgb: buffer, pool: Some(self.inner.clone()) }
    }
}

/// Holds only the newest frame: a slow consumer skips stale frames instead of
/// queueing them (bounded to one, newest wins).
pub struct LatestFrame {
    slot: Mutex<(Option<Arc<Frame>>, bool)>,
    ready: std::sync::Condvar,
    taken: std::sync::Condvar,
}

impl Default for LatestFrame {
    fn default() -> Self {
        Self { slot: Mutex::new((None, false)), ready: std::sync::Condvar::new(), taken: std::sync::Condvar::new() }
    }
}

impl LatestFrame {
    /// Publishes a frame, replacing (dropping) any frame not yet taken. Returns true when one was dropped.
    pub fn publish(&self, frame: Arc<Frame>) -> bool {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let dropped = slot.0.replace(frame).is_some();
        self.ready.notify_one();
        dropped
    }

    /// Marks the source as finished; waiting consumers wake up and get `None`.
    pub fn close(&self) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.1 = true;
        self.ready.notify_all();
    }

    /// Waits for the next frame (up to `timeout`). `None` on timeout or when closed and empty.
    pub fn take(&self, timeout: std::time::Duration) -> Option<Arc<Frame>> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(frame) = slot.0.take() {
                self.taken.notify_all();
                return Some(frame);
            }
            if slot.1 {
                return None;
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            slot = self.ready.wait_timeout(slot, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    /// Waits until the published frame has been taken (lockstep file playback). False on timeout.
    pub fn wait_taken(&self, timeout: std::time::Duration) -> bool {
        let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let (slot, result) = self.taken.wait_timeout_while(slot, timeout, |s| s.0.is_some() && !s.1).unwrap_or_else(|e| e.into_inner());
        drop(slot);
        !result.timed_out()
    }

    pub fn is_closed(&self) -> bool {
        self.slot.lock().unwrap_or_else(|e| e.into_inner()).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_recycles_buffers() {
        let pool = FramePool::new(2);
        let frame = pool.take(1, 4, 2);
        let pointer = frame.rgb.as_ptr();
        drop(frame);
        let again = pool.take(2, 4, 2);
        assert_eq!(again.rgb.as_ptr(), pointer);
    }

    #[test]
    fn latest_frame_keeps_newest() {
        let latest = LatestFrame::default();
        assert!(!latest.publish(Arc::new(Frame::new(1, 1, 1, vec![0; 3]))));
        assert!(latest.publish(Arc::new(Frame::new(2, 1, 1, vec![0; 3]))));
        assert_eq!(latest.take(std::time::Duration::from_millis(1)).unwrap().id, 2);
        assert!(latest.take(std::time::Duration::from_millis(1)).is_none());
        latest.close();
        assert!(latest.take(std::time::Duration::from_secs(1)).is_none());
    }
}
