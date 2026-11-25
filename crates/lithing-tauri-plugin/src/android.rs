//! Android: the camera is Camera2 in Kotlin (`android/`, `LithingCameraPlugin`), which
//! hands every YUV_420_888 image to Rust through JNI (`LithingNative.pushFrame`); the
//! frames feed a [`push_source`](lithing_tracking::push::push_source), so the tracker
//! and everything after it run exactly as on the desktop. The models ship as APK assets
//! (`models/*.onnx`) and are copied once per app update to the app's files folder,
//! where ONNX Runtime can read them.

use crate::{Error, OpenedCamera};
use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::jint;
use lithing_tracking::push::{FramePusher, PushInfo, YuvPlanes, push_source};
use lithing_tracking::source::CameraRequest;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;
use tauri::Runtime;
use tauri::plugin::PluginHandle;

/// The pusher of the running tracker, which the camera thread's JNI calls feed.
static PUSHER: Mutex<Option<FramePusher>> = Mutex::new(None);

/// `pushFrame` results (Kotlin `LithingNative`).
const PUSHED: jint = 0;
/// No tracker yet (models still loading): keep the camera running.
const NOT_READY: jint = 1;
/// The tracker is gone: Kotlin closes the camera.
const CLOSED: jint = 2;
/// The image could not be read (bad strides, not a direct buffer).
const INVALID: jint = 3;

/// The Kotlin half, registered by the plugin's setup.
pub struct AndroidCamera<R: Runtime>(pub PluginHandle<R>);

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CameraStarted {
    width: u32,
    height: u32,
    fps: f32,
    label: String,
}

#[derive(Deserialize)]
struct ModelsDir {
    dir: String,
}

#[derive(Deserialize)]
struct CameraList {
    cameras: Vec<String>,
}

fn kotlin(error: impl std::fmt::Display) -> Error {
    Error::Invalid(error.to_string())
}

impl<R: Runtime> AndroidCamera<R> {
    /// The folder with the models: the APK's `models/*.onnx`, copied out on the first run
    /// after an install or update. Blocking (call it off the main thread).
    pub fn models_dir(&self) -> Result<PathBuf, Error> {
        let models: ModelsDir = self.0.run_mobile_plugin("prepareModels", serde_json::json!({})).map_err(kotlin)?;
        Ok(PathBuf::from(models.dir))
    }

    /// "index: camera2 <id>, facing front (Camera2)", front cameras first (index 0 is the
    /// selfie camera). The names are the WebView's device labels, so the app's label
    /// matching finds the camera chosen in its list.
    pub fn list(&self) -> Result<Vec<String>, Error> {
        let list: CameraList = self.0.run_mobile_plugin("listCameras", serde_json::json!({})).map_err(kotlin)?;
        Ok(list.cameras)
    }

    /// Opens camera `request.index` (Camera2, asking for the camera permission first if
    /// needed) and returns the source its frames feed. Blocking until the camera streams.
    pub fn open(&self, request: &CameraRequest) -> Result<OpenedCamera, Error> {
        let (width, height, fps) = match &request.mode {
            Some(mode) => (mode.width, mode.height, mode.fps),
            None => (request.width, request.height, request.fps.min(30)),
        };
        let started: CameraStarted = self
            .0
            .run_mobile_plugin("startCamera", serde_json::json!({ "index": request.index, "width": width, "height": height, "fps": fps, "keepFrameRate": request.keep_frame_rate }))
            .map_err(kotlin)?;
        let (source, pusher) = push_source(PushInfo { width: started.width, height: started.height, fps: started.fps, description: started.label, format: "YUV_420_888".into() });
        *PUSHER.lock().unwrap_or_else(|e| e.into_inner()) = Some(pusher.clone());
        let handle = self.0.clone();
        let on_stop = Box::new(move || {
            pusher.close();
            let mut slot = PUSHER.lock().unwrap_or_else(|e| e.into_inner());
            if slot.as_ref().is_some_and(FramePusher::is_closed) {
                *slot = None;
            }
            drop(slot);
            stop_camera(handle);
        });
        Ok(OpenedCamera { source: Box::new(source), on_stop: Some(on_stop) })
    }
}

/// Closes the Kotlin camera and waits (up to 2 s) until it is closed, so the web view
/// can open it right after. On its own thread: a stop from the Android main thread
/// (the app closing) must not wait for a command that runs on that same thread.
fn stop_camera<R: Runtime>(handle: PluginHandle<R>) {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name("lithing-camera-stop".into()).spawn(move || {
        let _ = handle.run_mobile_plugin::<serde_json::Value>("stopCamera", serde_json::json!({}));
        let _ = done_tx.send(());
    });
    if spawned.is_ok() {
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    }
}

/// A direct ByteBuffer's bytes (an `Image.Plane` buffer: valid until Kotlin closes the image,
/// which happens after this call returns).
fn direct_bytes<'a>(env: &JNIEnv<'_>, buffer: &JByteBuffer<'_>) -> Option<&'a [u8]> {
    let pointer = env.get_direct_buffer_address(buffer).ok()?;
    let length = env.get_direct_buffer_capacity(buffer).ok()?;
    if pointer.is_null() || length == 0 {
        return None;
    }
    // SAFETY: the buffer is a live direct buffer of `length` bytes for the duration of the call.
    Some(unsafe { std::slice::from_raw_parts(pointer, length) })
}

/// `LithingNative.pushFrame` (Kotlin): one camera image, converted to RGB, rotated upright
/// and handed to the tracker. Runs on the camera's handler thread.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_lithing_tracking_LithingNative_pushFrame<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    y: JByteBuffer<'local>,
    u: JByteBuffer<'local>,
    v: JByteBuffer<'local>,
    y_row_stride: jint,
    uv_row_stride: jint,
    uv_pixel_stride: jint,
    width: jint,
    height: jint,
    rotation: jint,
) -> jint {
    // A panic must not cross into the JVM.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(pusher) = PUSHER.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
            return NOT_READY;
        };
        let (Some(y), Some(u), Some(v)) = (direct_bytes(&env, &y), direct_bytes(&env, &u), direct_bytes(&env, &v)) else {
            return INVALID;
        };
        if [y_row_stride, uv_row_stride, uv_pixel_stride, width, height, rotation].iter().any(|value| *value < 0) {
            return INVALID;
        }
        let planes = YuvPlanes {
            y,
            u,
            v,
            y_row_stride: y_row_stride as usize,
            uv_row_stride: uv_row_stride as usize,
            uv_pixel_stride: uv_pixel_stride as usize,
            width: width as u32,
            height: height as u32,
            // Camera2's YUV_420_888 is JFIF (BT.601 full range).
            full_range: true,
        };
        match pusher.push_yuv420(&planes, rotation as u32) {
            Ok(()) => PUSHED,
            Err(lithing_tracking::push::PushError::Closed) => CLOSED,
            Err(_) => INVALID,
        }
    }))
    .unwrap_or(INVALID)
}
