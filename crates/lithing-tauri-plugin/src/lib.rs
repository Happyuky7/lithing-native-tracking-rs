//! Tauri 2 plugin `lithing-tracking`: runs lithing-tracking natively and
//! streams every result to the web view through a [`Channel`] as a compact
//! binary frame (see [`wire`]). The web side is guest-js/index.ts, a
//! drop-in for the web `FaceTracker` (getTrackingData/getHands/getBlendshapes).
//!
//! ```ignore
//! tauri::Builder::default().plugin(tauri_plugin_lithing_tracking::init())
//! ```
//!
//! Commands (`plugin:lithing-tracking|<name>`): `start`, `stop`,
//! `update_settings`, `calibrate`, `status`, `list_cameras`,
//! `set_keep_frame_rate`, `list_camera_controls`, `capture_level`, `list_camera_formats`.

pub mod wire;

use lithing_tracking::inference::{ExecutionProvider, InferenceOptions};
use lithing_tracking::models::{self, HandModelOptions};
use lithing_tracking::pipeline::TrackerInfo;
use lithing_tracking::camera_controls::{CameraControlReport, DeviceControls};
use lithing_tracking::source::{CameraControlsHandle, CameraMode, CameraRequest, FrameSource, Y4mSource};
use lithing_tracking::{CaptureKind, Pacing, ThreadPriority, Tracker, TrackerConfig, TrackerSettings};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tauri::ipc::{Channel, InvokeResponseBody};
use tauri::plugin::{Builder, TauriPlugin};
use tauri::{Manager, Runtime, State};
use wire::WireOptions;

/// What `start` takes from the web view.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StartOptions {
    /// Camera to open (index, size, fps; 640x480 at 60 fps ideal by default;
    /// `keepFrameRate`, default true: no low-light frame rate drops).
    pub camera: CameraRequest,
    /// The camera mode (`{ format?, width, height, fps }`, from `list_camera_formats`);
    /// null or absent = Auto (`camera.mode`, else the best format for the camera request).
    /// A mode the device rejects opens in Auto with `Status::camera_warning`.
    pub camera_mode: Option<CameraMode>,
    /// Track this video file instead of a camera (tests, demos).
    pub video_file: Option<String>,
    /// "auto" (default), "directml", "cuda" or "cpu".
    pub provider: Option<String>,
    pub cpu_threads: Option<usize>,
    /// "below-normal" (default), "lowest", "normal", "above-normal".
    pub priority: Option<String>,
    pub palm_interval: Option<u32>,
    /// Model folder; default: the app's resources/models, then lithing-tracking's search.
    pub models_dir: Option<String>,
    pub settings: TrackerSettings,
    pub wire: WireOptions,
}

/// What `start` and `status` report.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub running: bool,
    pub frames_sent: u64,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub source: String,
    pub face_provider: String,
    pub hand_provider: String,
    pub load_ms: f32,
    pub dropped: u64,
    pub last_error: Option<String>,
    /// The camera's "Keep frame rate" report: controls found, what was
    /// changed, delivered fps before / after, restore result. None for a video file.
    pub camera_controls: Option<CameraControlReport>,
    /// Pixel format the camera delivers ("MJPEG", "YUYV", "NV12"...; the file's for a video).
    #[serde(default)]
    pub format: String,
    /// The camera mode asked for at start (None = Auto).
    #[serde(default)]
    pub camera_mode: Option<CameraMode>,
    /// Why the camera did not open in the requested mode (it fell back to Auto).
    #[serde(default)]
    pub camera_warning: Option<String>,
}

struct Running {
    tracker: Arc<Tracker>,
    stop: Arc<AtomicBool>,
    sent: Arc<AtomicU64>,
    forwarder: Option<JoinHandle<()>>,
    info: TrackerInfo,
    camera_controls: Option<CameraControlsHandle>,
    /// The streaming camera's index, requested mode and the modes it listed when opened.
    camera_index: Option<u32>,
    camera_mode: Option<CameraMode>,
    camera_modes: Option<Vec<CameraMode>>,
}

impl Running {
    fn shutdown(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.forwarder.take() {
            let _ = thread.join();
        }
        // The forwarder held the other reference; dropping the last one stops every tracker thread.
        drop(self.tracker);
    }
}

/// Plugin state: at most one tracker.
#[derive(Default)]
pub struct TrackingState {
    running: Mutex<Option<Running>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Tracker(#[from] lithing_tracking::TrackerError),
    #[error("{0}")]
    Source(#[from] lithing_tracking::source::SourceError),
    #[error("{0}")]
    Invalid(String),
}

impl Serialize for Error {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

fn status_of(running: Option<&Running>) -> Status {
    let Some(run) = running else { return Status::default() };
    let info = &run.info;
    Status {
        running: !run.tracker.finished(),
        frames_sent: run.sent.load(Ordering::Relaxed),
        width: info.source.width,
        height: info.source.height,
        fps: info.source.fps,
        source: info.source.description.clone(),
        face_provider: info.face_providers.get(1).map(|p| p.name().into()).unwrap_or_default(),
        hand_provider: info.hand_providers.get(1).map(|p| p.name().into()).unwrap_or_default(),
        load_ms: info.load_ms,
        dropped: run.tracker.stats.dropped.load(Ordering::Relaxed),
        last_error: run.tracker.stats.last_error.lock().ok().and_then(|e| e.clone()),
        camera_controls: run.camera_controls.as_ref().map(CameraControlsHandle::report),
        format: info.source.format.clone(),
        camera_mode: run.camera_mode.clone(),
        camera_warning: info.source.warning.clone(),
    }
}

/// Starts tracking (stopping any previous run) and forwards every result to `on_frame`.
/// Blocking: model loading takes up to a few seconds on a GPU.
pub fn start_tracking(state: &TrackingState, options: StartOptions, default_models: Option<PathBuf>, on_frame: Channel<InvokeResponseBody>) -> Result<Status, Error> {
    if let Some(previous) = state.running.lock().unwrap_or_else(|e| e.into_inner()).take() {
        previous.shutdown();
    }
    let provider = match options.provider.as_deref() {
        None => ExecutionProvider::Auto,
        Some(name) => ExecutionProvider::parse(name).ok_or_else(|| Error::Invalid(format!("unknown provider {name}")))?,
    };
    let priority = match options.priority.as_deref() {
        None => ThreadPriority::BelowNormal,
        Some(name) => ThreadPriority::parse(name).ok_or_else(|| Error::Invalid(format!("unknown priority {name}")))?,
    };
    let models_dir = options
        .models_dir
        .as_ref()
        .map(PathBuf::from)
        .or(default_models.filter(|dir| dir.join(models::FACE_LANDMARKS).is_file()))
        .unwrap_or_else(models::default_model_dir);
    let config = TrackerConfig {
        models_dir,
        inference: InferenceOptions { provider, cpu_threads: options.cpu_threads.unwrap_or(1), priority, ..Default::default() },
        hand_model: HandModelOptions { palm_interval: options.palm_interval.unwrap_or(2), ..Default::default() },
        pacing: Pacing::Realtime { speed: 1.0 },
        settings: TrackerSettings { include_raw: options.wire.face_landmarks, ..options.settings.clone() },
        ..Default::default()
    };
    let mut request = options.camera.clone();
    if options.camera_mode.is_some() {
        request.mode = options.camera_mode.clone();
    }
    let source: Box<dyn FrameSource> = match &options.video_file {
        Some(file) => Box::new(Y4mSource::open(file.as_ref())?),
        None => open_camera(&request)?,
    };
    let camera_controls = source.camera_controls();
    let camera_modes = source.camera_modes();
    let (camera_index, camera_mode) = if options.video_file.is_none() { (Some(request.index), request.mode) } else { (None, None) };
    let tracker = Arc::new(Tracker::start(source, config)?);
    let info = tracker.info.clone();
    let providers = [info.face_providers[1], info.hand_providers[1]];
    let stop = Arc::new(AtomicBool::new(false));
    let sent = Arc::new(AtomicU64::new(0));
    let forwarder = {
        let (tracker, stop, sent) = (tracker.clone(), stop.clone(), sent.clone());
        let wire = options.wire;
        std::thread::Builder::new()
            .name("lithing-forward".into())
            .spawn(move || {
                let mut buffer = Vec::with_capacity(1024);
                while !stop.load(Ordering::Relaxed) {
                    match tracker.results().recv_timeout(Duration::from_millis(100)) {
                        Ok(frame) => {
                            wire::encode(&frame, providers, wire, &mut buffer);
                            // The page closed or navigated away: stop forwarding.
                            if on_frame.send(InvokeResponseBody::Raw(buffer.clone())).is_err() {
                                break;
                            }
                            sent.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(crossbeam_timeout) => {
                            let _ = crossbeam_timeout;
                            if tracker.finished() {
                                break;
                            }
                        }
                    }
                }
            })
            .map_err(|e| Error::Invalid(e.to_string()))?
    };
    let running = Running { tracker, stop, sent, forwarder: Some(forwarder), info, camera_controls, camera_index, camera_mode, camera_modes };
    let status = status_of(Some(&running));
    *state.running.lock().unwrap_or_else(|e| e.into_inner()) = Some(running);
    Ok(status)
}

#[cfg(feature = "camera")]
fn open_camera(request: &CameraRequest) -> Result<Box<dyn FrameSource>, Error> {
    Ok(Box::new(lithing_tracking::source::CameraSource::open(request)?))
}

#[cfg(not(feature = "camera"))]
fn open_camera(_request: &CameraRequest) -> Result<Box<dyn FrameSource>, Error> {
    Err(Error::Invalid("built without camera support".into()))
}

/// Turns "Keep frame rate" on or off on the running camera (applied or
/// restored within a second). Returns false when no camera is running; the
/// option of the next `start` comes from `StartOptions::camera`.
pub fn set_keep_frame_rate(state: &TrackingState, enabled: bool) -> bool {
    let running = state.running.lock().unwrap_or_else(|e| e.into_inner());
    match running.as_ref().and_then(|run| run.camera_controls.as_ref()) {
        Some(handle) => {
            handle.set_keep_frame_rate(enabled);
            true
        }
        None => false,
    }
}

/// Every camera with its controls, read without streaming. Virtual cameras
/// are not opened; physical ones are skipped while another app uses a camera
/// (Windows privacy records) unless `force`.
pub fn list_camera_controls(force: bool) -> Result<Vec<DeviceControls>, Error> {
    lithing_tracking::camera_controls::list_device_controls(force).map_err(Error::Invalid)
}

/// Averages the raw level of the next `samples` tongue detections or face
/// frames (the in / out and relaxed / puffed calibration). None when nothing
/// runs, no face (or no tongue detection) came within `timeout`, or a newer
/// capture replaced this one. Blocks until then.
pub fn capture_level(state: &TrackingState, kind: CaptureKind, samples: u32, timeout: Duration) -> Option<f32> {
    let receiver = state.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.tracker.capture_level(kind, samples.clamp(1, 600));
    receiver.recv_timeout(timeout).ok().flatten()
}

/// The modes camera `index` offers (format, width, height, fps), read by opening it without
/// streaming; while this plugin streams that camera, the modes it listed when it was opened.
pub fn list_camera_formats(state: &TrackingState, index: u32) -> Result<Vec<CameraMode>, Error> {
    let cached = {
        let running = state.running.lock().unwrap_or_else(|e| e.into_inner());
        running.as_ref().filter(|run| run.camera_index == Some(index) && !run.tracker.finished()).and_then(|run| run.camera_modes.clone())
    };
    if let Some(modes) = cached {
        return Ok(modes);
    }
    #[cfg(feature = "camera")]
    return Ok(lithing_tracking::source::camera_modes(index)?);
    #[cfg(not(feature = "camera"))]
    Ok(Vec::new())
}

pub fn stop_tracking(state: &TrackingState) {
    if let Some(running) = state.running.lock().unwrap_or_else(|e| e.into_inner()).take() {
        running.shutdown();
    }
}

mod commands {
    use super::*;

    #[tauri::command]
    pub async fn start<R: Runtime>(app: tauri::AppHandle<R>, options: StartOptions, on_frame: Channel<InvokeResponseBody>) -> Result<Status, Error> {
        let models = app.path().resource_dir().ok().map(|dir| dir.join("models"));
        tauri::async_runtime::spawn_blocking(move || start_tracking(&app.state::<TrackingState>(), options, models, on_frame))
            .await
            .map_err(|e| Error::Invalid(e.to_string()))?
    }

    #[tauri::command]
    pub async fn stop<R: Runtime>(app: tauri::AppHandle<R>) -> Result<(), Error> {
        tauri::async_runtime::spawn_blocking(move || stop_tracking(&app.state::<TrackingState>())).await.map_err(|e| Error::Invalid(e.to_string()))
    }

    #[tauri::command]
    pub fn update_settings(state: State<'_, TrackingState>, settings: TrackerSettings) -> Result<(), Error> {
        if let Some(running) = state.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            running.tracker.update_settings(settings);
        }
        Ok(())
    }

    #[tauri::command]
    pub fn calibrate(state: State<'_, TrackingState>) {
        if let Some(running) = state.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            running.tracker.calibrate();
        }
    }

    #[tauri::command]
    pub fn status(state: State<'_, TrackingState>) -> Status {
        status_of(state.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref())
    }

    #[tauri::command]
    pub fn set_keep_frame_rate(state: State<'_, TrackingState>, enabled: bool) -> bool {
        super::set_keep_frame_rate(&state, enabled)
    }

    #[tauri::command]
    pub async fn list_camera_controls(force: Option<bool>) -> Result<Vec<DeviceControls>, Error> {
        tauri::async_runtime::spawn_blocking(move || super::list_camera_controls(force.unwrap_or(false))).await.map_err(|e| Error::Invalid(e.to_string()))?
    }

    /// `kind`: "tongue" (tongue tracking must be on) or "cheeks"; `samples` (default 30). Gives up after `timeoutMs` (default 6000).
    #[tauri::command]
    pub async fn capture_level<R: Runtime>(app: tauri::AppHandle<R>, kind: CaptureKind, samples: Option<u32>, timeout_ms: Option<u64>) -> Result<Option<f32>, Error> {
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(6000).min(30_000));
        tauri::async_runtime::spawn_blocking(move || super::capture_level(&app.state::<TrackingState>(), kind, samples.unwrap_or(30), timeout))
            .await
            .map_err(|e| Error::Invalid(e.to_string()))
    }

    #[tauri::command]
    pub async fn list_camera_formats<R: Runtime>(app: tauri::AppHandle<R>, index: u32) -> Result<Vec<CameraMode>, Error> {
        tauri::async_runtime::spawn_blocking(move || super::list_camera_formats(&app.state::<TrackingState>(), index)).await.map_err(|e| Error::Invalid(e.to_string()))?
    }

    #[tauri::command]
    pub fn list_cameras() -> Result<Vec<String>, Error> {
        #[cfg(feature = "camera")]
        return Ok(lithing_tracking::source::list_cameras()?);
        #[cfg(not(feature = "camera"))]
        Ok(Vec::new())
    }
}

/// The plugin. Register it with `tauri::Builder::plugin(init())` and allow
/// `lithing-tracking:default` in a capability.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("lithing-tracking")
        .invoke_handler(tauri::generate_handler![commands::start, commands::stop, commands::update_settings, commands::calibrate, commands::status, commands::list_cameras, commands::set_keep_frame_rate, commands::list_camera_controls, commands::capture_level, commands::list_camera_formats])
        .setup(|app, _api| {
            app.manage(TrackingState::default());
            // Feature `load-dynamic` (the Linux packages): ONNX Runtime is the shared library
            // the app bundles next to its models. Without one there, lithing-tracking tries
            // ORT_DYLIB_PATH and the system's library path; if that fails too, `start`
            // returns an error and the app keeps the web tracker.
            #[cfg(feature = "load-dynamic")]
            if let Ok(dir) = app.path().resource_dir() {
                let library = dir.join(lithing_tracking::inference::runtime_library_name());
                if library.is_file() {
                    lithing_tracking::inference::set_runtime_library(library);
                }
            }
            Ok(())
        })
        .on_drop(|app| {
            if let Some(state) = app.try_state::<TrackingState>() {
                stop_tracking(&state);
            }
        })
        .build()
}
