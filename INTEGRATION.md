# Integrating native tracking into Vtuber-Lithing

The plan for switching the app to the native tracker behind a setting, with
the web tracker kept as the fallback (and for the browser build, where there
is no Rust). The app has since done this (Settings -> Tracking engine, Beta:
`src/app/native-tracker.ts` wraps a copy of `guest-js/` in
`src/app/native-tracking/`); the steps below are what that integration
follows.

## What the app gets

* `crates/lithing-tauri-plugin` (crate `tauri-plugin-lithing-tracking`): a Tauri 2
  plugin named `lithing-tracking` with the commands `start`, `stop`,
  `update_settings`, `calibrate`, `status`, `list_cameras`,
  `set_keep_frame_rate`, `list_camera_controls` and `list_camera_formats`
  (all in `lithing-tracking:default`). `start` takes a
  `Channel` and pushes one ~600-byte binary frame per processed camera frame
  (layout in `src/wire.rs`). Under Tauri's 1 KiB limit a frame is delivered
  with a single evaluated message, no extra fetch round trip.
* `guest-js/index.ts`: `NativeFaceTracker`, the same public surface main.ts
  uses on `FaceTracker` (`initialize`, `start`, `stop`, `update(frameSkip)`,
  `getHands`, `getBlendshapes`, `calibrate`, every `set*` option) plus
  `getTrackingData()`. `guest-js/decode.ts` is the dependency-free decoder.
* The round trip is proven by `cargo test -p tauri-plugin-lithing-tracking`:
  frames from the real tracker go through a Channel, are decoded in Rust and by
  `decode.ts` under Node and compared field by field; the commands are invoked
  through Tauri's IPC with the mock runtime.

## Files to touch

1. **`src-tauri/Cargo.toml`**: add the plugin (path dependency while it lives
   in this folder, a git dependency later):
   ```toml
   [target.'cfg(not(any(target_os = "android", target_os = "ios")))'.dependencies]
   tauri-plugin-lithing-tracking = { path = "../vtuber-lithing-rust-native/crates/lithing-tauri-plugin" }
   ```
   The workspace pins `ort = 2.0.0-rc.15` (ONNX Runtime 1.31 with DirectML).
   The app's `[profile.release]` should keep `lto`/`opt-level` as they are;
   ONNX Runtime is linked statically, `DirectML.dll` is copied next to the exe
   by `ort` (`copy-dylibs`).
2. **`src-tauri/src/lib.rs`**: `.plugin(tauri_plugin_lithing_tracking::init())`
   on the builder (desktop only, inside the existing `cfg` for desktop plugins).
3. **`src-tauri/capabilities/default.json`**: add `"lithing-tracking:default"`.
4. **`src-tauri/tauri.conf.json`**: bundle the models and the DirectML runtime:
   ```json
   "bundle": { "resources": { "../vtuber-lithing-rust-native/models/*.onnx": "models/" } }
   ```
   plus `DirectML.dll` (from `target/release/`) in `bundle.windows` resources or
   next to the exe. The plugin looks in `<resource dir>/models` first.
   Add Apache-2.0 attribution for the models (MODELS.md).
5. **`src/app/native-tracker.ts`** (new): copy `guest-js/index.ts` and
   `guest-js/decode.ts` (or import them from a small npm workspace package).
6. **`src/app/app-settings.ts` / `settings.ts`**: a `trackingBackend:
   'web' | 'native'` setting (default `web` until the native one has had field
   time), shown only when running inside Tauri on Windows/Linux.
7. **`src/main.ts`** (line 93 `const tracker = new FaceTracker()`): create
   `NativeFaceTracker` instead when the setting says native. The calls at
   lines ~273-372 (`initialize`, `start`, `setFaceTrackingFps`, ...) and the
   render loop at ~847 (`tracker.update(...)`, `getHands()`,
   `getBlendshapes()`) work unchanged. Differences to handle:
   * `start(cameraId)`: the web passes a MediaDevices `deviceId`; native wants
     an index. `NativeFaceTracker.start` accepts an index or a label substring
     (matched against `list_cameras`). Map the selected device's `label`.
   * The phone link passes an external `MediaStream`: keep the web tracker for
     that source (or send its frames to Rust later).
   * Do **not** open the camera in the web view while native tracking runs
     (two openers of one webcam fail or halve its rate on some drivers).
   * `getDebugInfo()`, `getFaceContours()`, `getHandSide()`,
     `getTongueDetector()`, `captureSizeDiffers()` are web-only. For the debug
     view, start with `wire: { faceLandmarks: true, handLandmarks: true }` and
     read `getLatestFrame()` (landmarks, per-stage ms, providers); the camera
     preview is not sent (a later step could send a small JPEG on demand).
   * `setDelegate('GPU' | 'CPU')` maps to Auto / CPU and applies on the next
     `start()`.
8. **`src/app/tracking-debug.ts`**: show `faceMs`, `handsMs`, `latencyMs` and
   the provider from `NativeFrame` when native.

## Feature coverage (web Tracking options -> native)

Every option the web `FaceTracker` takes has a native equivalent with the same
behaviour, settable live through `update_settings` (the `TrackerSettings`
JSON: `face.lowLight`, `face.highLight`, `face.glassesMode`, ...) and through
the client's setters of the same names as the web tracker's:

| Web (FaceTracker / Tracking panel) | Native |
|---|---|
| `setLowLightMode` ("Mejorar camara con poca luz") | `face.lowLight` + `hands.lowLight`. The face models see the web's canvas filter `blur() brightness() contrast() saturate(0.78)` with the adaptive brightness from the 32x24 light meter (`light.rs`, same EMA, same `toFixed(2)` values: identical filter strings on every test frame), detection / presence thresholds 0.28 / 0.34, hold 950 ms (face) / 600 ms (hands), stability +0.24, 3-frame median and step limits x0.52, calmer wrist (x0.65) and finger (+0.15) smoothing. Recalibrates on a change. |
| `setHighLightMode` ("Mejorar camara sobreexpuesta") | `face.highLight` + `hands.highLight`: `brightness(0.72) contrast(1.1) saturate(0.9)`, thresholds 0.38 / 0.44, hold 550 ms, stability +0.14, median and step limits x0.68, wrist x0.85. Low light wins when both are set, as on the web. |
| `setGlassesTrackingMode` ("Tracking optimizado para gafas") | `face.glassesMode`: blink geometry weight 0.55 (else 0.9), gaze 0.8 / 0.2 (x) and 0.85 / 0.15 (y) blendshapes / iris (else 0.55 / 0.45, 0.65 / 0.35). |
| `setMirrorFaceTracking`, `setMirrorHandTracking` | `face.mirror`, `hands.mirror` |
| `setTrackingStability` | `face.stability` |
| `setFaceTrackingFps`, `setHandTrackingFps` | `faceFps`, `handFps` |
| `setHandTracking` | `handsEnabled` |
| `setHandQuality` | `hands.quality` (wrist tuning) + the camera size at `start` (`captureSizeDiffers` in the client) |
| `setFingerSmoothing` | `hands.fingerSmoothing` |
| `setTongueTracking(enabled, tuning)` | `tongueEnabled`, `tongueTuning` (readings arrive in the blendshapes) |
| `calibrate` | `calibrate` command (24 face frames, as on the web) |
| eye-side vote, wink separation, hold / ease to neutral | automatic, ported in `face_post.rs` |
| `predict(now)` (motion prediction) | `NativeFaceTracker.predict` (port of motion-predictor.ts, fed with each detection at its capture time) |
| `setDelegate('GPU' / 'CPU')` | provider Auto / CPU on the next `start` |
| `setDetectionThread`, `setWorkerFallback`, `setHandWorker` | accepted and ignored: native detection always has its own threads |

CLI equivalents: `--low-light`, `--high-light`, `--glasses`,
`--no-mirror-face`, `--no-mirror-hands`, `--stability`, `--finger-smoothing`,
`--hand-quality`, `--face-fps`, `--hand-fps`, `--no-hands`, `--tongue` on
`run` / `dump` (`bench` takes `--low-light`, `--high-light`, `--glasses`).

The app's copy of the client (`src/app/native-tracking/client.ts`, from
c1b06af) already has `setLowLightMode`, `setHighLightMode`,
`setGlassesTrackingMode` and the other setters, and `TrackerSettings` kept its
shape, so with this plugin version the modes take effect without re-copying.
Re-copying `guest-js/index.ts` adds `predict`, `captureSizeDiffers`, the
worker no-ops and setters that skip unchanged values; `decode.ts` (the wire
format) is unchanged.

### Keep frame rate (native only)

`StartOptions.camera.keepFrameRate` (default **true**; older clients that do
not send it get it on) stops the webcam from dropping to 15 fps in low light
and restores its controls on `stop` (README, "Keep frame rate"). Client:
`setKeepFrameRate(on)` (live on a running camera through
`set_keep_frame_rate`, and used by the next `start`), `getKeepFrameRate()`,
`listCameraControls(force?)`. `status()` carries `cameraControls`: the
controls found, each change, `fpsBefore` / `fpsAfter`, `strategy`,
`restored`, `virtualCamera`. A settings toggle "Keep frame rate (camera)" next
to the low light option is the natural place for it in the app.

### Camera mode (resolution / FPS / format)

`StartOptions.cameraMode` (`{ format?, width, height, fps }`, null or absent =
Auto) opens the camera in exactly that mode (`RequestedFormatType::Exact`);
Auto is `best_camera_format` (MJPEG, then YUYV / NV12, the resolution closest
to `camera.width` x `camera.height`, the highest rate up to `camera.fps`, never
below 15). A mode the device rejects, or does not list, opens in Auto and
`status().cameraWarning` says why; `status().format` is the delivered pixel
format and `status().cameraMode` the mode asked for. `list_camera_formats
{ index }` lists one camera's modes (format, width, height, fps; sorted, no
duplicates) by opening it without streaming; while the plugin streams that
camera it answers from the list read when the camera opened. Client:
`listCameraFormats(index)`, `setCameraMode(mode | null)` (next `start`),
`getCameraMode()`. CLI: `lithing-track run camera:0 --mode MJPEG:1280x720@30`
(`640x480@60` without a format takes the best format of that size and rate).

### Not covered (web only)

* The camera preview and `getFaceContours()` in the Debug tab (no image or
  landmarks cross the wire by default; `wire.faceLandmarks` sends landmarks).
* `getHandSide()` (finger filter / gesture classifier state) and
  `getTongueDetector()`: the tongue level capture ("tongue in / out" buttons,
  `captureLevel`) needs the detector object in the page. Tongue readings and
  tuning work.
* The phone camera (an external `MediaStream`): the app keeps the web tracker
  for that source.
* `update(frameSkip)`: the frame skip is ignored (detection is off the render
  loop).
* Exact pixel identity of the light filter on other GPUs: the filter
  reproduces Chromium's accelerated canvas as measured on D3D11 / NVIDIA (see
  ARCHITECTURE.md, "Light filters"); a software-rendered WebView skips the
  small blur, which the native filter does not imitate.

## Android

The same plugin runs on Android with `default-features = false, features = ["nnapi"]`
(no nokhwa, no DirectML). What differs:

* Camera: the plugin's Kotlin half (`android/`, registered by the plugin as
  `com.lithing.tracking.LithingCameraPlugin`) opens Camera2 with an `ImageReader`
  (YUV_420_888, no preview) and hands each image to Rust through JNI; the frames
  feed `push::push_source`, so the tracker runs unchanged. `list_cameras` lists
  front cameras first with the WebView's labels (`0: camera2 1, facing front
  (Camera2)`), so `matchNativeCamera` finds the camera chosen in the page and an
  empty choice opens the selfie camera. The camera permission is asked on `start`
  if the page has not got it yet. `keepFrameRate` picks the fixed AE range
  (30-30) over a variable one. The camera is closed while the activity is paused
  and reopened on resume; if another user takes it (the WebView's
  `getUserMedia`), it stays closed. `list_camera_formats` and
  `list_camera_controls` return nothing (Auto only).
* Models: bundled as APK assets under `models/` (the app's
  `tauri.android.conf.json` resources); `start` copies them to
  `<files>/lithing-models` once per install / update and loads them from there.
* ONNX Runtime: static, from `ort`'s Android download; the app must ship
  `libc++_shared.so` in `jniLibs/<abi>` (the app's build.rs copies it from the NDK).
* Providers: Auto is the CPU; `provider: "nnapi"` tries NNAPI.
* One camera user at a time: the page must stop native tracking before it opens
  the camera in the WebView (the app's "Use as PC camera" does).

## Behaviour notes

* `update()` returns TrackingData only when a new face detection arrived since
  the last call, like the web version; the render loop's `frameSkip` is
  ignored because detection no longer runs in the render loop.
* Hands: the native default rates are face 30/s and hands 30/s (the web
  default for hands is 12/s because each detection blocked the render thread
  for ~15 ms per hand; natively it runs on another thread).
* The tracker threads start below normal priority (`priority: 'below-normal'`)
  so a game keeps the CPU, and raise themselves to normal only while they are
  starved (ARCHITECTURE.md, "Thread priority finding"); Auto moves inference
  between GPU and CPU when one of them is saturated.
* Results carry the camera frame id; face and hands in one result come from
  the same frame.
