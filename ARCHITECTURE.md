# Architecture: Vtuber-Lithing towards native Rust

Today the app is Tauri 2: a small Rust backend and a TypeScript front end in
WebView2 that does everything else: camera, MediaPipe in WebAssembly, three.js
+ @pixiv/three-vrm rendering, UI and extensions. Tracking runs on the web
view's main thread: a face detection blocks rendering for ~7 ms and each hand
for ~15 ms (RTX 5060 Ti), the WebGL delegate shares the GPU with the game being
streamed, and the face rate tops out around 24 detections/s at the 30/s setting.

The plan agreed with the owner moves the app to native code in three phases,
each useful on its own, the way Discord is moving pieces off Electron rather
than rewriting at once:

1. **Tracking native** (this folder, done): models on ONNX Runtime on their
   own threads; the web view keeps rendering and receives results.
2. **Rendering native** (wgpu): VRM loading, MToon, spring bones,
   expressions, the virtual camera output; the web view becomes UI only.
3. **UI native** (last, or never): only if the web view itself is the cost.

## Phase 1: native tracking

```text
 camera (nokhwa: MSMF / V4L2)          video file (.y4m, or ffmpeg pipe)
        │ own thread, RGB8 frames from a buffer pool
        ▼
  newest-frame mailbox (1 slot: a slow consumer skips stale frames, never queues them)
        ▼
  face thread ── decides per frame what is due (face rate, hand rate, tongue rate)
   │   │ hands the same Arc<Frame> to ──▶ hand thread
   │   ▼                                   palm detector (only when a hand is missing,
   │  face: detector only when lost,        every N hand frames) → 21 landmarks per
   │  else ROI from last landmarks →        tracked ROI → dedupe → next ROIs
   │  478 landmarks → 52 blendshapes
   │   ◀───────────── waits for the hands of this frame ─────┘
   ▼
  post-processing (ports of face-tracker.ts, hand-pose.ts, tongue-detector.ts)
   ▼
  TrackingFrame { frame_id, timestamp, tracking, blendshapes, hands, raw..., timings }
   ▼ bounded result channel (newest wins)
  CLI / Tauri plugin ──Channel, 588-byte binary frame──▶ NativeFaceTracker (TS)
```

Key decisions:

* **Same models, converted, not replaced.** The app's own `.task` bundles are
  converted TFLite → ONNX (MODELS.md). Outputs match the web tracker within a
  pixel (see Accuracy), so mappings, calibrations and user habits carry over.
* **MediaPipe's graph logic reimplemented in Rust** (`models.rs`,
  `geometry.rs`): letterboxed detector input, SSD anchors, weighted NMS,
  rotated ROI crops (bilinear affine warp straight into the model's input
  tensor), landmark projection, ROI from landmarks, palm/track association and
  landmark-box deduplication. Two deliberate differences, both switchable:
  a lost face is re-detected on the same frame instead of the next one, and a
  missing second hand is looked for every 2nd hand frame instead of every frame
  (`--palm-interval 1` restores MediaPipe's behaviour; the palm detector is the
  most expensive model on a CPU).
* **Synchronised results.** One coordinator owns the frame; face and hands of
  a frame run in parallel and are joined before anything is emitted, so every
  result has one `frame_id` (the web worker experiment failed because hands
  arrived for other frames than the face).
* **Bounded everything.** A one-slot frame mailbox, a 4-deep result channel
  that drops the oldest, a frame-buffer pool: steady-state capture allocates
  nothing, inference reuses its input/output buffers, a slow stage never builds
  latency.
* **Polite to games.** Inference threads run below normal priority by
  default and raise themselves to normal only while starved (see "Thread
  priority finding"); CPU sessions run single-threaded on their worker (no ORT pool
  spinning on idle cores); when ORT does use a pool its threads are created at
  the same priority (`ThreadManager`).
* **Execution providers.** DirectML (Windows, any DX12 GPU), CUDA (feature),
  CPU (MLAS; XNNPACK behind a feature for ORT builds that include it), and
  **Auto**: every model is loaded on the GPU and the CPU, both are timed at
  start, the faster is used (the GPU must win by 25%), and every 4 s the other
  provider is timed once on the live input; a clear win (GPU→CPU at 15%,
  CPU→GPU at 40%) switches. A game saturating the GPU moves tracking to the
  CPU; one saturating the CPU moves it back.
* **Outputs keep the web names.** `TrackingData` (headYaw ... surprised),
  `HandPose` (handedness, userHand, wristX ... fingers[20], gesture), the
  blendshapes record with avatar-side names and the tongue channels.

### Code map

| File | Role |
|---|---|
| `frame.rs` | `Frame`, `FramePool`, `LatestFrame` (one-slot mailbox, lockstep wait) |
| `source.rs` | `Y4mSource` (+ ffmpeg pipe), `CameraSource` (nokhwa on its own thread, MJPEG→YUYV→fastest→any fallbacks, reports what it got; runs "Keep frame rate" on the camera thread: one-second delivered-rate windows + sparse frame luma) |
| `lithing-camera-controls` (crate) | `CameraControls` trait with the Media Foundation (`windows.rs`) and V4L2 (`linux.rs`) backends, the `KeepFrameRate` policy (`keep.rs`, mock-tested) and the restore journal |
| `geometry.rs` | rects, crops, anchors, decoding, NMS |
| `inference.rs` | sessions, providers, `AdaptiveModel` (Auto) |
| `models.rs` | face and hand model loops with ROI tracking |
| `face_post.rs`, `hands_post.rs`, `hand_pose.rs`, `tongue.rs` | the ported post-processing |
| `light.rs` | the low / high light image filter (Chromium canvas filter semantics) and per-mode face thresholds |
| `pipeline.rs` | threads, scheduling, `Tracker` API |
| `stress.rs` | in-memory MatMul chain for GPU-contention benchmarks |

## Measurements

All numbers: Ryzen 9 7900X (12 cores / 24 threads), RTX 5060 Ti, Windows 11,
`ghana640.y4m` (640x480, CC footage with a face and hands) played like a
30 fps camera (`lithing-track bench`, 12-15 s per scenario, face 30/s and hands
30/s requested). **The owner was playing Fortnite during every measurement**
(GPU 60-90 % busy and the CPU 45-65 % busy before any synthetic load was
added; logged per scenario in the bench report), so "game only" below is the
streaming situation, not an idle lab machine. Synthetic loads: `CPU load` = 24
busy threads at normal priority (every logical CPU), `GPU load` = a DirectML
MatMul chain in a loop (~6.7 TFLOP/s), `low-end` = the process pinned to 2 CPUs
(no SMT siblings) with the CPU provider and 1 thread per model.

### Native tracking

| Scenario / provider | face det/s | face ms p50/p95 | hands ms p50/p95 | capture→result p50/p95 | frames dropped | process CPU (100 = 1 core) | RAM |
|---|---|---|---|---|---|---|---|
| game only / CPU | 29.0 | 6.5 / 8.2 | 8.5 / 18.9 | 8.5 / 18.9 ms | 0 % | 57 % | 102 MB |
| game only / DirectML | 29.2 | 4.0 / 5.1 | 2.6 / 4.3 | 4.1 / 5.3 ms | 0 % | 20 % | 383 MB |
| game only / Auto | 29.1 | 3.8 / 6.6 | 2.8 / 6.7 | 3.9 / 7.7 ms | 0 % | 18 % | 443 MB |
| + GPU load / CPU | 30.1 | 8.3 / 11.6 | 11.0 / 23.6 | 11.4 / 23.7 ms | 0 % | 72 % | 117 MB |
| + GPU load / DirectML | 29.1 | 5.5 / 9.5 | 4.1 / 11.3 | 6.0 / 11.6 ms | 0 % | 20 % | 394 MB |
| + GPU load / Auto | 28.5 | 12.8 / 20.5 | 17.1 / 59.6 | 24.4 / 64.7 ms | 4 % | 55 % | 443 MB |
| + CPU load / CPU | 27.2 | 18.0 / 27.4 | 28.1 / 54.7 | 38.1 / 70.9 ms | 9 % | 98 % | 104 MB |
| + CPU load / DirectML | 24.9 | 15.7 / 32.2 | 7.6 / 16.9 | 16.6 / 37.3 ms | 15 % | 28 % | 387 MB |
| + CPU load / Auto | 25.0 | 9.6 / 16.9 | 7.8 / 19.0 | 11.2 / 23.2 ms | 14 % | 28 % | 439 MB |
| + CPU and GPU load / CPU | 26.5 | 19.2 / 31.6 | 27.5 / 55.0 | 37.8 / 69.3 ms | 9 % | 93 % | 119 MB |
| + CPU and GPU load / DirectML | 10.5 | 24.2 / 42.4 | 11.9 / 28.7 | 27.9 / 51.0 ms | 64 % | 18 % | 379 MB |
| + CPU and GPU load / Auto | 27.8 | 20.8 / 37.8 | 18.3 / 43.9 | 25.9 / 61.1 ms | 7 % | 32 % | 435 MB |
| low-end (2 CPUs) / CPU | 29.0 | 7.0 / 8.9 | 8.6 / 17.7 | 10.9 / 19.2 ms | 0 % | 62 % | 118 MB |
| low-end, its 2 CPUs also busy / CPU | 10.2 | 32.0 / 48.8 | 48.1 / 74.8 | 68.0 / 100.7 ms | 65 % | 30 % | 119 MB |

A face was found in every face run on this clip; hands ran on the same frames
at the same rate. RAM includes ONNX Runtime and, for DirectML/Auto, the D3D12
device (Auto keeps both providers' sessions loaded).

**Thread priority finding.** The first run of the matrix kept the inference
threads at below-normal priority unconditionally, as planned. With every core
busy at normal priority that starved tracking almost completely: CPU 7.1
det/s, DirectML 0.7 det/s (the thread that submits GPU work never got a core),
CPU+GPU load 0.1-5 det/s, low-end with its CPUs busy 0.2 det/s. The tracker now
**starts below normal and raises itself to normal priority while it is
starved** (under 60 % of the asked rate for a second), dropping back after
20 s of keeping up (`TrackerConfig::adaptive_priority`, on by default;
`--fixed-priority` disables it). That gave the CPU-load rows above (25-28
det/s instead of 0.7-7), while a game that leaves some CPU free still gets
the cores first.

**Auto** chose the CPU for some models under GPU load, which on this machine
cost latency (24 vs 6 ms): DirectML stays faster even on a saturated RTX 5060
Ti because the models are tiny, and one CPU probe against the GPU's moving
average is a noisy comparison. Auto pays off when both are loaded (27.8 det/s
vs 10.5 on DirectML alone). The switch thresholds (`AdaptiveModel::run`)
deserve tuning on weaker GPUs, where the GPU side would lose more clearly.

### Model cost per inference (`lithing-track model-times`, game running)

| Model | CPU, 1 thread (mean) | DirectML (p50) |
|---|---|---|
| face detector (128) | 1.6 ms | 0.7 ms |
| face landmarks (256, 478 points) | 7.7 ms | 1.9 ms |
| face blendshapes | 1.4 ms | 1.0 ms (kept on the CPU) |
| palm detector (192) | 13.0 ms | 6.8 ms |
| hand landmarks (224) | 10.5 ms | 1.6 ms |

The palm detector is the expensive model; it only runs when a hand is
missing, and with one of two hands tracked only every 2nd hand frame.

### Native vs web

| | Web (current app) | Native |
|---|---|---|
| Face detections/s at the 30/s setting | ~24/s (owner, same footage) | 29.0-29.2/s with a game running; 25-28/s under full CPU load |
| Render-thread time per face detection | ~7 ms blocked | 0 ms (the web view decodes a 588-byte frame) |
| Render-thread time per hand detection | ~15 ms blocked per hand | 0 ms |
| Face model time | 15.6 ms CPU / 16.9 ms GPU delegate (mean)* | 6.5 ms CPU, 1 thread / 4.0 ms DirectML (p50), own thread |
| Hand models (both hands) | 41.3 ms CPU / 24.2 ms GPU delegate (mean)* | 8.5 ms CPU / 2.6 ms DirectML (p50) |
| Face and hands of the same frame | only on the main thread (the worker desynced) | always: one frame id per result |

\* `tools/webref.mjs` the same evening: the app's MediaPipe build
(tasks-vision 0.10.35, same .task files and options) in headless Chrome on the
same 300 frames, game running. Headless Chrome's GPU path is not WebView2's;
the owner's 7 ms / 15 ms are the in-app figures.

## Accuracy against the web MediaPipe

`tools/webref.mjs` runs the app's MediaPipe (VIDEO mode, the app's
thresholds) headlessly on PNG frames; `lithing-track dump` runs the same frames
natively with `--palm-interval 1 --no-same-frame-redetect` (MediaPipe's own
schedule); `tools/compare_reference.mjs` compares them frame by frame. 300
frames per clip, CPU provider (DirectML on ghana gives the same: 0.54 px, MAD
0.0056).

| Clip | faces found (both) | face landmark error mean (p95) | in eye distances | iris | blendshape MAD, mean of 52 | worst blendshape | hands native / web | hand landmark error mean (p95) | world landmarks | handedness agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| ghana (signing, hands) | 300/300 | 0.54 px (0.98) | 0.77 % | 0.52 px | 0.0056 | mouthSmileLeft 0.021 | 237 / 236 | 1.22 px (4.69) | 1.8 mm | 100 % |
| monica (glasses, outdoor) | 296/300 (web 297) | 0.49 px (0.87) | 0.95 % | 0.47 px | 0.0091 | mouthUpperUpRight 0.032 | 531 / 525 | 1.34 px (4.70) | 2.9 mm | 99.2 % |
| pamela (640x360, hands together) | 300/300 | 0.51 px (0.82) | 0.98 % | 0.53 px | 0.0102 | mouthPucker 0.049 | 483 / 500 | 1.50 px (4.64) | 2.8 mm | 98.3 % |

The remaining differences come from crop resampling (bilinear affine warp vs
MediaPipe's image-to-tensor), float16 vs float32 kernels, and on pamela two
hands held together that MediaPipe keeps apart a little more often.

Post-processing parity: `tests/web_parity.rs` runs the app's own
`src/app/hand-pose.ts` under Node and the Rust port on 400 generated hands
(both sides, both mirror settings, several gestures): largest difference in
joint angles, features and One Euro output 1.2e-7; identical gestures. Tongue:
on the hand-labelled clips the web detector was tuned on
(`tools/eval_tongue.mjs`), the port scores AUC 0.983 (web: 0.99), recall 0.90
and precision 0.92 at 0.5.

## Light filters (low light, overexposed camera) and glasses mode

The web tracker's camera fixes are ported whole: the image filter the face
models see, the thresholds, holds, smoothing and step limits, and the glasses
weights (INTEGRATION.md, "Feature coverage", lists each value). The image
filter is `light.rs`: the frame (rescaled to the web's 640x480 processing
canvas when the camera has another size) goes through Chromium's canvas
`filter` semantics before the face detector and landmarker; hands and the
tongue detector keep the unfiltered frame, as on the web.

What Chromium does, measured in headless Chrome 156 (`--use-angle=d3d11`, the
app's WebView2 path on this machine) with probe images and real frames:

* `brightness()` / `contrast()` are FEComponentTransfer linear functions that
  Blink bakes into 8-bit tables in f32 and **truncates**
  (`trunc(slope * i + intercept * 255)`); the two tables compose. Exact on
  every input level, on the CPU and the GPU canvas.
* `saturate()` is the feColorMatrix saturate matrix, rounded to nearest
  (exact over all 16.7M colours on the CPU canvas; the GPU canvas comes out
  ~0.035 of a level low, `GPU_ROUNDING_BIAS`).
* `blur()` on the **accelerated** canvas (the app's processing canvas has
  `willReadFrequently: false`) is a separable Gaussian, radius `ceil(3σ)`,
  renormalised at the edges (transparent outside, unpremultiplied), written to
  8 bits before the colour tables run. The software canvas skips blurs this
  small (σ 0.42 / 0.65 leave the image unchanged), so a software-rendered
  WebView differs; the native filter follows the GPU canvas.
* The light meter (`drawImage` into a 32x24 software canvas) is a bilinear
  point sample whose value is truncated, at positions mapped through the
  inverse f32 scale: the luminance matches Chrome exactly at 640x480 and
  1280x960 (4e-5 off at 1280x720), and the low light filter strings
  (`blur(0.65px) brightness(2.45) contrast(1.16) saturate(0.78)`, values via
  a port of `toFixed(2)`) are identical on every test frame.

Fidelity against Chrome (`lithing-track filter` vs the app's code on the same
raw RGB frames, all three channels):

| Clip, mode | frames | filter strings equal | mean abs error per channel | max | values differing |
|---|---|---|---|---|---|
| ghana dark, low light | 300 | 300 | 0.057 / 0.050 / 0.051 | 5 (4 inside the border) | 3.4 % |
| ghana dim, low light | 300 | 300 | 0.058 / 0.058 / 0.054 | 5 | 4.1 % |
| ghana overexposed, high light | 300 | 300 | 0.002 / 0.004 / 0.004 | 1 | 0.3 % |
| ghana normal, high light | 300 | 300 | 0.006 / 0.002 / 0.112 | 1 | 4.0 % |
| dark 1280x720 camera, low light | 30 | 30 | 0.035 / 0.058 / 0.058 | 4 | 3.7 % |
| dark 1280x960 camera, low light | 60 | 60 | 0.034 / 0.057 / 0.048 | 4 | 3.5 % |

The remaining differences are the GPU's own rounding of the blur (±1 on ~1%
of values), multiplied by up to 2.8 by the brightness and contrast tables that
follow. Before the bias model the mean error was 0.11-0.14.

Cost (Ryzen 9 7900X, one thread, AVX2 path picked at run time, 300 frames,
game running, several runs): low light 0.67-0.77 ms p50 / 0.95-1.2 ms p95
per 640x480 frame (meter, blur, tables, saturate), high light 0.22-0.24 ms; a
1280x720 or 1280x960 camera adds the rescale to 640x480: 1.6-1.9 ms in all. It runs on the face thread only on frames the
face is detected (`FrameTimings::filter_ms`).

Accuracy per mode against the web tracker (`tools/webref.mjs --mode` runs the
app's own `FaceTracker`: its light code on its canvases, `processFace` for
TrackingData; `lithing-track dump --raw-size` on the same raw RGB pixels; CPU
on both sides; 300 frames at 25 fps; TrackingData compared frame by frame):

| Clip / mode | faces native / web | face landmarks mean (p95) | blendshape MAD | TrackingData mean abs diff (largest channel) | hands native / web |
|---|---|---|---|---|---|
| dark, none | 299 / **0** | - | - | - | 144 / 138 |
| dark, low light | 299 / 299 | 0.71 px (1.22) | 0.0104 | <= 0.018 (angry), head <= 0.006 | 144 / 138 |
| dim, none | 300 / 300 | 0.55 px (1.01) | 0.0072 | <= 0.013 (happy) | 214 / 219 |
| dim, low light | 300 / 300 | 0.54 px (1.02) | 0.0070 | <= 0.012 (blinkLeft) | 214 / 219 |
| overexposed, none | 300 / 300 | 0.49 px (0.91) | 0.0055 | <= 0.012 (happy) | 227 / 228 |
| overexposed, high light | 300 / 300 | 0.49 px (0.91) | 0.0054 | <= 0.013 (happy) | 227 / 228 |
| normal, high light | 300 / 300 | 0.50 px (0.90) | 0.0053 | <= 0.011 (happy) | 236 / 235 |
| monica (glasses), none | 296 / 297 | 0.45 px (0.79) | 0.0082 | <= 0.018 (blinkLeft) | 530 / 525 |
| monica (glasses), glasses mode | 296 / 297 | 0.45 px (0.79) | 0.0082 | <= 0.022 (eyeLookY) | 530 / 525 |
| normal, glasses mode | 300 / 300 | 0.50 px (0.89) | 0.0055 | <= 0.014 (happy) | 236 / 235 |

On the very dark clip without the low light mode the web MediaPipe finds no
face at all while the native detector finds it on 299 frames: the native
face detector is more sensitive on near-black input (its letterboxed
detector crop differs from MediaPipe's image-to-tensor). With the mode on,
both sides agree frame for frame. Hands are identical across modes (they
never see the filter).

## Keep frame rate (camera controls)

Low light makes many UVC webcams trade frame rate for exposure: with
"auto-exposure priority" on (UVC `CT_AE_PRIORITY_CONTROL`, the default on
e.g. the Sonix 0c45:636b), auto exposure may stretch the frame period, so 30
fps becomes 15 or 7.5. `lithing-camera-controls` keeps the rate:

* **Which controls.** Windows: `IKsControl` with
  `PROPSETID_VIDCAP_CAMERACONTROL` / `KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY`;
  `IMFExtendedCameraController` / `KSPROPERTY_CAMERACONTROL_EXTENDED_VFR`
  (only drivers or device MFTs with extended controls);
  `IAMCameraControl` exposure (log2 s, auto / manual flag);
  `IAMVideoProcAmp` gain, brightness and backlight compensation (reported,
  never changed: it moves the metering target, not the frame rate). Linux:
  `V4L2_CID_EXPOSURE_AUTO_PRIORITY`, `V4L2_CID_EXPOSURE_AUTO` +
  `V4L2_CID_EXPOSURE_ABSOLUTE` (100 us), `V4L2_CID_GAIN`, `V4L2_CID_BRIGHTNESS`.
  There is no standard control to cap automatic exposure to a range, so
  "auto exposure, but not longer than a frame" is exactly what priority = 0
  means; the manual cap is the fallback.
* **Why a second source.** nokhwa keeps its `IMFSourceReader` private and its
  `set_camera_control` keeps the current auto / manual flag (no switch to
  manual exposure) and has no priority control. The controls are opened on a
  second Media Foundation source of the same device (matched by symbolic
  link, after nokhwa started streaming so nokhwa's source stays the
  controlling one); property writes reach the driver and act on the running
  stream. On Linux the `/dev/videoN` node is opened a second time, which V4L2
  allows without touching the stream.
* **Policy** (`keep.rs`): probe 2 s -> priority / VFR off -> exposure cap with
  gain only after 3 s under 80 % of the negotiated rate -> back to automatic
  exposure after 5 bright seconds at original gain (retry after 30 s) ->
  undone and not retried if capping does not bring the rate back within 4 s
  (a camera that cannot deliver the rate in that format). Everything written
  is restored in reverse order; originals are journaled before the first write.
* **Listing without streaming** (`cameras --controls`, `list_camera_controls`):
  enumeration never opens a device; virtual cameras are never opened;
  physical ones are opened (no stream) only while no app is recorded as
  using a camera in the privacy settings (`ConsentStore\webcam`,
  `LastUsedTimeStop == 0`), unless forced.

## Phase 2: native rendering (wgpu)

Goal: the avatar is drawn by Rust into a window and/or an offscreen texture
(virtual camera), from `TrackingFrame`s directly, without a web view in the
loop. `crates/lithing-render` fixes the interfaces (`AvatarInput`, `Avatar`,
`Renderer`).

What is needed, and what exists:

| Piece | Existing crates | Gap |
|---|---|---|
| glTF loading | `gltf` (mature, complete) | none |
| VRM 0.x / 1.0 extensions (humanoid, meta, expressions/blendShapeGroups, lookAt, firstPerson) | `bevy_vrm` (VRM 0.x and 1.0 loading for Bevy, early), `vrm-rs`-style parsers are small | the extension JSON is simple to parse with serde: write our own, ~1-2k lines, mirroring @pixiv/three-vrm's loaders |
| MToon shading (lit/shade colour, shading shift/toony, rim, matcap, emission, outlines, UV animation) | `bevy_vrm` ships a partial MToon for Bevy | port three-vrm's MToon (GLSL) to WGSL; outlines need an inverted-hull pass; ~1 week incl. the app's "normal repair" and lighting presets |
| Spring bones (VRM 0 secondaryAnimation / VRMC_springBone, colliders) | `bevy_vrm` has a basic implementation | port three-vrm's spring bone solver (a few hundred lines, deterministic) |
| Expressions, morph targets, material binds | `gltf` gives morph targets | expression → morph/material bind evaluation, as in three-vrm |
| Renderer | `wgpu` (directly), `three-d` (simple, forward, no custom materials pipeline beyond its own), `rend3` (archived/unmaintained), `bevy` (complete engine, large, ECS) | Bevy + bevy_vrm is the fastest path to pixels but brings a whole engine and its release cadence; raw wgpu with a small forward renderer (skinning, morphs, MToon, outlines, MSAA, transparency sorting) is ~5-8k lines and fits the app better (one avatar, a few props, an offscreen target). Recommendation: raw wgpu, borrowing bevy_vrm's parsing ideas. |
| Virtual camera / OBS | the app already has `virtual_camera.rs` | feed it the wgpu offscreen texture (copy to a mapped buffer) instead of reading back the WebGL canvas |
| Props, lighting presets, hood/normal fixes the app has | | port case by case from renderer.ts / normal-repair.ts |

Transition: phase 2 can start with the web UI still on top: a native wgpu
child window under a transparent web view, or the web view only for settings
while the avatar window is native. Tracking data then never crosses into JS.

## Phase 3: native UI (optional)

Only worth doing if, after phase 2, the web view itself costs too much
(memory ~150-300 MB for WebView2, startup). Options:

| Toolkit | Fit |
|---|---|
| **egui** | Immediate mode, renders through wgpu (same device as the avatar), trivial to embed in the phase 2 renderer; looks "tool-like", theming is limited; fastest to build settings panels with. |
| **Slint** | Declarative `.slint` markup, designer-friendly, good on embedded/low-end, renders via its own software/femtovg/skia backends or wgpu interop; licence: GPLv3 / royalty-free for desktop apps / commercial. |
| **iced** | Elm-like, wgpu based, pure Rust; fine for settings UIs, less mature text/IME and accessibility. |

Extensions are the hard constraint: the app has a JavaScript SDK
(extension-runtime.ts, sdk-ui.ts: network, hotkeys, UI panels). To keep them:

* **Embedded JS runtime** exposing the same `api.*` surface: `rquickjs`
  (QuickJS: ~1 MB, no JIT, easy to sandbox per extension, fast enough for
  glue code) or `deno_core` (V8: fast, modern, but ~30 MB and heavier
  embedding). Recommendation: QuickJS first; extensions rarely need speed.
* **UI as data**: sdk-ui.ts already describes panels declaratively; the
  native shell renders the same description with its own widgets
  (`lithing-shell::Widget`).
* **WASM components** (wasmtime) as a second, sandboxed extension format for
  new extensions in any language.

`crates/lithing-shell` sketches the runtime trait, the widget model and the
shared permission check.

## Android

* Tracking: ORT ships Android builds with the **NNAPI** and **XNNPACK**
  providers (and QNN for Snapdragon NPUs); the same `.onnx` files run there.
  `ort` supports `aarch64-linux-android` (dynamic loading of
  `libonnxruntime.so` from the APK). Camera input would come from the app
  (CameraX on the Kotlin side → `ImageReader` YUV planes → JNI → `Frame`),
  since nokhwa has no Android backend. Thread priorities map to
  `setpriority`/`Process.setThreadPriority`.
* The current phone link (the Android app as a camera for the desktop) can
  stay as is: the desktop tracks the streamed frames natively instead of in
  the web view (a `FrameSource` fed from the WebRTC decoder).
* Rendering on Android in phase 2: wgpu runs on Vulkan/GLES there.
