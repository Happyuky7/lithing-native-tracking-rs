# Third-party notices

Lithing Native Tracking is licensed under the [MPL-2.0](LICENSE). It uses the components
below, each under its own license. When you ship a program built with this engine, ship
these notices with it.

## Models

| Component | License | Notes |
|---|---|---|
| MediaPipe Face Landmarker models (`models/face_detector.onnx`, `face_landmarks_detector.onnx`, `face_blendshapes.onnx`) | Apache-2.0 | Copyright Google LLC. Converted to ONNX from `face_landmarker.task` without retraining; see [MODELS.md](MODELS.md). https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker |
| MediaPipe Hand Landmarker models (`models/hand_detector.onnx`, `hand_landmarks_detector.onnx`) | Apache-2.0 | Copyright Google LLC. Converted to ONNX from `hand_landmarker.task` without retraining; see [MODELS.md](MODELS.md). https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |

MediaPipe is a trademark of Google LLC. This project is not affiliated with or endorsed by Google.

## Runtime

| Component | License | Notes |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Copyright (c) Microsoft Corporation. Its own notices: https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt |
| [ort](https://github.com/pykeio/ort) (Rust bindings to ONNX Runtime) | MIT OR Apache-2.0 | Copyright pyke.io |
| DirectML (`DirectML.dll`, Windows builds with the `directml` feature) | Microsoft DirectML license | Copyright (c) Microsoft Corporation. Redistributed unmodified under the license of the [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML) package |

## Rust crates

| Crate | License | Used by |
|---|---|---|
| [nokhwa](https://github.com/l1npengtul/nokhwa) | Apache-2.0 | Camera capture (`lithing-tracking`, `camera` feature) |
| [windows](https://github.com/microsoft/windows-rs), windows-sys | MIT OR Apache-2.0 | Media Foundation camera controls, Windows APIs |
| [libc](https://github.com/rust-lang/libc) | MIT OR Apache-2.0 | V4L2 camera controls on Linux |
| [crossbeam-channel](https://github.com/crossbeam-rs/crossbeam) | MIT OR Apache-2.0 | Newest-frame-wins mailboxes between threads |
| [serde](https://github.com/serde-rs/serde), serde_json | MIT OR Apache-2.0 | Settings and results |
| [thiserror](https://github.com/dtolnay/thiserror), [anyhow](https://github.com/dtolnay/anyhow) | MIT OR Apache-2.0 | Errors |
| [tauri](https://github.com/tauri-apps/tauri), tauri-plugin | MIT OR Apache-2.0 | `lithing-tauri-plugin` only |
| [clap](https://github.com/clap-rs/clap) | MIT OR Apache-2.0 | `lithing-tracking-cli` only |
| [sysinfo](https://github.com/GuillaumeGomez/sysinfo) | MIT | `lithing-tracking-cli` only |
| [sha2](https://github.com/RustCrypto/hashes) | MIT OR Apache-2.0 | `lithing-tracking-cli` only |

Their transitive dependencies keep their own licenses; `cargo tree` lists them and tools
such as `cargo about` or `cargo deny` can produce a full report for a release.
