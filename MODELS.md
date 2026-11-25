# Models

lithing-tracking runs the **same models the web app runs** (MediaPipe Tasks
Vision `face_landmarker.task` and `hand_landmarker.task`, the files in the main
repo's `public/mediapipe/`), converted from TFLite to ONNX. Nothing is
retrained or swapped for a different network, so the outputs can be compared
with the web tracker one to one (see "Accuracy" in ARCHITECTURE.md).

The model files are **not in git** (`models/*.onnx`, `*.task` are ignored).
`scripts/fetch_models.ps1` (Windows) or `scripts/fetch_models.sh` (Linux)
downloads the two bundles from Google, checks their SHA-256, converts them with
`tools/convert_models.py` and checks the ONNX files against
`models/SHA256SUMS`. The conversion is byte-for-byte reproducible (see below),
so the ONNX checksums are fixed too.

## Sources

| Bundle | URL | License | SHA-256 |
|---|---|---|---|
| face_landmarker.task (float16, v1) | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | Apache-2.0 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| hand_landmarker.task (float16, v1) | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task | Apache-2.0 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |

Both URLs serve the same bytes as their `/latest/` aliases (checked
2026-11-08) and as the main app's `public/mediapipe/*.task`. Licence: the
MediaPipe model cards publish these models under Apache-2.0
(https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker and
.../hand_landmarker, "Models" section); the MediaPipe code is Apache-2.0 as
well (https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). Apache-2.0
allows redistribution and modification (conversion) with attribution and the
licence text; ship `LICENSE-APACHE` / a NOTICE entry next to the models when
bundling them in the app.

## Converted models

| ONNX file | From (inside the .task) | Input | Outputs | SHA-256 |
|---|---|---|---|---|
| face_detector.onnx | face_detector.tflite (BlazeFace short range) | 1x128x128x3, RGB in [-1, 1], letterboxed | regressors 1x896x16, scores 1x896x1 (logits) | `893ed2d843c6f09035a35e65910b099d6db0290b7123cd402935ff36c67f1f19` |
| face_landmarks_detector.onnx | face_landmarks_detector.tflite (478 points with irises) | 1x256x256x3, RGB in [0, 1], rotated face crop | 1434 = 478 x (x, y, z) in crop pixels, face flag (logit), 1x1 | `ebdfd54693e9f8922287a3d4b1ffcd9a45614d788badd5883680e5ce1933d363` |
| face_blendshapes.onnx | face_blendshapes.tflite (MLP-Mixer) | 1x146x2, the 146 landmark subset in image pixels | 52 scores | `dfd7ed349018b83e53103a05ab83923edb2929351b6188118228f34495ef891d` |
| hand_detector.onnx | hand_detector.tflite (palm detector, full) | 1x192x192x3, RGB in [0, 1], letterboxed | regressors 1x2016x18, scores 1x2016x1 (logits) | `5c0cd92a0663f187d7a94ac5f8b3685ff4479f57906f8235c49b54217d379fed` |
| hand_landmarks_detector.onnx | hand_landmarks_detector.tflite (full) | 1x224x224x3, RGB in [0, 1], rotated hand crop | 63 screen landmarks (crop pixels), presence (probability), handedness (probability of right, selfie convention), 63 world landmarks (metres) | `59d18e97654c71c3350ecd0196444186f0010f922715612f4afad43be2916d32` |

Pre- and post-processing constants (anchors, crop scales, rotation keypoints,
thresholds, the 146-landmark blendshape subset) follow MediaPipe's graphs; the
blendshape subset was read from the app's own `vision_wasm_internal.wasm` to be
sure it matches the build the web app runs.

## How the conversion was done

* Tool: [tf2onnx](https://github.com/onnx/tensorflow-onnx) 1.17.0 (Apache-2.0)
  with TensorFlow 2.21 (CPU), opset 17, in a Python 3.12 venv
  (`tools/requirements.txt`). The machine's default Python 3.15 has no pip;
  the uv-managed CPython 3.12 does.
* float16 weights are dequantized to float32 by the converter (constant
  folded), so every execution provider runs them.
* `tools/convert_models.py` then pins the batch dimension to 1 (DirectML likes
  static shapes), drops the doc string (it contains the local path) and renames
  tensors and nodes by first use: tf2onnx numbers generated tensors in an order
  that changes from run to run, which made the bytes (not the maths) differ.
  With that, two conversions produce identical files.
* Verification, printed by the script: each ONNX model and the original TFLite
  model run on the same random input; largest output difference:
  face_detector 8.2e-5, face_landmarks 4.3e-4 (in pixels of a 256 crop),
  face_blendshapes 9.8e-7, hand_detector 1.3e-4, hand_landmarks 1.5e-4.

## Alternatives considered (not used)

* **Ready-made ONNX conversions** (PINTO0309's model zoo, onnx-community on
  Hugging Face) exist for several MediaPipe models, but they are conversions
  of other releases (for example the legacy 192x192 face mesh) and none was
  checked to be the exact 2023 Tasks bundle with the blendshapes model.
  Converting the very files the app ships is the only way to guarantee the
  native and web outputs match, and keeps the provenance to a single Google URL
  per bundle.
