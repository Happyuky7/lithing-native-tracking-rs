#!/usr/bin/env sh
# Linux/macOS version of fetch_models.ps1. PYTHON may point at a Python 3.12 with tools/requirements.txt.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
models="$root/models"
base=https://storage.googleapis.com/mediapipe-models
check() { (cd "$models" && grep " $1\$" SHA256SUMS | sha256sum -c -); }
[ -f "$models/face_landmarker.task" ] || curl -fL -o "$models/face_landmarker.task" "$base/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
[ -f "$models/hand_landmarker.task" ] || curl -fL -o "$models/hand_landmarker.task" "$base/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
check face_landmarker.task
check hand_landmarker.task
onnx="face_detector.onnx face_landmarks_detector.onnx face_blendshapes.onnx hand_detector.onnx hand_landmarks_detector.onnx"
missing=0
for f in $onnx; do [ -f "$models/$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  python=${PYTHON:-}
  if [ -z "$python" ]; then
    python3.12 -m venv "$models/work/venv"
    "$models/work/venv/bin/python" -m pip install -q -r "$root/tools/requirements.txt"
    python="$models/work/venv/bin/python"
  fi
  "$python" -I "$root/tools/convert_models.py" "$models" "$models"
fi
for f in $onnx; do check "$f"; done
echo "models ready in $models"
