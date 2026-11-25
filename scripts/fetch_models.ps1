# Downloads MediaPipe's face and hand landmarker bundles (Apache-2.0) from Google,
# verifies their SHA-256, converts them to ONNX (tools/convert_models.py) and
# verifies the ONNX files against models/SHA256SUMS.
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_models.ps1 [-Python <python with tools/requirements.txt>]
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root "models"
$sums = @{}
Get-Content (Join-Path $models "SHA256SUMS") | ForEach-Object { $hash, $name = $_ -split "\s+", 2; $sums[$name.Trim()] = $hash }
$urls = @{
  "face_landmarker.task" = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
  "hand_landmarker.task" = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
}
function Check($file) {
  $name = Split-Path -Leaf $file
  $actual = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
  if ($actual -ne $sums[$name]) { throw "$name checksum mismatch: $actual (expected $($sums[$name]))" }
  Write-Host "ok  $name"
}
foreach ($name in $urls.Keys) {
  $target = Join-Path $models $name
  if (-not (Test-Path $target)) { Write-Host "downloading $name"; Invoke-WebRequest -Uri $urls[$name] -OutFile $target }
  Check $target
}
$onnx = "face_detector.onnx","face_landmarks_detector.onnx","face_blendshapes.onnx","hand_detector.onnx","hand_landmarks_detector.onnx"
if (($onnx | Where-Object { -not (Test-Path (Join-Path $models $_)) }).Count -gt 0) {
  if (-not $Python) {
    $venv = Join-Path $models "work\venv"
    if (-not (Test-Path $venv)) {
      Write-Host "creating a Python 3.12 venv for the conversion"
      & py -3.12 -m venv $venv
      & "$venv\Scripts\python.exe" -m pip install -q -r (Join-Path $root "tools\requirements.txt")
    }
    $Python = "$venv\Scripts\python.exe"
  }
  & $Python -I (Join-Path $root "tools\convert_models.py") $models $models
}
foreach ($name in $onnx) { Check (Join-Path $models $name) }
Write-Host "models ready in $models"
