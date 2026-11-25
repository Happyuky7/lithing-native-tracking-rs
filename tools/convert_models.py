"""Converts the MediaPipe .task bundles (Apache-2.0) to the ONNX models lithing-tracking loads.

Usage (Python 3.12 venv with the pinned packages in tools/requirements.txt):
    python tools/convert_models.py <dir with face_landmarker.task and hand_landmarker.task> <out dir>

Each .task is a zip of .tflite files. tf2onnx converts them graph-for-graph (float16
weights are dequantized to float32 by a Cast, so the ONNX models run on every EP).
Afterwards each ONNX model is run next to the TFLite interpreter on random input and
the largest output difference is printed, so a bad conversion cannot go unnoticed.
"""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

MODELS = {
    # task bundle -> [(tflite inside, onnx name)]
    "face_landmarker.task": [
        ("face_detector.tflite", "face_detector.onnx"),
        ("face_landmarks_detector.tflite", "face_landmarks_detector.onnx"),
        ("face_blendshapes.tflite", "face_blendshapes.onnx"),
    ],
    "hand_landmarker.task": [
        ("hand_detector.tflite", "hand_detector.onnx"),
        ("hand_landmarks_detector.tflite", "hand_landmarks_detector.onnx"),
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize(graph) -> None:
    """tf2onnx numbers the tensors it creates in an order that varies between runs.
    Renaming every tensor and node by first use, and ordering the initializers the
    same way, makes the output byte-for-byte reproducible (checksums in models/SHA256SUMS)."""
    keep = {v.name for v in list(graph.input) + list(graph.output)}
    names = {}

    def rename(name: str) -> str:
        if not name or name in keep:
            return name
        if name not in names:
            names[name] = f"t{len(names)}"
        return names[name]

    for index, node in enumerate(graph.node):
        node.name = f"n{index}_{node.op_type}"
        for i, name in enumerate(node.input):
            node.input[i] = rename(name)
        for i, name in enumerate(node.output):
            node.output[i] = rename(name)
    for initializer in graph.initializer:
        initializer.name = rename(initializer.name)
    ordered = sorted(graph.initializer, key=lambda t: int(t.name[1:]) if t.name[1:].isdigit() else -1)
    copies = [type(t)() for t in ordered]
    for copy, original in zip(copies, ordered):
        copy.CopyFrom(original)
    del graph.initializer[:]
    graph.initializer.extend(copies)
    for info in graph.value_info:
        info.name = rename(info.name)


def fix_batch(path: Path) -> None:
    """Pins symbolic dimensions (the batch) to 1: DirectML compiles static shapes best.
    Also drops the doc string tf2onnx fills with the local input path, so the bytes do not depend on it."""
    import onnx

    model = onnx.load(str(path))
    model.graph.doc_string = ""
    canonicalize(model.graph)
    for value in list(model.graph.input) + list(model.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.Clear()
                dim.dim_value = 1
    onnx.save(model, str(path))


def verify(tflite: Path, onnx_path: Path) -> float:
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    feeds = {}
    for detail, onnx_input in zip(interpreter.get_input_details(), session.get_inputs()):
        data = rng.random(detail["shape"], dtype=np.float32)
        interpreter.set_tensor(detail["index"], data)
        feeds[onnx_input.name] = data
    interpreter.invoke()
    onnx_outputs = session.run(None, feeds)
    tflite_outputs = {d["name"]: interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()}
    worst = 0.0
    for meta, value in zip(session.get_outputs(), onnx_outputs):
        reference = tflite_outputs.get(meta.name)
        if reference is None:
            # tf2onnx keeps the TFLite output names; fall back to shape matching.
            reference = next(v for v in tflite_outputs.values() if v.size == value.size)
        worst = max(worst, float(np.max(np.abs(reference.reshape(value.shape) - value))))
    return worst


def main() -> int:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    for bundle, entries in MODELS.items():
        with zipfile.ZipFile(source / bundle) as archive:
            for inner, onnx_name in entries:
                tflite = work / inner
                tflite.write_bytes(archive.read(inner))
                target = out / onnx_name
                # A fixed hash seed makes tf2onnx's naming (and so the file bytes) reproducible.
                env = dict(os.environ, PYTHONHASHSEED="0", TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="2")
                subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--tflite", str(tflite), "--output", str(target), "--opset", "17"], check=True, env=env)
                fix_batch(target)
                error = verify(tflite, target)
                print(f"{onnx_name}: sha256 {sha256(target)}  max |onnx - tflite| = {error:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
