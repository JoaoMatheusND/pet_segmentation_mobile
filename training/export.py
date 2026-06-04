"""
Export pipeline: PyTorch UNet → ONNX → TF SavedModel → TFLite FP32

Run after training:
    python export.py

Output: model.tflite  →  copy to android/app/src/main/assets/
"""
import subprocess
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import tensorflow as tf

from model import UNet

CHECKPOINT  = "best.pth"
ONNX_PATH   = "unet.onnx"
SAVED_MODEL = "saved_model"
TFLITE_PATH = "model.tflite"
IMG_SIZE    = 256


class NHWCWrapper(nn.Module):
    """Wraps UNet to add NCHW→NHWC permute at output for TFLite."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 3, 1)   # [B,C,H,W] → [B,H,W,C]


def export_onnx(export_model: nn.Module) -> None:
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        export_model, dummy, ONNX_PATH,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print(f"  ONNX exported and verified: {ONNX_PATH}")


def onnx_to_savedmodel() -> None:
    result = subprocess.run(
        ["onnx2tf", "-i", ONNX_PATH, "-o", SAVED_MODEL,
         "--non_verbose", "--overwrite_saved_model"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("onnx2tf stderr:\n", result.stderr)
        sys.exit(1)
    print(f"  TF SavedModel: {SAVED_MODEL}/")


def savedmodel_to_tflite() -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    # No optimizations → full FP32 (task requirement)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite saved: {TFLITE_PATH}  ({len(tflite_model)/1e6:.1f} MB)")


def verify_tflite() -> None:
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  Input  shape={in_d['shape']}  dtype={in_d['dtype'].__name__}")
    print(f"  Output shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")

    dummy_np = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    interp.set_tensor(in_d["index"], dummy_np)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])

    assert list(out.shape) == [1, IMG_SIZE, IMG_SIZE, 3], \
        f"Unexpected output shape: {out.shape}"
    print("  Shape verification passed ✓")


def main() -> None:
    device = torch.device("cpu")
    base   = UNet(in_channels=3, num_classes=3)
    base.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    base.eval()

    export_model = NHWCWrapper(base).eval()

    print("Step 1/4  Export ONNX...")
    export_onnx(export_model)

    print("Step 2/4  ONNX → TF SavedModel...")
    onnx_to_savedmodel()

    print("Step 3/4  SavedModel → TFLite FP32...")
    savedmodel_to_tflite()

    print("Step 4/4  Verify TFLite...")
    verify_tflite()

    print(f"\nDone.  Next: cp {TFLITE_PATH} ../android/app/src/main/assets/model.tflite")


if __name__ == "__main__":
    main()
