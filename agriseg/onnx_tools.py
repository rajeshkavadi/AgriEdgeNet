"""Device-agnostic deployment checks: ONNX export (FP32), static INT8 quantisation (QDQ, per-channel) and
accuracy of the exported graphs with ONNX Runtime on a dataset split.

    python onnx_tools.py --ckpt runs/nano/best.pt --data <export> --imgsz 640 --out deploy/
writes deploy/model_fp32.onnx, deploy/model_int8.onnx and deploy/onnx_eval.json (mIoU FP32 vs INT8).

The QDQ INT8 graph is what TensorRT (Jetson), ONNX Runtime (Pi / x86 / ARM) and OpenVINO consume directly.
For microcontrollers convert the FP32 ONNX to TFLite INT8 (e.g. onnx2tf + TFLite converter) with the same
calibration images; that step is not automated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CropSegEval, find_split_dir, IMG_EXT, preprocess
from engine import build_from_ckpt
from metrics import ConfusionMatrix


def export(model, path, size, output_stride=1):
    model.output_stride = output_stride if hasattr(model, "output_stride") else 1
    x = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, x, str(path), opset_version=13, input_names=["image"], output_names=["logits"],
                      dynamic_axes=None, dynamo=False)
    model.output_stride = 1
    return path


def _square(x, size):
    """Pad a preprocessed CHW tensor to size x size (the exported graph has a fixed input shape)."""
    c, h, w = x.shape
    out = np.zeros((1, c, size, size), np.float32)
    out[0, :, :min(h, size), :min(w, size)] = x[:, :size, :size].numpy()
    return out


def calibration_reader(root, imgsz, size, n=100, seed=0):
    from onnxruntime.quantization import CalibrationDataReader
    import cv2
    imgs = sorted(p for p in (find_split_dir(root, "train") / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    rng = np.random.default_rng(seed)
    pick = [imgs[i] for i in rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(pick)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            x, _ = preprocess(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), imgsz)
            return {"image": _square(x, size)}
    return Reader()


QUANT_OPS = ["Conv", "MatMul"]   # same INT8 recipe for every model compared: weights + inputs of Conv/MatMul


def export_size(arch, imgsz):
    m = 128 if arch == "agrisegnet" else 32
    return (imgsz + m - 1) // m * m


def quantize_int8(fp32, int8, reader):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = Path(str(fp32).replace(".onnx", "_pre.onnx"))
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, op_types_to_quantize=QUANT_OPS)
    pre.unlink(missing_ok=True)
    import onnx                      # keep the FP32 model's metadata (Ultralytics reads task/names/imgsz from it)
    src, q = onnx.load(str(fp32)), onnx.load(str(int8))
    del q.metadata_props[:]
    q.metadata_props.extend(src.metadata_props)
    onnx.save(q, str(int8))
    return int8


def evaluate_onnx(path, root, split, imgsz, size, output_stride=1):
    import onnxruntime as ort
    import torch.nn.functional as F
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ds = CropSegEval(root, split, imgsz)
    cm = ConfusionMatrix(ds.nc + 1)
    for i in range(len(ds)):
        b = ds[i]
        nh, nw = b["valid_hw"]
        logits = torch.from_numpy(sess.run(None, {"image": _square(b["image"], size)})[0])
        if output_stride != 1:
            logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
        logits = F.interpolate(logits[:, :, :nh, :nw], size=tuple(b["gt"].shape), mode="bilinear", align_corners=False)
        cm.update(b["gt"].numpy().astype(np.uint8), logits.argmax(1)[0].numpy().astype(np.uint8))
    return cm.summary(["background"] + list(ds.names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib", type=int, default=100, help="number of train images for INT8 calibration")
    ap.add_argument("--output-stride", type=int, default=1, choices=[1, 8],
                    help="8: graph emits 1/8-resolution logits (upsample on the host); nano only")
    ap.add_argument("--out", default="deploy")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, ck = build_from_ckpt(a.ckpt)
    size = export_size(ck["cfg"].get("arch", "agrisegnet"), a.imgsz)
    fp32 = export(model, out / "model_fp32.onnx", size, a.output_stride)
    int8 = quantize_int8(fp32, out / "model_int8.onnx", calibration_reader(a.data, a.imgsz, size, a.calib))
    res = {"input": [1, 3, size, size], "output_stride": a.output_stride}
    for tag, p in (("fp32", fp32), ("int8", int8)):
        r = evaluate_onnx(p, a.data, a.split, a.imgsz, size, a.output_stride)
        res[tag] = {"file_kib": round(p.stat().st_size / 1024, 1), **r}
        print(f"{tag}: {p.name} {res[tag]['file_kib']} KiB  mIoU(crops) {r['mIoU_crops']:.4f}  pixel acc {r['pixel_acc']:.4f}")
    res["int8_drop_mIoU_crops"] = res["fp32"]["mIoU_crops"] - res["int8"]["mIoU_crops"]
    json.dump(res, open(out / "onnx_eval.json", "w"), indent=2)
    print(f"INT8 mIoU drop: {res['int8_drop_mIoU_crops']:+.4f}")


if __name__ == "__main__":
    main()
