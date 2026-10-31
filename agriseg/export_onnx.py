"""Export a trained AgriSegNet to ONNX (aux head already dropped) for TensorRT / OpenVINO.

    python export_onnx.py --ckpt runs/agri_s0/best.pt --imgsz 640 --out agrisegnet.onnx
    # on the Jetson:
    trtexec --onnx=agrisegnet.onnx --fp16 --saveEngine=agri_fp16.engine --warmUp=500 --iterations=200
    # YOLO26 for the SAME comparison:
    yolo export model=yolo26n-seg.pt format=engine half=True imgsz=640
    # power: run `tegrastats --interval 200` (or an inline USB-C meter) during the benchmark loop.
"""
import argparse

import torch

from engine import build_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="agrisegnet.onnx")
a = ap.parse_args()

model, ck = build_from_ckpt(a.ckpt)
s = (a.imgsz + 127) // 128 * 128   # multiple of 128 => static AveragePool, no Gather/Where ops
if s != a.imgsz:
    print(f'note: imgsz {a.imgsz} -> {s} (multiple of 128 required for a clean TensorRT graph)')
x = torch.randn(1, 3, s, s)
torch.onnx.export(model, x, a.out, opset_version=17, input_names=["image"], output_names=["logits"],
                  dynamic_axes=None)
print("saved", a.out, "input", tuple(x.shape))
