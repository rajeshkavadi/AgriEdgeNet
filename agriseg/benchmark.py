"""Deployment benchmark: every model x every input size x {FP32, INT8}, all run as ONNX on ONNX Runtime.

    python benchmark.py --data <export> --split test --sizes 640 480 320 \
        --custom runs/nano_s0/best.pt runs/nano_s1/best.pt --yolo yolo/yolo26n_scratch/weights/best.pt \
        --yolo-conf 0.25 --out results/benchmark

Same treatment for both model families, so the comparison is like for like:
  * export to a static ONNX graph at each input size;
  * INT8 = ONNX Runtime static quantisation, QDQ, per-channel, Conv/MatMul only, calibrated on the same
    `--calib` training images (each model with its own preprocessing);
  * accuracy = semantic mIoU on the split at the original label resolution (YOLO instances -> semantic map,
    highest confidence wins, conf fixed from the valid split);
  * latency = median ONNX Runtime CPU time of the graph alone (no pre/post-processing), `--threads` threads;
  * paired bootstrap 95 % CI of (custom - YOLO) per-image mIoU for each size x precision.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import IMG_EXT, CropSegEval, find_split_dir, preprocess
from engine import build_from_ckpt
from evaluate import YoloPredictor, img_miou_crops, model_name, paired_bootstrap
from metrics import ConfusionMatrix
from onnx_tools import _square, export, export_size, quantize_int8


# ----------------------------------------------------------------------------- calibration readers
def _calib_images(root, n, seed=0):
    imgs = sorted(p for p in (find_split_dir(root, "train") / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    idx = np.random.default_rng(seed).choice(len(imgs), size=min(n, len(imgs)), replace=False)
    return [imgs[i] for i in idx]


def _reader(images, feed):
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(images)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else feed(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    return R()


def letterbox(rgb, size):
    """Ultralytics-style input: long side -> size, pad (114) to size x size, /255, NCHW."""
    h, w = rgb.shape[:2]
    r = size / max(h, w)
    im = cv2.resize(rgb, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR)
    c = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    c[top:top + im.shape[0], left:left + im.shape[1]] = im
    return c.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- per-model runners
class CustomOnnx:
    def __init__(self, path, imgsz, size, threads):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.imgsz, self.size = imgsz, size
        self.feed_shape = (1, 3, size, size)

    def __call__(self, batch):
        nh, nw = batch["valid_hw"]
        logits = torch.from_numpy(self.sess.run(None, {"image": _square(batch["image"], self.size)})[0])
        if logits.shape[-1] != self.size:                       # output-stride 8 graphs
            logits = F.interpolate(logits, size=(self.size, self.size), mode="bilinear", align_corners=False)
        logits = F.interpolate(logits[:, :, :nh, :nw], size=tuple(batch["gt"].shape), mode="bilinear",
                               align_corners=False)
        return logits.argmax(1)[0].numpy().astype(np.uint8)

    def latency_ms(self, n=30):
        return _latency(self.sess, "image", self.feed_shape, n)


class YoloOnnx(YoloPredictor):
    def __init__(self, path, size, conf, threads):
        super().__init__(str(path), size, conf, "cpu")
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.feed_shape = (1, 3, size, size)

    def latency_ms(self, n=30):
        return _latency(self.sess, self.sess.get_inputs()[0].name, self.feed_shape, n)


def _latency(sess, name, shape, n):
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {name: x}); t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


# ----------------------------------------------------------------------------- exports
def export_custom(ckpt, size, out, calib):
    model, ck = build_from_ckpt(ckpt)
    s = export_size(ck["cfg"].get("arch", "agrisegnet"), size)
    fp32 = export(model, out / f"{model_name(ckpt)}_{s}_fp32.onnx", s)
    int8 = quantize_int8(fp32, out / f"{model_name(ckpt)}_{s}_int8.onnx",
                         _reader(calib, lambda rgb: {"image": _square(preprocess(rgb, size)[0], s)}))
    n_params = sum(p.numel() for p in model.parameters())
    return s, {"fp32": fp32, "int8": int8}, n_params


def export_yolo(weights, size, out, calib):
    from ultralytics import YOLO
    y = YOLO(str(weights))
    n_params = sum(p.numel() for p in y.model.parameters())
    tmp = Path(y.export(format="onnx", imgsz=size, simplify=True, dynamic=False, verbose=False))
    fp32 = out / f"{model_name(weights)}_{size}_fp32.onnx"
    tmp.replace(fp32)
    int8 = quantize_int8(fp32, out / f"{model_name(weights)}_{size}_int8.onnx",
                         _reader(calib, lambda rgb: {"images": letterbox(rgb, size)}))
    return size, {"fp32": fp32, "int8": int8}, n_params


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 320])
    ap.add_argument("--custom", nargs="*", default=[])
    ap.add_argument("--yolo", nargs="*", default=[])
    ap.add_argument("--yolo-conf", type=float, nargs="*", default=[0.25], help="one per --yolo, or one for all")
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1, help="ORT CPU threads for latency (1 = edge-like)")
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    out = Path(a.out); (out / "onnx").mkdir(parents=True, exist_ok=True)
    calib = _calib_images(a.data, a.calib)
    confs = a.yolo_conf if len(a.yolo_conf) == len(a.yolo) else [a.yolo_conf[0]] * len(a.yolo)

    rows, boot = [], {}
    for size in a.sizes:
        ds = CropSegEval(a.data, a.split, size)
        names = ["background"] + list(ds.names)
        runners = []
        for ck in a.custom:
            s, files, n = export_custom(ck, size, out / "onnx", calib)
            for prec, f in files.items():
                runners.append((model_name(ck), "custom", prec, n, f, CustomOnnx(f, size, s, a.threads)))
        for w, c in zip(a.yolo, confs):
            s, files, n = export_yolo(w, size, out / "onnx", calib)
            for prec, f in files.items():
                runners.append((f"yolo_{model_name(w)}", "yolo", prec, n, f, YoloOnnx(f, s, c, a.threads)))

        cms = {r[0] + r[2]: ConfusionMatrix(len(names)) for r in runners}
        per_img = {k: [] for k in cms}
        for i in range(len(ds)):
            b = ds[i]
            gt = b["gt"].numpy().astype(np.uint8)
            for name, _, prec, _, _, run in runners:
                pm = run(b)
                cms[name + prec].update(gt, pm)
                one = ConfusionMatrix(len(names)); one.update(gt, pm)
                per_img[name + prec].append(img_miou_crops(one))
        for name, kind, prec, n, f, run in runners:
            r = cms[name + prec].summary(names)
            rows.append({"model": name, "kind": kind, "size": size, "precision": prec, "params": n,
                         "onnx_kib": round(f.stat().st_size / 1024, 1), "latency_ms_cpu": run.latency_ms(),
                         **r})
            print(f"{size:>4}px {prec} {name:<28s} mIoU(crops) {r['mIoU_crops']:.4f}  pix acc {r['pixel_acc']:.4f}  "
                  f"{rows[-1]['latency_ms_cpu']:.1f} ms")
        for prec in ("fp32", "int8"):
            for cn, ck_, cp, *_ in [r for r in runners if r[1] == "custom" and r[2] == prec]:
                for yn, yk, yp, *_ in [r for r in runners if r[1] == "yolo" and r[2] == prec]:
                    m, lo, hi = paired_bootstrap(per_img[cn + prec], per_img[yn + prec])
                    boot[f"{size}px {prec}: {cn} - {yn}"] = (m, lo, hi)
                    print(f"   {size}px {prec}: {cn} - {yn}: dmIoU {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    json.dump({"rows": rows, "paired_bootstrap_mIoU_crops(mean,lo95,hi95)": boot, "args": vars(a)},
              open(out / "benchmark.json", "w"), indent=2)
    print(f"\n{'size':>4} {'prec':<5}{'model':<28}{'params':>11}{'ONNX KiB':>10}{'CPU ms':>8}{'mIoU crops':>12}")
    for r in sorted(rows, key=lambda r: (-r["size"], r["precision"], r["kind"], r["model"])):
        print(f"{r['size']:>4} {r['precision']:<5}{r['model']:<28}{r['params']:>11,}{r['onnx_kib']:>10.0f}"
              f"{r['latency_ms_cpu']:>8.1f}{r['mIoU_crops']:>12.4f}")


if __name__ == "__main__":
    main()
