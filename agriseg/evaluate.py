"""Like-for-like comparison of AgriSegNet checkpoint(s) against YOLO26 segmentation weights.

    python evaluate.py --data /path/to/roboflow_export --split test --imgsz 640 \
        --custom runs/agri_s0/best.pt runs/agri_s1/best.pt runs/agri_s2/best.pt \
        --yolo yolo26n_seg_best.pt yolo26s_seg_best.pt --yolo-conf 0.25 --out results/compare

Fairness rules enforced here:
  * every model is scored on the SAME images at the ORIGINAL label resolution,
  * YOLO instance masks are converted to one semantic map (higher-confidence instance wins),
  * mIoU(crops) excludes background and is the headline number; mIoU(all) is also reported,
  * a paired bootstrap over test images gives a CI for (custom - yolo).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from agrisegnet import count_flops, count_params
from dataset import CropSegEval, eval_collate
from engine import build_from_ckpt, predict_semantic
from metrics import ConfusionMatrix, instances_to_semantic

PALETTE = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)


def img_miou_crops(cm: ConfusionMatrix):
    tp = np.diag(cm.m).astype(float)
    union = cm.m.sum(0) + cm.m.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou[1:])) if np.any(~np.isnan(iou[1:])) else np.nan


def model_name(path) -> str:
    """Readable unique label: runs/agri_s0/best.pt -> agri_s0 ; runs/x/weights/best.pt -> x."""
    p = Path(path)
    d = p.parent.parent.name if p.parent.name == "weights" else p.parent.name
    return d if p.stem in ("best", "last") else f"{d}_{p.stem}" if d else p.stem


class CustomPredictor:
    def __init__(self, ckpt, device, amp, tta):
        self.model, self.ck = build_from_ckpt(ckpt, device)
        self.device, self.amp, self.tta = device, amp, tta
        self.name = model_name(ckpt)

    def __call__(self, batch):
        return predict_semantic(self.model, batch, self.device, self.amp, self.tta)


_YOLO_CACHE = {}


class YoloPredictor:
    def __init__(self, weights, imgsz, conf, device, tag_conf=False):
        from ultralytics import YOLO
        if weights not in _YOLO_CACHE:                 # one network in memory per weights file
            _YOLO_CACHE[weights] = YOLO(weights)
        self.m = _YOLO_CACHE[weights]
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.name = "yolo_" + model_name(weights) + (f"@conf{conf:g}" if tag_conf else "")

    def __call__(self, batch):
        h, w = batch["gt"].shape
        r = self.m.predict(batch["path"], imgsz=self.imgsz, conf=self.conf, device=self.device,
                           verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.masks) == 0:
            return np.zeros((h, w), np.uint8)
        return instances_to_semantic(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int),
                                     r.boxes.conf.cpu().numpy(), h, w)


def bench_custom(pred: CustomPredictor, imgsz, device, n=100):
    m = pred.model
    x = torch.randn(1, 3, (imgsz + 127) // 128 * 128, (imgsz + 127) // 128 * 128, device=device)
    out = {}
    for tag, half in [("fp32", False), ("fp16", True)]:
        if half and device.type != "cuda":
            continue
        mm = m.half() if half else m.float()
        xx = x.half() if half else x.float()
        with torch.no_grad():
            for _ in range(20):
                mm(xx).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                mm(xx).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
        out[f"latency_ms_{tag}"] = (time.perf_counter() - t) / n * 1000
    pred.model.float()
    return out


def bench_yolo(pred: YoloPredictor, sample_path, n=50):
    for _ in range(5):
        pred.m.predict(sample_path, imgsz=pred.imgsz, conf=pred.conf, device=pred.device, verbose=False)
    t = time.perf_counter()
    for _ in range(n):
        pred.m.predict(sample_path, imgsz=pred.imgsz, conf=pred.conf, device=pred.device, verbose=False)
    return {"latency_ms_e2e": (time.perf_counter() - t) / n * 1000}


def yolo_size(pred: YoloPredictor):
    try:
        pred.m.fuse()
        params = sum(p.numel() for p in pred.m.model.parameters())
        # verbose=True is required: with verbose=False ultralytics returns None (no counts)
        info = pred.m.info(verbose=True, imgsz=pred.imgsz)
        return {"params": params, "gflops": float(info[3]) if info and len(info) > 3 else None}
    except Exception as e:                                     # noqa: BLE001
        return {"params": None, "gflops": None, "note": str(e)}


def paired_bootstrap(a, b, n=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    k = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[k]
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def colorize(m):
    return PALETTE[np.minimum(m, len(PALETTE) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=512, help="inference size for AgriSegNet")
    ap.add_argument("--custom", nargs="*", default=[])
    ap.add_argument("--yolo", nargs="*", default=[])
    ap.add_argument("--yolo-imgsz", type=int, default=None, help="defaults to --imgsz")
    ap.add_argument("--yolo-conf", type=float, nargs="+", default=[0.25],
                    help="one or more confidence thresholds for the YOLO instance->semantic conversion. "
                         "Tune on --split valid, then report --split test at that single value.")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--viz", type=int, default=12, help="number of qualitative montages to save")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--out", default="results/compare")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    yimgsz = a.yolo_imgsz or a.imgsz

    preds = [CustomPredictor(c, dev, amp=False, tta=a.tta) for c in a.custom]
    ydev = 0 if dev.type == "cuda" else "cpu"
    preds += [YoloPredictor(w, yimgsz, c, ydev, tag_conf=len(a.yolo_conf) > 1) for w in a.yolo for c in a.yolo_conf]
    if not preds:
        raise SystemExit("give at least one of --custom / --yolo")

    ds = CropSegEval(a.data, a.split, a.imgsz)
    names = ["background"] + list(ds.names)
    cms = {p.name: ConfusionMatrix(len(names)) for p in preds}
    per_img = {p.name: [] for p in preds}
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=eval_collate)

    for i, batch in enumerate(dl):
        gt = batch["gt"].numpy().astype(np.uint8)
        maps = {}
        for p in preds:
            pm = p(batch)
            maps[p.name] = pm
            cms[p.name].update(gt, pm)
            c1 = ConfusionMatrix(len(names)); c1.update(gt, pm)
            per_img[p.name].append(img_miou_crops(c1))
        if i < a.viz:
            tiles = [batch["rgb"], colorize(gt)] + [colorize(m) for m in maps.values()]
            h = 360
            tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h), interpolation=cv2.INTER_NEAREST) for t in tiles]
            cv2.imwrite(str(out / f"viz_{i:03d}.jpg"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))

    rows = []
    for p in preds:
        s = cms[p.name].summary(names)
        row = {"model": p.name, "kind": "custom" if isinstance(p, CustomPredictor) else "yolo", **s}
        if isinstance(p, CustomPredictor):
            row["params"] = count_params(p.model)
            row["gflops"] = 2 * count_flops(p.model, (a.imgsz, a.imgsz))
            if not a.no_bench:
                row.update(bench_custom(p, a.imgsz, dev))
        else:
            row.update(yolo_size(p))
            if not a.no_bench and len(ds):
                row.update(bench_yolo(p, str(ds.images[0])))
        rows.append(row)

    # seed aggregation for the custom model
    cust = [r for r in rows if r["kind"] == "custom"]
    if len(cust) > 1:
        v = np.array([r["mIoU_crops"] for r in cust])
        print(f"\nAgriSegNet over {len(cust)} checkpoints: mIoU(crops) = {v.mean():.4f} +/- {v.std(ddof=1):.4f}")

    boot = {}
    ys = [p for p in preds if isinstance(p, YoloPredictor)]
    cs = [p for p in preds if isinstance(p, CustomPredictor)]
    for c in cs:
        for y in ys:
            boot[f"{c.name} - {y.name}"] = paired_bootstrap(per_img[c.name], per_img[y.name])

    json.dump({"rows": rows, "paired_bootstrap_mIoU_crops(mean,lo95,hi95)": boot, "classes": names,
               "args": vars(a)}, open(out / "results.json", "w"), indent=2)

    print(f"\n{'model':<28}{'params':>10}{'GFLOPs':>9}{'mIoU_crops':>12}{'mIoU_all':>10}{'pixAcc':>8}{'ms fp32/e2e':>13}{'ms fp16':>9}")
    for r in rows:
        ms = r.get("latency_ms_fp32", r.get("latency_ms_e2e"))
        print(f"{r['model']:<28}{(r.get('params') or 0):>10,}{(r.get('gflops') or 0):>9.2f}"
              f"{r['mIoU_crops']:>12.4f}{r['mIoU_all']:>10.4f}{r['pixel_acc']:>8.4f}"
              f"{(ms or 0):>13.1f}{(r.get('latency_ms_fp16') or 0):>9.1f}")
    print("\nper-class IoU:")
    for r in rows:
        print(f"  {r['model']:<26}", {k: (None if v is None else round(v, 3)) for k, v in r["per_class_iou"].items()})
    for k, (m, lo, hi) in boot.items():
        print(f"paired bootstrap  {k}: dmIoU = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (CI excluding 0 => real difference)")


if __name__ == "__main__":
    main()
