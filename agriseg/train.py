"""Train AgriSegNet on a Roboflow YOLO-segmentation export.

    python train.py --data /path/to/roboflow_export --imgsz 640 --epochs 150 --bs 16 \
                    --width 1.0 --depth 1 --seed 0 --out runs/agri_w1d1_s0

Use the SAME --imgsz you trained YOLO26 with, otherwise the comparison is not like-for-like.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agrisegnet import PARAM_BUDGET, build_model, count_flops, count_params
from dataset import CropSegDataset, class_frequencies, find_split_dir, load_names
from engine import evaluate_model


class ModelEMA:
    def __init__(self, model, decay=0.999, tau=1000):
        self.ema = copy.deepcopy(model).eval()
        self.decay, self.tau, self.updates = decay, tau, 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def dice_loss(logits, target, n_cls, ignore=255, eps=1.0):
    valid = (target != ignore)
    probs = logits.softmax(1) * valid.unsqueeze(1)
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, n_cls).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    present = onehot.sum((0, 2, 3)) > 0
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(base_seed):
    # deterministic per worker (the previous version mixed in wall-clock time, so --seed did not
    # reproduce a run)
    def _init(wid):
        info = torch.utils.data.get_worker_info()
        s = base_seed * 100_003 + 1000 * wid
        info.dataset.rng = random.Random(s)
        np.random.seed(s % (2 ** 32))
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/agrisegnet")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arch", choices=["agrisegnet", "nano"], default="agrisegnet")
    ap.add_argument("--max-params", type=int, default=None, help="default: 50k for nano, 1M for agrisegnet")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--aux-w", type=float, default=0.4)
    ap.add_argument("--dice-w", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-val-images", type=int, default=None)
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (dev.type == "cuda") and not a.no_amp
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = load_names(Path(a.data))
    n_cls = len(names) + 1
    ds = CropSegDataset(a.data, "train", a.imgsz, train=True, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=len(ds) >= a.bs,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0,
                    worker_init_fn=worker_init(a.seed))

    freq = class_frequencies(find_split_dir(Path(a.data), "train") / "labels", len(names))
    cw = 1.0 / np.sqrt(np.maximum(freq, 1e-3))
    cw = np.clip(cw / cw.mean(), 0.25, 4.0)
    print("class pixel share:", dict(zip(["background"] + names, np.round(freq, 3))))
    print("class weights    :", dict(zip(["background"] + names, np.round(cw, 2))))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), ignore_index=255)

    model = build_model(a.arch, n_cls, width=a.width, depth=a.depth, aux=a.aux_w > 0).to(dev)
    model = model.to(memory_format=torch.channels_last)
    n_params = count_params(model)
    gmacs = count_flops(build_model(a.arch, n_cls, a.width, a.depth, aux=False), (a.imgsz, a.imgsz))
    budget = a.max_params or PARAM_BUDGET[a.arch]
    print(f"{a.arch} w={a.width} d={a.depth}: {n_params:,} params (budget {budget:,}), {gmacs:.3f} GMACs @ {a.imgsz}")
    if n_params >= budget:
        raise SystemExit(f"{n_params:,} parameters exceeds the {budget:,} budget")

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps_per_epoch = len(dl)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model)

    cfg = {"arch": a.arch, "num_classes": n_cls, "width": a.width, "depth": a.depth, "imgsz": a.imgsz, "names": names}
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "loss", "val_mIoU_crops", "val_mIoU_all", "val_pixel_acc", "lr", "sec"])
    best, t0 = -1.0, time.time()

    for ep in range(a.epochs):
        model.train()
        run, nb = 0.0, 0
        for x, y in dl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                o = model(x)
                main_o = (o["out"] if isinstance(o, dict) else o).float()   # tensor when --aux-w 0
                loss = ce(main_o, y) + a.dice_w * dice_loss(main_o, y, n_cls)
                if isinstance(o, dict):
                    aux_o = o["aux"].float()
                    loss = loss + a.aux_w * (ce(aux_o, y) + a.dice_w * dice_loss(aux_o, y, n_cls))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run += loss.item(); nb += 1

        miou = miou_all = pacc = float("nan")
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            res, _ = evaluate_model(ema.ema.to(dev), a.data, "valid", a.imgsz, dev, names, amp=amp,
                                    workers=min(2, a.workers), max_images=a.max_val_images)
            miou, miou_all, pacc = res["mIoU_crops"], res["mIoU_all"], res["pixel_acc"]
            if miou > best:
                best = miou
                torch.save({"model": ema.ema.state_dict(), "cfg": cfg, "epoch": ep, "val": res}, out / "best.pt")
        torch.save({"model": ema.ema.state_dict(), "cfg": cfg, "epoch": ep}, out / "last.pt")
        wr.writerow([ep, run / max(nb, 1), miou, miou_all, pacc, opt.param_groups[0]["lr"], time.time() - t0]); log.flush()
        print(f"ep {ep+1:3d}/{a.epochs} loss {run/max(nb,1):.4f} val mIoU(crops) {miou:.4f} best {best:.4f}", flush=True)

    json.dump({"params": n_params, "gmacs": gmacs, "best_val_mIoU_crops": best, "args": vars(a)},
              open(out / "summary.json", "w"), indent=2)
    print("done. best val mIoU(crops) =", best)


if __name__ == "__main__":
    main()
