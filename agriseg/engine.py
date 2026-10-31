"""Shared inference / evaluation helpers (used by train.py and evaluate.py)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agrisegnet import build_model
from dataset import CropSegEval, eval_collate
from metrics import ConfusionMatrix


def build_from_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg.get("arch", "agrisegnet"), cfg["num_classes"], width=cfg["width"],
                        depth=cfg["depth"], aux=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("aux_head")}
    model.load_state_dict(sd)
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_semantic(model, batch, device, amp=False, tta=False) -> np.ndarray:
    """Returns uint8 map at ORIGINAL image resolution (0=bg, c+1)."""
    x = batch["image"].to(device, non_blocking=True)
    nh, nw = batch["valid_hw"]
    h, w = batch["gt"].shape
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = model(x).float()
        if tta:
            logits = logits.softmax(1)
            logits = logits + model(x.flip(3)).float().softmax(1).flip(3)
            logits = logits + model(x.flip(2)).float().softmax(1).flip(2)
    logits = logits[:, :, :nh, :nw]
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def evaluate_model(model, root, split, imgsz, device, names, amp=False, tta=False, workers=2,
                   max_images=None, save_pred_dir=None):
    model.eval()
    ds = CropSegEval(root, split, imgsz)
    n_cls = ds.nc + 1
    cm = ConfusionMatrix(n_cls)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers, collate_fn=eval_collate)
    for i, batch in enumerate(dl):
        if max_images and i >= max_images:
            break
        pred = predict_semantic(model, batch, device, amp, tta)
        gt = batch["gt"].numpy().astype(np.uint8)
        cm.update(gt, pred)
        if save_pred_dir is not None:
            from pathlib import Path
            import cv2
            Path(save_pred_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_pred_dir) / (Path(batch["path"]).stem + ".png")), pred)
    return cm.summary(["background"] + list(names)), cm
