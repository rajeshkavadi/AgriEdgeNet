"""Confusion-matrix based segmentation metrics + semantic conversion of YOLO instance masks."""
from __future__ import annotations

import cv2
import numpy as np


class ConfusionMatrix:
    def __init__(self, n):
        self.n = n
        self.m = np.zeros((n, n), np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray):
        k = gt != 255
        idx = self.n * gt[k].astype(np.int64) + pred[k].astype(np.int64)
        self.m += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def summary(self, names):
        tp = np.diag(self.m).astype(np.float64)
        gt_sum, pr_sum = self.m.sum(1), self.m.sum(0)
        union = gt_sum + pr_sum - tp
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        f1 = np.where(gt_sum + pr_sum > 0, 2 * tp / np.maximum(gt_sum + pr_sum, 1), np.nan)
        return {
            "pixel_acc": float(tp.sum() / max(self.m.sum(), 1)),
            "mIoU_all": float(np.nanmean(iou)),                       # incl. background
            "mIoU_crops": float(np.nanmean(iou[1:])),                 # crop classes only (headline number)
            "mF1_crops": float(np.nanmean(f1[1:])),
            "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, iou)},
        }


def instances_to_semantic(polys_px, classes, confs, h, w):
    """YOLO instance masks -> one semantic map. Low-confidence instances are painted first so
    higher-confidence ones win overlaps. polys_px: list of Nx2 pixel polygons (original image
    coordinates). classes are dataset class ids (0-based); output is 0=bg, class+1."""
    out = np.zeros((h, w), np.uint8)
    for i in np.argsort(np.asarray(confs)):
        pts = np.round(np.asarray(polys_px[i])).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(out, [pts], int(classes[i]) + 1)
    return out
