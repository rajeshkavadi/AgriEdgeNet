"""Instance outputs and instance metrics.

* group_instances(): turns AgriSegNet-Nano's dense outputs (class probabilities + field-boundary probability) into
  separate fields, each with class, confidence, mask and box: bottom-up (proposal-free) instance segmentation.
  Without an instance head it falls back to connected regions of each class.
* yolo_instances(): the same structure from an Ultralytics result.
* MaskAP: COCO-style mask AP (AP50, AP75, AP50-95), identical code for every model.
"""
from __future__ import annotations

import cv2
import numpy as np

BOUNDARY_T = 0.5            # pixels with boundary probability >= this separate fields
MIN_PART = 0.1              # a region is split only into parts each >= this share of the region ...
MIN_INST_FRAC = 0.002       # ... and >= this share of the image (also the minimum field size)


def _box(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def group_instances(probs, boundary=None, min_frac=MIN_INST_FRAC):
    """probs: (C, H, W) class probabilities incl. background (index 0); boundary: (H, W) in [0, 1] or None.
    Returns a list of {'class': crop id (0-based), 'score', 'mask' (bool H x W), 'box'}.

    1. Connected regions of each predicted crop are candidate fields.
    2. With a boundary map, a region is split when the predicted field boundaries cut it into >= 2 parts that are
       each large enough to be a field (touching fields of the same crop); boundary pixels are then given back to
       the nearest part by watershed on the boundary map. Small cuts (noise inside a field) do not split it.
    3. Score = mean probability of the field's class inside the field."""
    C, H, W = probs.shape
    label = probs.argmax(0)
    min_px = max(1, int(min_frac * H * W))
    out = []
    for c in range(1, C):
        n, comp = cv2.connectedComponents((label == c).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            region = comp == j
            area = int(region.sum())
            if area < min_px:
                continue
            parts = [region]
            if boundary is not None:
                ns, sc = cv2.connectedComponents((region & (boundary < BOUNDARY_T)).astype(np.uint8), connectivity=4)
                big = [s for s in range(1, ns) if (sc == s).sum() >= max(min_px, MIN_PART * area)]
                if len(big) >= 2:
                    from skimage.segmentation import watershed
                    markers = np.zeros((H, W), np.int32)
                    for q, s in enumerate(big, 1):
                        markers[sc == s] = q
                    lab = watershed(boundary, markers, mask=region)
                    parts = [lab == q for q in range(1, len(big) + 1)]
            for m in parts:
                if m.sum() >= min_px:
                    out.append({'class': c - 1, 'score': float(probs[c][m].mean()), 'mask': m, 'box': _box(m)})
    return out


def split_outputs(logits, n_cls):
    """logits: (C[+1], H, W) numpy -> probs (C, H, W), boundary probability (H, W) or None."""
    z = logits[:n_cls] - logits[:n_cls].max(0, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(0, keepdims=True)
    if logits.shape[0] > n_cls:
        return probs, 1.0 / (1.0 + np.exp(-logits[n_cls]))
    return probs, None


def instances_to_label_map(instances, h, w):
    """Paint instances (lowest score first) into a semantic map: 0 = background, class + 1."""
    out = np.zeros((h, w), np.uint8)
    for inst in sorted(instances, key=lambda d: d['score']):
        out[inst['mask']] = inst['class'] + 1
    return out


def yolo_instances(result, h, w):
    """Ultralytics segmentation result (predicted with retina_masks=True) -> instance list."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != (h, w):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    return [{'class': int(c), 'score': float(s), 'mask': m, 'box': _box(m)}
            for m, c, s in zip(masks, cls, conf) if m.any()]


def gt_instances(gt_inst, gt_cls):
    """Instance-id map + {id: class} from the dataset -> list of {'class', 'mask'} (ids hidden by overlaps dropped)."""
    return [{'class': gt_cls[i], 'mask': gt_inst == i} for i in np.unique(gt_inst) if i != 0 and i in gt_cls]


def _iou_matrix(pred, gt):
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    P = np.stack([p['mask'].ravel() for p in pred]).astype(np.float32)
    G = np.stack([g['mask'].ravel() for g in gt]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


class MaskAP:
    """COCO-style mask AP: greedy matching by score per class, 101-point interpolated precision,
    IoU thresholds 0.50:0.05:0.95. Same code for every model, so numbers are directly comparable."""

    def __init__(self, n_classes, thresholds=np.round(np.arange(0.5, 0.96, 0.05), 2)):
        self.nc, self.t = n_classes, thresholds
        self.scores = [[] for _ in range(n_classes)]
        self.tp = [[] for _ in range(n_classes)]
        self.n_gt = np.zeros(n_classes, int)

    def add(self, preds, gts):
        for c in range(self.nc):
            P = sorted([p for p in preds if p['class'] == c], key=lambda d: -d['score'])
            G = [g for g in gts if g['class'] == c]
            self.n_gt[c] += len(G)
            iou = _iou_matrix(P, G)
            for i, p in enumerate(P):
                self.scores[c].append(p['score'])
            flags = np.zeros((len(P), len(self.t)), bool)
            for ti, t in enumerate(self.t):
                taken = np.zeros(len(G), bool)
                for i in range(len(P)):
                    if not len(G):
                        break
                    cand = np.where(~taken & (iou[i] >= t))[0]
                    if len(cand):
                        j = cand[np.argmax(iou[i, cand])]
                        taken[j] = True
                        flags[i, ti] = True
            self.tp[c].extend(flags)

    def summary(self, names=None):
        ap = np.full((self.nc, len(self.t)), np.nan)
        rec_pts = np.linspace(0, 1, 101)
        for c in range(self.nc):
            if self.n_gt[c] == 0:
                continue
            if not self.scores[c]:
                ap[c] = 0.0
                continue
            order = np.argsort(-np.asarray(self.scores[c]), kind='stable')
            tp = np.asarray(self.tp[c])[order]
            for ti in range(len(self.t)):
                ctp = np.cumsum(tp[:, ti]); cfp = np.cumsum(~tp[:, ti])
                rec = ctp / self.n_gt[c]
                prec = ctp / np.maximum(ctp + cfp, 1)
                prec = np.maximum.accumulate(prec[::-1])[::-1]            # precision envelope
                idx = np.searchsorted(rec, rec_pts, side='left')
                ap[c, ti] = np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])
        names = names or [str(c) for c in range(self.nc)]
        i50, i75 = int(np.argmin(abs(self.t - 0.5))), int(np.argmin(abs(self.t - 0.75)))
        return {'mask_mAP50': float(np.nanmean(ap[:, i50])), 'mask_mAP75': float(np.nanmean(ap[:, i75])),
                'mask_mAP50_95': float(np.nanmean(ap)),
                'per_class_AP50_95': {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, np.nanmean(ap, 1))
                                      } if not np.all(np.isnan(ap)) else {},
                'n_gt': {n: int(k) for n, k in zip(names, self.n_gt)}}
