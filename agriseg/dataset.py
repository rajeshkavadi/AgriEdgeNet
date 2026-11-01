"""Roboflow / Ultralytics YOLO-segmentation export  ->  semantic segmentation dataset.

Expected layout (what Roboflow's "YOLOv8/YOLO26 segmentation" export gives you):

    <root>/data.yaml            (names: [...])
    <root>/train/images/*.jpg   <root>/train/labels/*.txt
    <root>/valid/images/*.jpg   <root>/valid/labels/*.txt     ('val' also accepted)
    <root>/test/images/*.jpg    <root>/test/labels/*.txt

Each label line:  class_id x1 y1 x2 y2 ...   (normalised polygon)
Semantic mask   :  0 = background, 1..nc = class_id + 1, 255 = ignore (padding only)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
IGNORE = 255
PAD_MULT = 32    # network stride; the onnx exporter rounds agrisegnet inputs to 128 (context-pool bins 1,2,4)


def load_names(root: Path) -> List[str]:
    cfg = yaml.safe_load(open(Path(root) / "data.yaml"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def find_split_dir(root: Path, split: str) -> Path:
    aliases = {"val": ["valid", "val"], "valid": ["valid", "val"]}.get(split, [split])
    for a in aliases:
        if (Path(root) / a / "images").is_dir():
            return Path(root) / a
    raise FileNotFoundError(f"No '{split}' split under {root} (looked for {aliases})")


def parse_label_file(path: Path) -> List[Tuple[int, np.ndarray]]:
    """-> list of (class_id, polygon[N,2] normalised). Bounding boxes (5 values) become rectangles."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cls = int(float(v[0]))
        vals = np.array(v[1:], dtype=np.float64)
        if len(vals) == 4:                       # cx cy w h  -> rectangle
            cx, cy, w, h = vals
            poly = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                             [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        else:
            if len(vals) % 2:
                vals = vals[:-1]
            poly = vals.reshape(-1, 2)
        if len(poly) >= 3:
            out.append((cls, poly))
    return out


def rasterize(polys: List[Tuple[int, np.ndarray]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for cls, poly in polys:                       # later polygons overwrite earlier ones
        pts = np.round(poly * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(mask, [pts], int(cls) + 1)
    return mask


def rasterize_instances(polys: List[Tuple[int, np.ndarray]], h: int, w: int):
    """-> (semantic map uint8: 0=bg, c+1; instance map int32: 0=bg, 1..N; class of each instance id).
    Roboflow polygons are per-field annotations, so each polygon is one instance (later polygons win overlaps)."""
    sem = np.zeros((h, w), np.uint8)
    inst = np.zeros((h, w), np.int32)
    classes = {}
    for k, (cls, poly) in enumerate(polys, start=1):
        pts = np.round(poly * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(sem, [pts], int(cls) + 1)
        cv2.fillPoly(inst, [pts], k)
        classes[k] = int(cls)
    return sem, inst, classes


BOUNDARY_WIDTH = 3   # px at training resolution


def instance_targets(inst: np.ndarray, sem: np.ndarray) -> np.ndarray:
    """Boundary target (1, H, W) for the instance head, from an instance-id map: 1 on field edges
    (field/field and field/background), width ~BOUNDARY_WIDTH px, 0 elsewhere, -1 where `sem` is ignore (255)."""
    f = inst.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(f, k) != cv2.erode(f, k)).astype(np.uint8)
    if BOUNDARY_WIDTH > 1:
        edge = cv2.dilate(edge, np.ones((BOUNDARY_WIDTH, BOUNDARY_WIDTH), np.uint8))
    t = edge.astype(np.float32)[None]
    t[:, sem == IGNORE] = -1.0
    return t


def class_frequencies(label_dir: Path, nc: int) -> np.ndarray:
    """Approximate pixel share of [background, class0..] from polygon areas (no image reads)."""
    areas = np.zeros(nc + 1)
    for f in Path(label_dir).glob("*.txt"):
        tot = 0.0
        for cls, p in parse_label_file(f):
            x, y = p[:, 0], p[:, 1]
            a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            areas[cls + 1] += a
            tot += a
        areas[0] += max(0.0, 1.0 - tot)
    return areas / max(areas.sum(), 1e-9)


# ----------------------------------------------------------------------------- augmentation
def _color_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                    # contrast / brightness
    if rng.random() < 0.5:                                                      # gamma
        g = rng.uniform(0.7, 1.4)
        img = 255 * (np.clip(img, 0, 255) / 255) ** g
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180                      # hue
    hsv[..., 1] *= rng.uniform(0.7, 1.3)                                        # saturation
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.2:                                                      # motion-blur proxy
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        kern /= max(kern.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kern)
    if rng.random() < 0.2:                                                      # sensor noise
        img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img


def _affine(img, mask, out: int, rng: random.Random, scale_rng=(0.6, 1.6), inst=None):
    """Random scale / rotation (any heading) / flip / translation in one warp."""
    h, w = mask.shape
    base = out / max(h, w)
    s = base * rng.uniform(*scale_rng)
    ang = rng.choice([0, 90, 180, 270]) + rng.uniform(-20, 20) if rng.random() < 0.8 else 0.0
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[0, 2] += out / 2 - cx
    M[1, 2] += out / 2 - cy
    if rng.random() < 0.5:                       # horizontal flip of the output
        M = np.array([[-1, 0, out - 1], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
    img = cv2.warpAffine(img, M[:2], (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M[:2], (out, out), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
    if inst is not None:
        inst = cv2.warpAffine(inst.astype(np.float32), M[:2], (out, out), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
        return img, mask, inst
    return img, mask


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ----------------------------------------------------------------------------- datasets
class CropSegDataset(Dataset):
    """Training dataset: random affine + colour aug, fixed square output."""

    def __init__(self, root, split="train", imgsz=512, train=True, seed=0):
        self.root, self.imgsz, self.train = Path(root), imgsz, train
        d = find_split_dir(self.root, split)
        self.images = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in IMG_EXT)
        self.label_dir = d / "labels"
        self.names = load_names(self.root)
        self.nc = len(self.names)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def load(self, i):
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask, inst, _ = rasterize_instances(parse_label_file(self.label_dir / (p.stem + ".txt")), h, w)
        return img, mask, inst

    def __getitem__(self, i):
        """-> image tensor, semantic target (H, W), field-boundary target (1, H, W)."""
        img, mask, inst = self.load(i)
        rng = self.rng if self.train else random.Random(i)
        img, mask, inst = _affine(img, mask, self.imgsz, rng, inst=inst)
        img = _color_jitter(img, rng)
        return (to_tensor(img), torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(instance_targets(inst, mask)))


def preprocess(img_rgb: np.ndarray, imgsz: int, pad_mult: int = PAD_MULT):
    """Aspect-preserving resize (long side = imgsz), reflect-pad bottom/right to a multiple of pad_mult.
    Returns (CHW float tensor, (valid_h, valid_w)). Used for evaluation AND deployment, so both see the same input."""
    h, w = img_rgb.shape[:2]
    s = imgsz / max(h, w)
    nh, nw = max(32, round(h * s)), max(32, round(w * s))
    small = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    small = cv2.copyMakeBorder(small, 0, (-nh) % pad_mult, 0, (-nw) % pad_mult, cv2.BORDER_REFLECT_101)
    return to_tensor(small), (nh, nw)


class CropSegEval(Dataset):
    """Evaluation dataset: aspect-preserving resize (long side = imgsz), pad to a multiple of 128.
    Returns the ORIGINAL-resolution mask so every model is scored at the same resolution."""

    def __init__(self, root, split="test", imgsz=512):
        self.root, self.imgsz = Path(root), imgsz
        d = find_split_dir(self.root, split)
        self.images = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in IMG_EXT)
        self.label_dir = d / "labels"
        self.names = load_names(self.root)
        self.nc = len(self.names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt, gt_inst, gt_cls = rasterize_instances(parse_label_file(self.label_dir / (p.stem + ".txt")), h, w)
        x, (nh, nw) = preprocess(img, self.imgsz)
        return {"image": x, "gt": torch.from_numpy(gt.astype(np.int64)), "gt_inst": gt_inst, "gt_cls": gt_cls,
                "valid_hw": (nh, nw), "path": str(p), "rgb": img}


def eval_collate(batch):          # batch size is 1 for eval; keep python objects intact
    assert len(batch) == 1
    b = batch[0]
    b["image"] = b["image"].unsqueeze(0)
    return b
