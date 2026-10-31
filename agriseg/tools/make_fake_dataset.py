"""Synthetic Roboflow-style YOLO-segmentation dataset for smoke-testing the pipeline only.
Crops are separable by texture/colour so a working pipeline should learn them quickly."""
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1]); n_train, n_val, n_test = 24, 6, 8
names = ["wheat", "potato", "methi"]
rng = np.random.default_rng(0)
(root).mkdir(parents=True, exist_ok=True)
(root / "data.yaml").write_text(
    "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
    f"nc: {len(names)}\nnames: {names}\n")


def texture(cls, h, w):
    base = np.array([[90, 150, 60], [60, 110, 50], [40, 170, 90]][cls], np.float32)
    img = np.ones((h, w, 3), np.float32) * base
    yy, xx = np.mgrid[0:h, 0:w]
    if cls == 0:   img *= (1 + 0.25 * np.sin(xx / 2.5))[..., None]            # fine rows
    elif cls == 1: img *= (1 + 0.25 * np.sin(yy / 9.0) * np.sin(xx / 9.0))[..., None]  # blobs
    else:          img *= (1 + 0.3 * rng.standard_normal((h, w)))[..., None]   # noisy
    return img


for split, n in [("train", n_train), ("valid", n_val), ("test", n_test)]:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        h = w = 640
        img = np.full((h, w, 3), (120, 100, 80), np.float32)           # soil
        lines = []
        for _ in range(rng.integers(2, 5)):
            cls = int(rng.integers(0, 3))
            cx, cy = rng.uniform(0.2, 0.8, 2)
            r = rng.uniform(0.12, 0.25)
            ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
            pts = np.stack([cx + r * np.cos(ang) * rng.uniform(0.7, 1.2, 7),
                            cy + r * np.sin(ang) * rng.uniform(0.7, 1.2, 7)], 1).clip(0, 1)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(pts * [w, h]).astype(np.int32)], 1)
            img = np.where(m[..., None] > 0, texture(cls, h, w), img)
            lines.append(f"{cls} " + " ".join(f"{v:.5f}" for v in pts.reshape(-1)))
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / split / "images" / f"{split}_{i:03d}.jpg"), img)
        (root / split / "labels" / f"{split}_{i:03d}.txt").write_text("\n".join(lines))
print("fake dataset at", root)
