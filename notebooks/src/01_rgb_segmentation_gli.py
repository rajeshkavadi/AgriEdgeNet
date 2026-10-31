# %% [markdown]
# # RGB crop segmentation (YOLO26n-seg) with per-patch GLI and area
#
# Corrected version of `v4_segmentation+ indices.ipynb`.
#
# **What was wrong in the previous version** (details in `docs/REVIEW.md`):
#
# 1. **Leaked credential.** The Roboflow API key was hard-coded in two cells of a notebook shared on
#    Drive. Rotate it in Roboflow now. This notebook reads it from Colab Secrets (`ROBOFLOW_API_KEY`).
# 2. **Wrong weights could run silently.** The notes say the correct model is `v4_best_epoch78.pt`, but
#    `segment_best.pt` was used, and `prepare()` would silently swap in any `*54*.pt` it found. The
#    model is now loaded from one explicit path, and its SHA-256 and class names are logged.
# 3. **Class names were hard-coded** (`['fenugreek','gram','linseed','potato','wheat']`) instead of being
#    read from the model. A mismatch would mislabel every patch. They are now read from
#    `model.names` and checked against `data.yaml`.
# 4. **Masks were resized naively.** `result.masks.data` lives in the letterboxed inference frame.
#    Stretching it to the image size misaligns masks for any non-square image. The notebook now uses
#    `retina_masks=True` (masks at original resolution) and checks the shape.
# 5. **Overlapping instances were double-counted** in area and yield. Each pixel now belongs to one
#    instance only (the highest-confidence one).
# 6. **The GSD block was inconsistent.** Footprint / 640 was applied to masks resized to the image's
#    own size. GSD is now computed from each image's actual width and height. This is valid only if the
#    dataset images are full camera frames (resized, not cropped or tiled). See `IMAGES_ARE_FULL_FRAMES`.
# 7. **Mask GLI mixed soil and canopy.** The mean GLI over a crop polygon depends on canopy cover. The
#    notebook now reports the vegetation fraction and GLI over vegetation pixels separately.
# 8. NaN GLI was silently mapped to `MODERATE`; it is now `UNKNOWN`. `cv2.putText` cannot draw `²`,
#    so labels use `m2`.
# 9. **No accuracy was ever computed.** The notebook only drew predictions. It now runs `model.val()` on the test
#    split (mask mAP) and checks for near-duplicate images across train/valid/test (leakage).
# 10. The "NDVI" part ran on a third-party paddy dataset (IIT Tirupati `Set_1/Set_2`) with the same
#    broken radiometry. It is removed here. Sequoia indices live in notebook 02.
#
# **Read before quoting numbers.** The GLI condition thresholds and the yield factors are not
# validated against any field measurement, so "yield" here is a scenario (reference yield x area x
# factor), not an estimate. It is labelled `indicative` in every output.

# %%
# ---------------------------------------------------------------- setup
import os, sys, csv, json, hashlib, subprocess, shutil, warnings
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'ultralytics', 'roboflow'], check=True)

import numpy as np
import cv2
import yaml
import torch
from ultralytics import YOLO

# %%
# ---------------------------------------------------------------- configuration
ROOT = Path('/content') if Path('/content').exists() else Path.cwd()

# Weights: upload ONE file and point to it. The notes say v4_best_epoch78.pt is the correct model.
MODEL_PATH = Path(os.environ.get('SEG_MODEL', str(ROOT / 'v4_best_epoch78.pt')))

# Roboflow export (YOLO segmentation format)
RF_WORKSPACE, RF_PROJECT, RF_VERSION, RF_FORMAT = 'kavadis-workspace', 'crop_classification_orin', 4, 'yolo26'
DATASET_DIR = Path(os.environ.get('SEG_DATASET', str(ROOT / f'{RF_PROJECT}-{RF_VERSION}')))
EVAL_SPLIT = 'test'

RESULTS = Path(os.environ.get('SEG_OUT', str(ROOT / 'Results' / 'Segmentation')))
IMGSZ, CONF = 640, 0.25

# Ground scale. These are the Raspberry Pi Camera v2 (IMX219) values from the old notebook.
# Replace them with the camera and altitude that actually took the images.
CAMERA = {'sensor_w_mm': 3.68, 'sensor_h_mm': 2.76, 'focal_mm': 3.04, 'height_agl_m': 20.0}
IMAGES_ARE_FULL_FRAMES = True   # False if Roboflow cropped or tiled the frames: area is then meaningless

# GLI vegetation mask and condition classes (illustrative, NOT validated)
GLI_VEG_THRESHOLD = 0.0
GLI_THRESHOLDS = {'wheat': (0.05, 0.10, 0.15), 'potato': (0.04, 0.08, 0.13), 'gram': (0.03, 0.07, 0.12),
                  'fenugreek': (0.05, 0.09, 0.14), 'linseed': (0.05, 0.09, 0.14)}
REFERENCE_YIELD_KG_HA = {'wheat': 3547, 'potato': 24629, 'gram': 1199, 'fenugreek': 1582, 'linseed': 633}
YIELD_FACTORS = {'STRESSED / POOR': 0.50, 'MODERATE': 0.75, 'GOOD': 1.00, 'VERY GOOD': 1.10, 'UNKNOWN': np.nan}

PALETTE = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 0, 0), (0, 0, 255)]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# %%
# ---------------------------------------------------------------- data and model
def download_dataset():
    """Download the Roboflow export once. The key comes from Colab Secrets or the environment, never the notebook."""
    if (DATASET_DIR / 'data.yaml').is_file():
        return DATASET_DIR
    key = os.environ.get('ROBOFLOW_API_KEY')
    if not key and IN_COLAB:
        from google.colab import userdata
        key = userdata.get('ROBOFLOW_API_KEY')
    if not key:
        raise RuntimeError('Set ROBOFLOW_API_KEY in Colab Secrets (key icon, left sidebar).')
    from roboflow import Roboflow
    ds = Roboflow(api_key=key).workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION).download(
        RF_FORMAT, location=str(DATASET_DIR))
    return Path(ds.location)


def dataset_yaml_abs(root):
    """Roboflow's data.yaml uses '../train/images' paths; write a copy with paths Ultralytics resolves reliably."""
    cfg = yaml.safe_load(open(root / 'data.yaml'))
    names = cfg['names']
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    split_dir = lambda s: next((f'{a}/images' for a in ([s] if s != 'valid' else ['valid', 'val'])
                                if (root / a / 'images').is_dir()), None)
    out = {'path': str(root.resolve()), 'names': {i: n for i, n in enumerate(names)}}
    for k, s in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        if split_dir(s):
            out[k] = split_dir(s)
    p = root / 'data_abs.yaml'
    yaml.safe_dump(out, open(p, 'w'), sort_keys=False)
    return p, names


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(n), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(path, expected_names=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found. Upload the weights and set MODEL_PATH; no fallback is attempted.')
    model = YOLO(str(path))
    if model.task != 'segment':
        raise ValueError(f'{path.name}: task {model.task!r}, expected segment')
    names = [model.names[i] for i in sorted(model.names)]
    print(f'model  : {path.name}  sha256={sha256(path)[:16]}  classes={names}')
    if expected_names is not None and list(expected_names) != names:
        raise ValueError(f'class order mismatch: model {names} vs data.yaml {list(expected_names)}')
    return model, names


# %%
# ---------------------------------------------------------------- split leakage check
def dhash(path, size=8):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def leakage_report(root, max_hamming=6):
    """Near-duplicate images across splits (consecutive UAV frames, Roboflow augment copies).
    Any hit means the test score is optimistic."""
    splits = {s: sorted(p for p in (root / s / 'images').iterdir() if p.suffix.lower() in IMG_EXT)
              for s in ('train', 'valid', 'test') if (root / s / 'images').is_dir()}
    hashes = {s: np.stack([dhash(p) for p in ps]) for s, ps in splits.items() if ps}
    rows = []
    for a, b in (('train', 'test'), ('train', 'valid'), ('valid', 'test')):
        if a in hashes and b in hashes:
            d = np.unpackbits(hashes[b][:, None, :] ^ hashes[a][None, :, :], axis=2).sum(2)
            nearest = d.min(1)
            rows.append({'pair': f'{b} vs {a}', f'{b}_images': len(nearest),
                         f'near_duplicates(<={max_hamming})': int((nearest <= max_hamming).sum()),
                         'median_nearest_hamming': float(np.median(nearest))})
    for r in rows:
        print(r)
    return rows


# %%
# ---------------------------------------------------------------- ground scale
def area_per_pixel_m2(img_w, img_h):
    if not IMAGES_ARE_FULL_FRAMES:
        return float('nan')
    fw = CAMERA['height_agl_m'] * CAMERA['sensor_w_mm'] / CAMERA['focal_mm']   # footprint width, m
    fh = CAMERA['height_agl_m'] * CAMERA['sensor_h_mm'] / CAMERA['focal_mm']
    return (fw / img_w) * (fh / img_h)


# %%
# ---------------------------------------------------------------- per-image analysis
def gli_map(bgr):
    b, g, r = [bgr[..., i].astype(np.float32) for i in range(3)]
    den = 2 * g + r + b
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (2 * g - r - b) / den
    out[den <= 0] = np.nan
    return out


def condition(cls, gli):
    if not np.isfinite(gli) or cls not in GLI_THRESHOLDS:
        return 'UNKNOWN'
    t1, t2, t3 = GLI_THRESHOLDS[cls]
    return 'STRESSED / POOR' if gli < t1 else 'MODERATE' if gli < t2 else 'GOOD' if gli < t3 else 'VERY GOOD'


def instance_ownership(result, shape):
    """-> (owner map: -1 or instance index, per-instance boolean masks). Highest confidence wins overlaps."""
    h, w = shape
    owner = np.full((h, w), -1, np.int32)
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return owner, []
    m = result.masks.data.cpu().numpy() > 0.5
    if m.shape[1:] != (h, w):           # should not happen with retina_masks=True
        m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for x in m])
        warnings.warn('mask resolution differed from image; resized (check retina_masks)')
    conf = result.boxes.conf.cpu().numpy()
    for i in np.argsort(conf):           # low confidence first, overwritten by higher
        owner[m[i]] = i
    return owner, m


def draw_label(img, text, x, y, placed):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    H, W = img.shape[:2]
    for dx, dy in ((-tw // 2, -10), (-tw // 2, tht + 20), (12, 0), (-tw - 12, 0)):
        x0 = int(np.clip(x + dx, 4, W - tw - 4)); y0 = int(np.clip(y + dy, tht + 4, H - base - 4))
        box = (x0 - 4, y0 - tht - 4, x0 + tw + 4, y0 + base + 4)
        if not any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]) for b in placed):
            break
    cv2.rectangle(img, box[:2], box[2:], (0, 0, 0), -1)
    cv2.putText(img, text, (x0, y0), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    placed.append(box)


def analyse_image(path, result, names):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    owner, _ = instance_ownership(result, (h, w))
    gli = gli_map(img)
    a_px = area_per_pixel_m2(w, h)
    vis, placed, rows = img.copy(), [], []
    if result.boxes is None or len(result.boxes) == 0:
        return vis, rows
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for i in range(len(cls_ids)):
        own = owner == i
        npx = int(own.sum())
        if npx == 0:
            continue                      # fully covered by a higher-confidence instance
        name = names[cls_ids[i]]
        g = gli[own]; g = g[np.isfinite(g)]
        veg = g > GLI_VEG_THRESHOLD
        gli_all = float(g.mean()) if g.size else np.nan
        gli_veg = float(g[veg].mean()) if veg.any() else np.nan
        cond = condition(name, gli_all)
        area = npx * a_px
        rate = REFERENCE_YIELD_KG_HA.get(name, np.nan) / 1e4 * YIELD_FACTORS[cond]
        rows.append({'image': path.name, 'patch': i + 1, 'class': name, 'confidence': float(confs[i]),
                     'pixels': npx, 'area_m2': area, 'veg_fraction': float(veg.mean()) if g.size else np.nan,
                     'gli_mean': gli_all, 'gli_median': float(np.median(g)) if g.size else np.nan,
                     'gli_std': float(g.std()) if g.size else np.nan, 'gli_mean_veg_only': gli_veg,
                     'condition_indicative': cond, 'indicative_yield_kg': area * rate})
        col = np.array(PALETTE[cls_ids[i] % len(PALETTE)], np.float32)
        vis[own] = (vis[own].astype(np.float32) * 0.55 + col * 0.45).astype(np.uint8)
        ys, xs = np.nonzero(own)
        draw_label(vis, f'#{i + 1} {name} {confs[i]:.2f} A:{area:.1f}m2 GLI:{gli_all:.2f}',
                   int(xs.mean()), int(ys.mean()), placed)
    return vis, rows


# %%
# ---------------------------------------------------------------- run
def run_segmentation(model, names, image_dir, out_dir=RESULTS):
    out_dir = Path(out_dir); (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise FileNotFoundError(f'no images in {image_dir}')
    dev = 0 if torch.cuda.is_available() else 'cpu'
    rows = []
    for path in images:
        r = model.predict(str(path), imgsz=IMGSZ, conf=CONF, device=dev, retina_masks=True, verbose=False)[0]
        vis, rr = analyse_image(path, r, names)
        cv2.imwrite(str(out_dir / 'overlays' / path.name), vis)
        rows += rr
    keys = list(rows[0].keys()) if rows else ['image']
    with open(out_dir / 'segmentation_patches.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)

    total_area = sum(r['area_m2'] for r in rows)
    by_class = {}
    for r in rows:
        d = by_class.setdefault(r['class'], {'patches': 0, 'area_m2': 0.0, 'gli': []})
        d['patches'] += 1; d['area_m2'] += r['area_m2']; d['gli'].append(r['gli_mean'])
    with open(out_dir / 'SEGMENTATION_REPORT.txt', 'w') as f:
        f.write(f'YOLO-seg crop segmentation report\n{"=" * 70}\n')
        f.write(f'model {MODEL_PATH.name} | images {len(images)} | patches {len(rows)} | conf {CONF} | imgsz {IMGSZ}\n')
        f.write(f'camera {CAMERA} | full frames: {IMAGES_ARE_FULL_FRAMES}\n')
        f.write('Yield and condition are INDICATIVE (unvalidated thresholds/factors).\n\n')
        for c, d in sorted(by_class.items()):
            f.write(f'{c:<10s} patches {d["patches"]:3d}  area {d["area_m2"]:10.2f} m2  '
                    f'mean GLI {np.nanmean(d["gli"]):.3f}\n')
        f.write(f'\nTotal area {total_area:.2f} m2\n')
    print(f'{len(images)} images, {len(rows)} patches, total area {total_area:.1f} m2 -> {out_dir}')
    return rows


def evaluate(model, data_yaml, split=EVAL_SPLIT):
    """Mask/box mAP on a held-out split. Report this, not the training-time valid score."""
    m = model.val(data=str(data_yaml), split=split, imgsz=IMGSZ, batch=8, plots=False, verbose=False,
                  device=0 if torch.cuda.is_available() else 'cpu')
    res = {'mask_mAP50': float(m.seg.map50), 'mask_mAP50_95': float(m.seg.map),
           'box_mAP50_95': float(m.box.map),
           'per_class_mask_mAP50_95': {model.names[int(c)]: float(v) for c, v in zip(m.seg.ap_class_index, m.seg.maps[m.seg.ap_class_index])}}
    print(json.dumps(res, indent=2))
    return res


# %%
if os.environ.get('SEG_SKIP_RUN') != '1':
    ds_root = download_dataset()
    data_yaml, ds_names = dataset_yaml_abs(ds_root)
    leakage_report(ds_root)
    seg_model, seg_names = load_model(MODEL_PATH, ds_names)
    seg_metrics = evaluate(seg_model, data_yaml)
    split_dir = ds_root / EVAL_SPLIT / 'images'
    seg_rows = run_segmentation(seg_model, seg_names, split_dir)
