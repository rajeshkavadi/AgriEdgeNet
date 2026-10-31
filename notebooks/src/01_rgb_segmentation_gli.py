# %% [markdown]
# # RGB crop segmentation with a device-agnostic lightweight CNN (AgriSegNet-Nano) + per-patch GLI and area
#
# Replaces YOLO26n-seg with **AgriSegNet-Nano**: 34.7k parameters, 0.25 GMACs at 640 px, built only from
# operators that every edge runtime supports. After BN folding the exported graph contains Conv
# (dense + depthwise, no dilation), Add, Mul, Clip (ReLU6), GlobalAveragePool, Sigmoid and Resize with fixed
# integer factors. It is static, with no Shape/Slice ops, so it runs on TensorRT (Jetson Nano/Orin), ONNX Runtime
# (Raspberry Pi), OpenVINO, NPUs and, via TFLite-Micro, microcontrollers.
#
# Flow: Roboflow export (train/valid/test, YOLO polygon labels) -> semantic masks -> train AgriSegNet-Nano ->
# train two YOLO26n-seg baselines on the same split (COCO-pretrained and from scratch) -> pick YOLO's confidence
# threshold on **valid** -> compare all models on **test** (mIoU, per-class IoU, paired bootstrap CI, params, GFLOPs)
# -> ONNX FP32 + INT8 with accuracy check -> per-patch area and GLI on the test images.
#
# **Why two YOLO baselines.** `yolo26n-seg` (COCO-pretrained) is what a practitioner would deploy; it is the
# practical bar. `yolo26n-seg` from scratch has the same starting point as Nano (no pretraining) and isolates
# architecture and size. Nano has no pretrained weights, so the pretrained comparison is deliberately unequal.
#
# **On devices the binding limit is activation memory, not parameters.** INT8 peak activations are about 2.7 MiB at
# 640 px but 0.44 MiB at 256 px. For microcontrollers, run at 256 to 320 px; the evaluation below reports
# the accuracy cost of doing so.
#
# Fixes carried over from the review (`docs/REVIEW.md`): the API key comes from Colab Secrets (**rotate the old
# leaked key**); GSD is computed per image; GLI is reported over vegetation pixels; yield is labelled indicative;
# leakage across splits is checked.

# %%
# ---------------------------------------------------------------- setup
import os, sys, csv, json, subprocess, shutil
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'roboflow', 'onnx', 'onnxruntime', 'onnxscript'],
                   check=True)

import numpy as np
import cv2
import yaml
import torch

ROOT = Path('/content') if Path('/content').exists() else Path.cwd()
if IN_COLAB:                       # results (incl. trained checkpoints) go to Drive so a disconnect loses nothing
    from google.colab import drive
    drive.mount('/content/drive')
RESULTS_ROOT = Path('/content/drive/MyDrive/AgriEdgeNet_Results') if IN_COLAB else ROOT / 'Results'
AGRISEG_DIR = Path(os.environ.get('AGRISEG_DIR', str(ROOT / 'agriseg')))

# %% [agriseg]

# %%
# ---------------------------------------------------------------- configuration
# Roboflow export (YOLO segmentation format: train / valid / test with polygon labels).
# Key: preferably a Colab Secret named ROBOFLOW_API_KEY (key icon, left sidebar). You may paste it below instead,
# but then do not share this notebook: anyone with the file can use your Roboflow account.
ROBOFLOW_API_KEY = ''
RF_WORKSPACE, RF_PROJECT, RF_VERSION, RF_FORMAT = 'kavadis-workspace', 'crop_classification_orin', 4, 'yolo26'
DATASET_DIR = Path(os.environ.get('SEG_DATASET', str(ROOT / f'{RF_PROJECT}-{RF_VERSION}')))
RESULTS = Path(os.environ.get('SEG_OUT', str(RESULTS_ROOT / 'Segmentation')))

# Model and training
ARCH, WIDTH = 'nano', 1.0          # nano: 34.7k params (width 0.5 -> 11.4k). 'agrisegnet' width 0.25 -> 43.9k
TRAIN_IMGSZ = int(os.environ.get('SEG_IMGSZ', 640))
EPOCHS = int(os.environ.get('SEG_EPOCHS', 150))
BATCH = 16
SEEDS = [0]                        # use [0, 1, 2] for any reported number (mean +/- std)
EVAL_SIZES = [TRAIN_IMGSZ, 320]    # 320 = microcontroller-class input

# YOLO26 baselines: trained here on the same split, same imgsz and epochs
YOLO_BASELINES = {'yolo26n_coco': 'yolo26n-seg.pt',        # COCO-pretrained (Ultralytics release weights)
                  'yolo26n_scratch': 'yolo26n-seg.yaml'}   # random init, same architecture
YOLO_CONF_SWEEP = [0.05, 0.1, 0.25, 0.4]                    # chosen on valid, then fixed for test
EXISTING_YOLO = ROOT / 'v4_best_epoch78.pt'                 # your earlier weights: evaluated too if uploaded

# Ground scale (Raspberry Pi Camera v2 / IMX219 values from the old notebook; replace with the real camera and AGL)
CAMERA = {'sensor_w_mm': 3.68, 'sensor_h_mm': 2.76, 'focal_mm': 3.04, 'height_agl_m': 20.0}
IMAGES_ARE_FULL_FRAMES = True      # False if Roboflow cropped/tiled frames: area is then meaningless
MIN_PATCH_FRACTION = 0.002         # connected regions smaller than this share of the image are ignored

# GLI (illustrative, NOT validated)
GLI_VEG_THRESHOLD = 0.0
GLI_THRESHOLDS = {'wheat': (0.05, 0.10, 0.15), 'potato': (0.04, 0.08, 0.13), 'gram': (0.03, 0.07, 0.12),
                  'fenugreek': (0.05, 0.09, 0.14), 'linseed': (0.05, 0.09, 0.14)}
REFERENCE_YIELD_KG_HA = {'wheat': 3547, 'potato': 24629, 'gram': 1199, 'fenugreek': 1582, 'linseed': 633}
YIELD_FACTORS = {'STRESSED / POOR': 0.50, 'MODERATE': 0.75, 'GOOD': 1.00, 'VERY GOOD': 1.10, 'UNKNOWN': np.nan}

PALETTE = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 0, 0), (0, 0, 255)]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

sys.path.insert(0, str(AGRISEG_DIR))
from agrisegnet import build_model, count_params, count_flops, peak_activation_bytes   # noqa: E402
from engine import build_from_ckpt, predict_image                                      # noqa: E402
from dataset import load_names, find_split_dir                                          # noqa: E402


def run(cmd):
    """Run an agriseg script and stream its output."""
    print('$', ' '.join(map(str, cmd)))
    subprocess.run([sys.executable, *map(str, cmd)], cwd=AGRISEG_DIR, check=True)


# %%
# ---------------------------------------------------------------- dataset
def download_dataset():
    if (DATASET_DIR / 'data.yaml').is_file():
        return DATASET_DIR
    key = ROBOFLOW_API_KEY or os.environ.get('ROBOFLOW_API_KEY')
    if not key and IN_COLAB:
        from google.colab import userdata
        key = userdata.get('ROBOFLOW_API_KEY')
    if not key:
        raise RuntimeError('No Roboflow key: add the Colab Secret ROBOFLOW_API_KEY or fill ROBOFLOW_API_KEY above.')
    from roboflow import Roboflow
    ds = Roboflow(api_key=key).workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION).download(
        RF_FORMAT, location=str(DATASET_DIR))
    return Path(ds.location)


def dataset_yaml_abs(root):
    """Roboflow's data.yaml uses '../train/images'; Ultralytics needs paths it can resolve. Writes data_abs.yaml."""
    names = load_names(root)
    cfg = {'path': str(Path(root).resolve()), 'names': dict(enumerate(names))}
    for key, split in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        cfg[key] = str(find_split_dir(root, split).relative_to(root) / 'images')
    out = Path(root) / 'data_abs.yaml'
    yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False)
    return out


def dataset_summary(root):
    names = load_names(root)
    print('classes:', names)
    for split in ('train', 'valid', 'test'):
        d = find_split_dir(root, split)
        imgs = sorted(p for p in (d / 'images').iterdir() if p.suffix.lower() in IMG_EXT)
        sizes = {}
        for p in imgs:
            h, w = cv2.imread(str(p)).shape[:2]
            sizes[f'{w}x{h}'] = sizes.get(f'{w}x{h}', 0) + 1
        print(f'{split:5s}: {len(imgs):4d} images, sizes {sizes}')
    return names


def dhash(path, size=8):
    g = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def leakage_report(root, max_hamming=6):
    """Near-duplicate images across splits (overlapping UAV frames, augmentation copies). Any hit inflates test scores."""
    hashes = {}
    for s in ('train', 'valid', 'test'):
        d = find_split_dir(root, s) / 'images'
        hashes[s] = np.stack([dhash(p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT])
    for a, b in (('train', 'test'), ('train', 'valid'), ('valid', 'test')):
        dist = np.unpackbits(hashes[b][:, None, :] ^ hashes[a][None, :, :], axis=2).sum(2).min(1)
        print(f'{b} vs {a}: {int((dist <= max_hamming).sum())}/{len(dist)} near-duplicates '
              f'(median nearest Hamming distance {np.median(dist):.0f}/64)')


ds_root = download_dataset()
data_yaml = dataset_yaml_abs(ds_root)
class_names = dataset_summary(ds_root)
leakage_report(ds_root)

# %%
# ---------------------------------------------------------------- model budget (no data needed)
_m = build_model(ARCH, len(class_names) + 1, WIDTH, aux=False)
print(f'{ARCH} width {WIDTH}: {count_params(_m):,} params, {count_flops(_m, (TRAIN_IMGSZ, TRAIN_IMGSZ)):.3f} GMACs @ {TRAIN_IMGSZ}')
for s in sorted({256, 320, TRAIN_IMGSZ}):
    print(f'  peak INT8 activations @ {s}px: {peak_activation_bytes(_m, (s, s)) / 2**20:.2f} MiB')

# %%
# ---------------------------------------------------------------- train
RUNS = RESULTS / 'runs'
ckpts = []
for seed in SEEDS:
    out = RUNS / f'{ARCH}_w{WIDTH}_s{seed}'
    run(['train.py', '--data', ds_root, '--arch', ARCH, '--width', WIDTH, '--imgsz', TRAIN_IMGSZ,
         '--epochs', EPOCHS, '--bs', BATCH, '--seed', seed, '--workers', 2, '--out', out])
    ckpts.append(out / 'best.pt')

# %%
# ---------------------------------------------------------------- YOLO26n-seg baselines (same split, imgsz, epochs)
from ultralytics import YOLO

yolo_ckpts = {}
for tag, init in YOLO_BASELINES.items():
    y = YOLO(init)
    y.train(data=str(data_yaml), imgsz=TRAIN_IMGSZ, epochs=EPOCHS, batch=BATCH, seed=SEEDS[0],
            pretrained=init.endswith('.pt'), project=str(RESULTS / 'yolo'), name=tag, exist_ok=True,
            plots=False, device=0 if torch.cuda.is_available() else 'cpu', workers=2)
    yolo_ckpts[tag] = RESULTS / 'yolo' / tag / 'weights' / 'best.pt'
if EXISTING_YOLO.is_file():
    yolo_ckpts['existing_v4'] = EXISTING_YOLO

# native instance metrics (mask mAP) on test, for reference
yolo_native = {}
for tag, w in yolo_ckpts.items():
    m = YOLO(str(w)).val(data=str(data_yaml), split='test', imgsz=TRAIN_IMGSZ, plots=False, verbose=False)
    yolo_native[tag] = {'mask_mAP50': float(m.seg.map50), 'mask_mAP50_95': float(m.seg.map)}
print(json.dumps(yolo_native, indent=2))

# %%
# ---------------------------------------------------------------- YOLO confidence threshold: chosen on VALID only
yolo_conf = {}
for tag, w in yolo_ckpts.items():
    out = RESULTS / 'yolo_conf_on_valid' / tag
    run(['evaluate.py', '--data', ds_root, '--split', 'valid', '--imgsz', TRAIN_IMGSZ, '--yolo', w,
         '--yolo-conf', *YOLO_CONF_SWEEP, '--no-bench', '--viz', 0, '--out', out])
    rows = json.load(open(out / 'results.json'))['rows']
    best = max(rows, key=lambda r: r['mIoU_crops'])
    yolo_conf[tag] = float(best['model'].rsplit('@conf', 1)[1])
    print(f'{tag}: conf {yolo_conf[tag]} (valid mIoU crops {best["mIoU_crops"]:.4f})')

# %%
# ---------------------------------------------------------------- comparison on the held-out TEST split
eval_results = {}
for tag, w in yolo_ckpts.items():
    out = RESULTS / 'compare_test' / tag
    run(['evaluate.py', '--data', ds_root, '--split', 'test', '--imgsz', TRAIN_IMGSZ, '--custom', *ckpts,
         '--yolo', w, '--yolo-conf', yolo_conf[tag], '--viz', 6, '--out', out])
    eval_results[f'{TRAIN_IMGSZ}px vs {tag}'] = json.load(open(out / 'results.json'))
for s in EVAL_SIZES[1:]:          # Nano at a microcontroller-class input size
    out = RESULTS / f'nano_test_{s}'
    run(['evaluate.py', '--data', ds_root, '--split', 'test', '--imgsz', s, '--custom', *ckpts, '--viz', 0, '--out', out])
    eval_results[f'{s}px nano only'] = json.load(open(out / 'results.json'))

print(f"\n{'setting':<28}{'model':<26}{'params':>11}{'GFLOPs':>8}{'mIoU crops':>12}{'pix acc':>9}")
for k, r in eval_results.items():
    for row in r['rows']:
        print(f"{k:<28}{row['model']:<26}{(row.get('params') or 0):>11,}{(row.get('gflops') or 0):>8.2f}"
              f"{row['mIoU_crops']:>12.4f}{row['pixel_acc']:>9.4f}")
    for pair, (m, lo, hi) in r.get('paired_bootstrap_mIoU_crops(mean,lo95,hi95)', {}).items():
        verdict = 'Nano better' if lo > 0 else 'YOLO better' if hi < 0 else 'no significant difference'
        print(f"   {pair}: dmIoU {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
json.dump({'yolo_native_test': yolo_native, 'yolo_conf_from_valid': yolo_conf,
           'comparison': eval_results}, open(RESULTS / 'comparison_summary.json', 'w'), indent=2)

# %%
# ---------------------------------------------------------------- export: ONNX FP32 + INT8, accuracy of the exported graphs
DEPLOY = RESULTS / 'deploy'
run(['onnx_tools.py', '--ckpt', ckpts[0], '--data', ds_root, '--imgsz', TRAIN_IMGSZ, '--split', 'test', '--out', DEPLOY])
run(['onnx_tools.py', '--ckpt', ckpts[0], '--data', ds_root, '--imgsz', 320, '--split', 'test', '--out', DEPLOY / 'mcu_320'])


# %%
# ---------------------------------------------------------------- per-patch area and GLI from the custom model
def area_per_pixel_m2(img_w, img_h):
    if not IMAGES_ARE_FULL_FRAMES:
        return float('nan')
    fw = CAMERA['height_agl_m'] * CAMERA['sensor_w_mm'] / CAMERA['focal_mm']
    fh = CAMERA['height_agl_m'] * CAMERA['sensor_h_mm'] / CAMERA['focal_mm']
    return (fw / img_w) * (fh / img_h)


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


def draw_label(img, text, x, y, placed):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    H, W = img.shape[:2]
    for dx, dy in ((-tw // 2, -10), (-tw // 2, tht + 20), (12, 0), (-tw - 12, 0)):
        x0 = int(np.clip(x + dx, 4, max(4, W - tw - 4))); y0 = int(np.clip(y + dy, tht + 4, H - base - 4))
        box = (x0 - 4, y0 - tht - 4, x0 + tw + 4, y0 + base + 4)
        if not any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]) for b in placed):
            break
    cv2.rectangle(img, box[:2], box[2:], (0, 0, 0), -1)
    cv2.putText(img, text, (x0, y0), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    placed.append(box)


def analyse_image(path, label_map, names):
    """label_map: 0 = background, c+1 = class c. Patches = connected regions of one class."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    gli, a_px = gli_map(img), area_per_pixel_m2(w, h)
    vis, placed, rows = img.copy(), [], []
    min_px = MIN_PATCH_FRACTION * h * w
    for c, name in enumerate(names, start=1):
        n, comp, stats, cent = cv2.connectedComponentsWithStats((label_map == c).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_px:
                continue
            own = comp == k
            g = gli[own]; g = g[np.isfinite(g)]
            veg = g > GLI_VEG_THRESHOLD
            gli_all = float(g.mean()) if g.size else np.nan
            cond = condition(name, gli_all)
            area = int(own.sum()) * a_px
            rate = REFERENCE_YIELD_KG_HA.get(name, np.nan) / 1e4 * YIELD_FACTORS[cond]
            rows.append({'image': path.name, 'patch': len(rows) + 1, 'class': name, 'pixels': int(own.sum()),
                         'area_m2': area, 'veg_fraction': float(veg.mean()) if g.size else np.nan,
                         'gli_mean': gli_all, 'gli_median': float(np.median(g)) if g.size else np.nan,
                         'gli_mean_veg_only': float(g[veg].mean()) if veg.any() else np.nan,
                         'condition_indicative': cond, 'indicative_yield_kg': area * rate})
            col = np.array(PALETTE[(c - 1) % len(PALETTE)], np.float32)
            vis[own] = (vis[own].astype(np.float32) * 0.55 + col * 0.45).astype(np.uint8)
            draw_label(vis, f'#{len(rows)} {name} A:{area:.1f}m2 GLI:{gli_all:.2f}',
                       int(cent[k][0]), int(cent[k][1]), placed)
    return vis, rows


def run_patch_analysis(ckpt, image_dir, out_dir=RESULTS, imgsz=TRAIN_IMGSZ):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = build_from_ckpt(ckpt, dev)
    names = ck['cfg']['names']
    (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMG_EXT):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        vis, rr = analyse_image(path, predict_image(model, rgb, imgsz, dev), names)
        cv2.imwrite(str(out_dir / 'overlays' / path.name), vis)
        rows += rr
    if rows:
        with open(out_dir / 'segmentation_patches.csv', 'w', newline='') as f:
            wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(out_dir / 'SEGMENTATION_REPORT.txt', 'w') as f:
        f.write(f'AgriSegNet-Nano crop segmentation report\n{"=" * 70}\n')
        f.write(f'checkpoint {ckpt} | arch {ck["cfg"].get("arch")} | width {ck["cfg"]["width"]} | imgsz {imgsz}\n')
        f.write(f'camera {CAMERA} | full frames: {IMAGES_ARE_FULL_FRAMES}\n')
        for k, r in eval_results.items():
            for row in r['rows']:
                f.write(f"test {k} | {row['model']}: mIoU(crops) {row['mIoU_crops']:.4f}, per class {row['per_class_iou']}\n")
        f.write('Condition and yield are INDICATIVE (unvalidated thresholds and factors).\n\n')
        for name in sorted({r['class'] for r in rows}):
            rr = [r for r in rows if r['class'] == name]
            f.write(f'{name:<10s} patches {len(rr):3d}  area {sum(r["area_m2"] for r in rr):10.2f} m2  '
                    f'mean GLI {np.nanmean([r["gli_mean"] for r in rr]):.3f}\n')
        f.write(f'\nTotal area {sum(r["area_m2"] for r in rows):.2f} m2\n')
    print(f'{len(rows)} patches -> {out_dir}')
    return rows


seg_rows = run_patch_analysis(ckpts[0], find_split_dir(ds_root, 'test') / 'images')
