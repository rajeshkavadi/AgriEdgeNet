# %% [markdown]
# # Parrot Sequoia: radiometric calibration, band alignment, NDVI and NDRE
#
# Corrected version of `12_parrot_images (corrected).ipynb`.
#
# **What was wrong in the previous version** (details in `docs/REVIEW.md`):
#
# 1. **Radiometry was a no-op.** The code asked exiftool for the tags `Black Level`, `Sensor Gain` and
#    `Irradiance`. Sequoia files have none of them, so every band fell back to black = 64, gain = 1,
#    irradiance = 1 (visible in the old output). The real dark level is about 4,300 to 4,600 DN (the
#    `SensorModel` B coefficient) and band sensitivity differs about 3x between RED and REG
#    (coefficient A). The old NDVI and NDRE were therefore ratios of raw DN. That is why mean NDRE
#    came out *higher* than mean NDVI, which is physically backwards for vegetation.
# 2. **No vignetting correction.** Corners are 17 to 30 % darker than the centre (`VignettingPolynomial2D`).
# 3. Alignment bugs: affine composition in the wrong order (NIR -> REG -> RED), NCC scored on the
#    zero-filled border, and failed captures fell back to *identity* (about 20 px misregistration)
#    instead of the rig's known transform.
# 4. NDVI category thresholds (0.2 / 0.5) were reused for NDRE, which has a different range.
# 5. "Reflectance" CSVs were each band divided by its own maximum, which destroys the band ratios.
#    They are removed.
#
# **What this notebook does**
#
# 1. Reads the calibration metadata straight from each TIFF (XMP + EXIF), with no exiftool.
# 2. DN -> at-sensor irradiance with Parrot's model `I = f^2 (p - B) / (A*t*ISO + C)`, then divides by the
#    vignetting polynomial.
# 3. Optional illumination normalisation from the Sunshine sensor, and optional reflectance-panel
#    calibration.
# 4. Registers the bands (gradient images, phase-correlation start, coarse-to-fine ECC homography), checks
#    every capture against the flight median, and falls back to the median transform when a capture fails.
# 5. Computes NDVI and NDRE on the common valid area and saves GeoTIFFs, PNGs, QC overlays and CSVs.
#
# **Read before quoting numbers:** without a reflectance panel the indices are computed from
# calibrated at-sensor irradiance, not surface reflectance. They are consistent within a flight but are
# not comparable to literature thresholds. Photograph the panel before and after each flight and fill in
# `PANEL` below.

# %%
# ---------------------------------------------------------------- setup
import os, sys, re, json, base64, struct, subprocess, warnings
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rasterio', 'pillow', 'opencv-python-headless'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')

import numpy as np
import pandas as pd
import cv2
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# %%
# ---------------------------------------------------------------- configuration
# The 12 TIFs (4 captures x NIR/RED/REG). A folder that was *shared with you* is not under MyDrive until you
# right-click it in Drive -> Organize -> Add shortcut -> My Drive. Then this path works.
SEQ_IMAGE_FOLDER = Path(os.environ.get('SEQUOIA_DIR', '/content/drive/MyDrive/parrot 12 images'))
# Outputs go to your own Drive (writing into someone else's shared folder can fail on permissions)
SEQ_OUT_ROOT = Path(os.environ.get('SEQUOIA_OUT', '/content/drive/MyDrive/AgriEdgeNet_Results/Sequoia_Indices'))
SEQ_EXPECTED_CAPTURES = 4

SEQ_BAND_SUFFIXES = {'GRE': ['GRE'], 'RED': ['RED'], 'REG': ['REG'], 'NIR': ['NIR']}
SEQ_BAND_ORDER = ['GRE', 'RED', 'REG', 'NIR']
SEQ_REFERENCE_BAND = 'REG'      # spectrally between RED and NIR: the best-correlated partner for both

# Radiometry
NORMALISE_ILLUMINATION = True   # scale each capture by Sunshine irradiance relative to the flight median
SUNSHINE_GAIN = {0: 1.0, 1: 4.0, 2: 16.0, 3: 64.0}   # gain index -> gain; only matters if indices differ
# Reflectance panel (set to None if no panel image was taken).
# 'capture' is the capture prefix of the panel shot, 'roi' = (row0, row1, col0, col1) INSIDE the panel
# in the aligned frame, 'reflectance' = the panel's certified reflectance per band.
PANEL = None
# PANEL = {'capture': 'IMG_260707_103000_0001', 'roi': (400, 520, 560, 700),
#          'reflectance': {'GRE': 0.18, 'RED': 0.18, 'REG': 0.18, 'NIR': 0.18}}

# Registration
PYRAMID_LEVELS = 4
ECC_MAX_ITER = 300
ECC_EPS = 1e-6
GRAD_BLUR_SIGMA = 1.5
MIN_ECC_CC = 0.30               # below this a capture is flagged
MAX_DEV_FROM_MEDIAN_PX = 3.0    # corner deviation from the flight-median transform before fallback

# Index category thresholds: illustrative only, calibrate against field data
INDEX_CLASSES = {
    'NDVI': (0.2, 0.5),
    'NDRE': (0.1, 0.3),
}


# %%
# ---------------------------------------------------------------- metadata (XMP + EXIF, no exiftool)
def _xmp_text(path):
    raw = Path(path).read_bytes()
    s, e = raw.find(b'<x:xmpmeta'), raw.find(b'</x:xmpmeta>')
    if s < 0 or e < 0:
        raise ValueError(f'{path}: no XMP block; is this an original Sequoia TIFF?')
    return raw[s:e].decode('utf-8', 'replace')


def _xmp_attr(xmp, key):
    m = re.search(rf'Camera:{key}="([^"]*)"', xmp)
    return m.group(1) if m else None


def _xmp_seq(xmp, key):
    m = re.search(rf'<Camera:{key}>.*?<rdf:li>(.*?)</rdf:li>', xmp, re.S)
    return m.group(1).strip() if m else None


def _floats(s):
    return [float(v) for v in s.split(',')] if s else None


def read_sequoia_meta(path):
    """All calibration fields needed downstream. Raises if a required field is missing,
    instead of silently falling back to defaults (the old code's failure mode)."""
    xmp = _xmp_text(path)
    im = Image.open(path)
    exif = im.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in list(exif.items()) + list(exif.get_ifd(0x8769).items())}

    sensor_model = _floats(_xmp_seq(xmp, 'SensorModel'))
    vig_pw = _xmp_seq(xmp, 'VignettingPolynomial2DName')
    vig_c = _floats(_xmp_seq(xmp, 'VignettingPolynomial2D'))
    meta = {
        'path': str(path),
        'band_name': _xmp_seq(xmp, 'BandName'),
        'exposure_s': float(tags.get('ExposureTime', 'nan')),
        'iso': float(tags.get('ISOSpeedRatings', 'nan')),
        'fnumber': float(tags.get('FNumber', 'nan')),
        'sensor_model': sensor_model,                     # A, B, C
        'vig_powers': [int(v) for v in vig_pw.split(',')] if vig_pw else None,
        'vig_coeffs': vig_c,
        'camera_ypr_deg': [_floats(_xmp_attr(xmp, k))[0] if _xmp_attr(xmp, k) else np.nan
                           for k in ('Yaw', 'Pitch', 'Roll')],
        'irradiance': None,
    }
    required = ['sensor_model', 'vig_powers', 'vig_coeffs']
    missing = [k for k in required if not meta[k]] + \
              [k for k in ('exposure_s', 'iso', 'fnumber') if not np.isfinite(meta[k])]
    if missing:
        raise ValueError(f'{Path(path).name}: missing calibration metadata {missing}')

    b64 = re.search(r'<Camera:IrradianceList>\s*(.*?)\s*</Camera:IrradianceList>', xmp, re.S)
    if b64:
        raw = base64.b64decode(b64.group(1).strip())
        recs = [struct.unpack('<QHHHH3f', raw[i:i + 28]) for i in range(0, len(raw) - 27, 28)]
        # (timestamp_us, ch0_count, ch1_count, gain_index, integration_ms, yaw, pitch, roll)
        if recs:
            meta['irradiance'] = {
                'gain_index': sorted({r[3] for r in recs}),
                'value': float(np.median([r[1] / (SUNSHINE_GAIN[r[3]] * r[4]) for r in recs])),
                'min_count': int(min(r[1] for r in recs)),
            }
    return meta


# %%
# ---------------------------------------------------------------- radiometry
def vignetting_map(meta, shape):
    """Parrot 2-D vignetting polynomial on normalised coordinates x=col/W, y=row/H."""
    h, w = shape
    x = (np.arange(w, dtype=np.float64) / w)[None, :]
    y = (np.arange(h, dtype=np.float64) / h)[:, None]
    p = meta['vig_powers']
    v = np.zeros(shape, np.float64)
    for c, i, j in zip(meta['vig_coeffs'], p[0::2], p[1::2]):
        v += c * x ** i * y ** j
    return v.astype(np.float32)


def dn_to_sensor_irradiance(dn, meta):
    """Parrot Sequoia sensor model:  I = f^2 * (p - B) / (A * t * ISO + C), then / vignetting."""
    A, B, C = meta['sensor_model']
    f, t, iso = meta['fnumber'], meta['exposure_s'], meta['iso']
    I = f * f * (dn.astype(np.float32) - B) / (A * t * iso + C)
    return I / vignetting_map(meta, dn.shape)


# %%
# ---------------------------------------------------------------- registration
def _unit(a):
    a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


def gradient_image(a):
    g = cv2.GaussianBlur(_unit(a), (0, 0), GRAD_BLUR_SIGMA)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return (m / (m.max() + 1e-12)).astype(np.float32)


def warp(img, W, shape, interp=cv2.INTER_LINEAR, border=0.0):
    h, w = shape
    return cv2.warpPerspective(img, W, (w, h), flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def register(ref_g, mov_g):
    """Homography W with warp(mov, W) ~ ref. Phase-correlation start, coarse-to-fine ECC."""
    pr, pm = [ref_g], [mov_g]
    for _ in range(PYRAMID_LEVELS - 1):
        pr.append(cv2.pyrDown(pr[-1])); pm.append(cv2.pyrDown(pm[-1]))
    pr, pm = pr[::-1], pm[::-1]
    (dx, dy), _ = cv2.phaseCorrelate(pr[0], pm[0])
    cands = [np.eye(3, dtype=np.float32)]
    for s in (1.0, -1.0):
        Wc = np.eye(3, dtype=np.float32); Wc[0, 2], Wc[1, 2] = s * dx, s * dy
        cands.append(Wc)
    W = cands[int(np.argmax([ncc(pr[0], warp(pm[0], c, pr[0].shape)) for c in cands]))]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)
    cc = float('nan')
    S = np.diag([0.5, 0.5, 1.0])
    for lvl, (r, m) in enumerate(zip(pr, pm)):
        try:
            cc, W = cv2.findTransformECC(r, m, W.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error:
            pass
        if lvl < len(pr) - 1:
            W = np.linalg.inv(S) @ W @ S
            W = (W / W[2, 2]).astype(np.float32)
    return W.astype(np.float32), float(cc)


def corner_deviation(W1, W2, shape):
    h, w = shape
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1], [w / 2, h / 2, 1]], np.float64).T
    a, b = W1 @ pts, W2 @ pts
    return float(np.max(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0)))


def centre_shift(W, shape):
    h, w = shape
    p = W @ np.array([w / 2, h / 2, 1.0])
    return float(p[0] / p[2] - w / 2), float(p[1] / p[2] - h / 2)


# %%
# ---------------------------------------------------------------- indices and output helpers
def normalized_difference(a, b, valid):
    with np.errstate(divide='ignore', invalid='ignore'):
        nd = (a - b) / (a + b)
    ok = valid & (a > 0) & (b > 0) & np.isfinite(nd)
    out = np.full(a.shape, np.nan, np.float32)
    out[ok] = nd[ok]
    return out


def class_breakdown(index, thresholds):
    v = index[np.isfinite(index)]
    lo, hi = thresholds
    n = max(v.size, 1)
    return {'low_%': 100 * np.sum(v < lo) / n, 'mid_%': 100 * np.sum((v >= lo) & (v <= hi)) / n,
            'high_%': 100 * np.sum(v > hi) / n}


def save_tif(path, arrays, names, tags=None):
    arrays = [np.asarray(a, np.float32) for a in arrays]
    h, w = arrays[0].shape
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=len(arrays),
                       dtype='float32', nodata=np.nan, compress='deflate') as dst:
        for i, (a, n) in enumerate(zip(arrays, names), 1):
            dst.write(a, i); dst.set_band_description(i, n)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def save_index_png(index, title, path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7, 5.4))
    cmap = matplotlib.colormaps['RdYlGn'].copy(); cmap.set_bad('black')
    im = ax.imshow(index, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=150); plt.close(fig)


def false_colour(bands, order):
    out = []
    for b in order:
        a = np.nan_to_num(bands[b], nan=0.0)
        lo, hi = np.nanpercentile(bands[b], (1, 99))
        out.append(np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))
    return np.dstack(out)


def save_qc(name, before, after, path):
    order = ['NIR', 'RED', 'GRE' if 'GRE' in after else 'REG']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].imshow(false_colour(before, order)); ax[0].set_title(f'{name}: before registration')
    ax[1].imshow(false_colour(after, order)); ax[1].set_title(f'{name}: registered (colour fringes = residual error)')
    for a in ax:
        a.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=120); plt.close(fig)


# %%
# ---------------------------------------------------------------- pipeline
def find_capture_sets(folder):
    lut = {s.upper(): b for b, sufs in SEQ_BAND_SUFFIXES.items() for s in sufs}
    pat = re.compile(r'^(?P<prefix>.+?)_(?P<band>' + '|'.join(sorted(lut, key=len, reverse=True)) + r')$', re.I)
    sets = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}:
            m = pat.match(p.stem)
            if m:
                sets.setdefault(m.group('prefix'), {})[lut[m.group('band').upper()]] = p
    return sets


def calibrate(files):
    """-> metas, sensor-irradiance bands (float32), shape."""
    metas = {b: read_sequoia_meta(files[b]) for b in files}
    dn = {b: cv2.imread(str(files[b]), cv2.IMREAD_UNCHANGED) for b in files}
    shapes = {a.shape for a in dn.values()}
    if len(shapes) != 1:
        raise ValueError(f'band shapes differ: {shapes}')
    bands = {b: dn_to_sensor_irradiance(dn[b], metas[b]) for b in files}
    return metas, bands, shapes.pop()


def run_sequoia_pipeline(folder=None, out_root=None):
    folder = Path(folder or SEQ_IMAGE_FOLDER)
    out_root = Path(out_root or SEQ_OUT_ROOT)
    if not folder.is_dir():
        raise FileNotFoundError(f'{folder} not found. If the folder was shared with you, add a shortcut to '
                                'My Drive (Drive -> right-click -> Organize -> Add shortcut), or set SEQ_IMAGE_FOLDER.')
    dirs = {k: out_root / k for k in ('Calibrated_Bands', 'NDVI', 'NDRE', 'Alignment_QC')}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    sets = find_capture_sets(folder)
    sets = {k: v for k, v in sets.items() if SEQ_REFERENCE_BAND in v and len(v) >= 2}
    if not sets:
        raise FileNotFoundError(f'No Sequoia capture sets with a {SEQ_REFERENCE_BAND} band in {folder}')
    print(f'{len(sets)} capture set(s) in {folder}')
    for k, v in sets.items():
        print(f'  {k}: {sorted(v)}')
    if len(sets) != SEQ_EXPECTED_CAPTURES:
        print(f'  WARNING: expected {SEQ_EXPECTED_CAPTURES} captures, found {len(sets)}')

    # ---- pass 1: radiometry and per-capture registration
    cap = {}
    for name, files in sets.items():
        metas, bands, shape = calibrate(files)
        ref_g = gradient_image(bands[SEQ_REFERENCE_BAND])
        reg = {}
        for b in bands:
            if b == SEQ_REFERENCE_BAND:
                continue
            W, cc = register(ref_g, gradient_image(bands[b]))
            reg[b] = {'W': W, 'cc': cc}
        cap[name] = {'files': files, 'metas': metas, 'bands': bands, 'shape': shape, 'reg': reg}
        print(f'  {name}: ' + ', '.join(f'{b} cc={r["cc"]:.3f}' for b, r in reg.items()))

    # ---- pass 2: rig consistency. The bands are rigidly mounted, so transforms should agree across
    # captures at similar altitude. Captures that disagree with the flight median use the median.
    bands_all = sorted({b for c in cap.values() for b in c['reg']})
    median_W = {}
    for b in bands_all:
        Ws = [c['reg'][b]['W'] for c in cap.values() if b in c['reg'] and c['reg'][b]['cc'] >= MIN_ECC_CC]
        if Ws:
            median_W[b] = np.median(np.stack(Ws), axis=0).astype(np.float32)

    # ---- illumination reference per band (flight median of Sunshine irradiance)
    sun_ref = {}
    for b in SEQ_BAND_ORDER:
        vals = [c['metas'][b]['irradiance']['value'] for c in cap.values()
                if b in c['metas'] and c['metas'][b]['irradiance']]
        if vals:
            sun_ref[b] = float(np.median(vals))
    gains = {b: c['metas'][b]['irradiance']['gain_index'] for c in cap.values() for b in c['metas']
             if c['metas'][b]['irradiance']}
    if len({tuple(g) for g in gains.values()}) > 1:
        print('  NOTE: Sunshine gain index varies; check SUNSHINE_GAIN mapping before trusting normalisation')

    align_rows, index_rows = [], []
    for name, c in cap.items():
        shape, bands = c['shape'], c['bands']
        ref_g = gradient_image(bands[SEQ_REFERENCE_BAND])
        aligned = {SEQ_REFERENCE_BAND: bands[SEQ_REFERENCE_BAND]}
        valid = np.isfinite(bands[SEQ_REFERENCE_BAND])
        for b, r in c['reg'].items():
            W, used = r['W'], 'ECC homography'
            dev = corner_deviation(W, median_W[b], shape) if b in median_W else float('nan')
            if r['cc'] < MIN_ECC_CC or (np.isfinite(dev) and dev > MAX_DEV_FROM_MEDIAN_PX):
                if b in median_W:
                    W, used = median_W[b], 'flight-median fallback'
                print(f'  {name} {b}: cc={r["cc"]:.3f}, deviation {dev:.1f} px -> {used}')
            mov_g = gradient_image(bands[b])
            out = warp(bands[b], W, shape, border=np.nan)
            ok = np.isfinite(out)
            ncc_before = ncc(ref_g, mov_g, valid & np.isfinite(bands[b]))
            ncc_after = ncc(ref_g, warp(mov_g, W, shape), valid & ok)
            aligned[b] = out
            valid &= ok
            sx, sy = centre_shift(W, shape)
            align_rows.append({'capture': name, 'band': b, 'reference': SEQ_REFERENCE_BAND, 'method': used,
                               'ecc_cc': r['cc'], 'grad_ncc_before': ncc_before, 'grad_ncc_after': ncc_after,
                               'centre_shift_x_px': sx, 'centre_shift_y_px': sy,
                               'dev_from_flight_median_px': dev, 'homography': json.dumps(np.round(W, 6).tolist())})
        # common valid area -> NaN elsewhere (no cropping, so every capture keeps the same pixel grid)
        for b in aligned:
            aligned[b] = np.where(valid, aligned[b], np.nan).astype(np.float32)

        # illumination normalisation (Sunshine), relative to the flight median per band
        level = 'sensor_irradiance'
        if NORMALISE_ILLUMINATION and all(b in sun_ref and c['metas'][b]['irradiance'] for b in aligned):
            for b in aligned:
                aligned[b] = aligned[b] * (sun_ref[b] / c['metas'][b]['irradiance']['value'])
            level = 'sensor_irradiance_sun_normalised'
        c['aligned'], c['valid'], c['level'] = aligned, valid, level

    # ---- optional panel calibration -> reflectance
    K_panel = None
    if PANEL:
        pc = cap.get(PANEL['capture'])
        if pc is None:
            raise KeyError(f'panel capture {PANEL["capture"]} not found')
        r0, r1, c0, c1 = PANEL['roi']
        K_panel = {b: PANEL['reflectance'][b] / float(np.nanmean(pc['aligned'][b][r0:r1, c0:c1]))
                   for b in pc['aligned']}
        print('  panel factors:', {b: f'{k:.4g}' for b, k in K_panel.items()})

    for name, c in cap.items():
        aligned, valid, level = c['aligned'], c['valid'], c['level']
        if K_panel:
            aligned = {b: a * K_panel[b] for b, a in aligned.items()}
            level = 'reflectance_panel'
        names = [b for b in SEQ_BAND_ORDER if b in aligned]
        save_tif(dirs['Calibrated_Bands'] / f'{name}_bands.tif', [aligned[b] for b in names], names,
                 {'radiometric_level': level, 'reference_band': SEQ_REFERENCE_BAND})
        save_qc(name, c['bands'], aligned, dirs['Alignment_QC'] / f'{name}_alignment_QC.png')

        row = {'capture': name, 'radiometric_level': level, 'valid_area_%': 100 * valid.mean()}
        idx = {}
        if 'RED' in aligned and 'NIR' in aligned:
            idx['NDVI'] = normalized_difference(aligned['NIR'], aligned['RED'], valid)
        if 'REG' in aligned and 'NIR' in aligned:
            idx['NDRE'] = normalized_difference(aligned['NIR'], aligned['REG'], valid)
        for k, a in idx.items():
            save_tif(dirs[k] / f'{name}_{k}.tif', [a], [k], {'radiometric_level': level})
            save_index_png(a, f'{name} {k} ({level})', dirs[k] / f'{name}_{k}.png',
                           -0.2 if k == 'NDVI' else -0.1, 1.0 if k == 'NDVI' else 0.6)
            v = a[np.isfinite(a)]
            row.update({f'{k}_mean': float(v.mean()), f'{k}_median': float(np.median(v)),
                        f'{k}_p05': float(np.percentile(v, 5)), f'{k}_p95': float(np.percentile(v, 95))})
            row.update({f'{k}_{kk}': vv for kk, vv in class_breakdown(a, INDEX_CLASSES[k]).items()})
        if {'NDVI', 'NDRE'} <= idx.keys():
            veg = idx['NDVI'] > 0.4
            row['NDRE_gt_NDVI_on_veg_%'] = float(100 * np.mean(idx['NDRE'][veg] > idx['NDVI'][veg])) if veg.any() else np.nan
            if row['NDRE_gt_NDVI_on_veg_%'] > 5:
                print(f'  WARNING {name}: NDRE > NDVI on {row["NDRE_gt_NDVI_on_veg_%"]:.1f}% of vegetation '
                      'pixels; radiometry is probably wrong')
        index_rows.append(row)
        c['indices'] = idx

    pd.DataFrame(align_rows).to_csv(out_root / 'alignment_summary.csv', index=False)
    summary = pd.DataFrame(index_rows)
    summary.to_csv(out_root / 'indices_summary.csv', index=False)
    with open(out_root / 'PROCESSING_REPORT.txt', 'w') as f:
        f.write('PARROT SEQUOIA: CALIBRATED + ALIGNED NDVI / NDRE\n' + '=' * 70 + '\n')
        f.write(f'Input folder        : {folder}\nReference band      : {SEQ_REFERENCE_BAND}\n')
        f.write('Radiometry          : I = f^2 (DN - B) / (A t ISO + C) / V(x,y)   [Parrot SensorModel, vignetting]\n')
        f.write(f'Illumination        : {"Sunshine-normalised to flight median" if NORMALISE_ILLUMINATION else "none"}\n')
        f.write(f'Panel calibration   : {"yes" if K_panel else "NO -> indices are not surface reflectance"}\n')
        f.write('Geometry            : ECC homography per band on gradient images (raw frames)\n\n')
        f.write(pd.DataFrame(align_rows).drop(columns='homography').to_string(index=False) + '\n\n')
        f.write(summary.to_string(index=False) + '\n')
    print(summary.round(3).to_string(index=False))
    print(f'\nOutputs: {out_root}')
    return cap, summary


# %%
if __name__ == '__main__' or IN_COLAB:
    if os.environ.get('SEQ_SKIP_RUN') != '1':
        seq_captures, seq_summary = run_sequoia_pipeline()
