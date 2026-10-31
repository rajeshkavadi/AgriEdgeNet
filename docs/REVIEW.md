# Review: segmentation + Sequoia index notebooks, and the sub-50k CNN proposal

Date: 2026-10-31. Inputs reviewed: `v4_segmentation+_indices.ipynb`, `12_parrot_images.ipynb` (the Drive copy
shared on 2026-10-29; the 15.7 MB "(corrected)" copy is over the Drive connector's 10 MB limit and was
not readable), `agriseg.zip`, and the literature review *UAV Edge Computing for Precision Agriculture*.
One Sequoia capture (`IMG_260707_103958_0064`, NIR/RED/REG) was downloaded and used to test every
radiometric and alignment claim below.

## Verdict

* **The Sequoia NDVI and NDRE previously produced are wrong, and not marginally.** Radiometric correction
  never ran. Median NDRE on the test capture moves from 0.39 to 0.07 once the sensor model is applied.
  Discard the old index outputs and the aligned-vs-unaligned comparison built on them.
* **The segmentation notebook ran but measured nothing.** It produced overlays from possibly the wrong
  weights, with a leaked API key, double-counted overlapping areas, and never computed accuracy. The
  "yield" figures are scenarios built on unvalidated thresholds.
* **The consolidated notebook can run both pipelines, but it cannot produce per-crop NDVI.** The 12 Sequoia
  frames show tree canopy over a road, not the five crops, and the RGB frames come from a different camera
  with no shared georeference.
* **"< 50k parameters and better than existing models" is not yet a testable claim.** The model fits
  (33k parameters, built and smoke-tested), but the claim has no defined metric, split, baseline or
  deployment target. Section 4 rewrites it into one that can survive review.

## 1. Notebook 1: RGB segmentation + GLI (`notebooks/01_rgb_segmentation_gli.ipynb`)

| # | Issue | Consequence | Fix |
|---|---|---|---|
| 1 | Roboflow API key hard-coded in cells 0 and 8; the notebook is shared on Drive | Anyone with the file can use your Roboflow account | **Rotate the key now.** The notebook reads `ROBOFLOW_API_KEY` from Colab Secrets |
| 2 | Notes say the correct weights are `v4_best_epoch78.pt`, but `segment_best.pt` was run; `prepare()` silently falls back to any `*54*.pt` | Reported results may come from a different model | One explicit `MODEL_PATH`, no fallback; SHA-256 and class list logged |
| 3 | Class names hard-coded, not read from `model.names` | Silent mislabelling if the export order differs | Read from model, asserted equal to `data.yaml` |
| 4 | `masks.data` (letterboxed inference frame) stretched to image size | Masks shifted for any non-square input | `retina_masks=True`, shape asserted. Tested on an 810x1080 image |
| 5 | Overlapping instances each counted in full | Area and yield inflated wherever masks overlap | Each pixel owned by the highest-confidence instance |
| 6 | GSD = footprint / 640 applied to masks resized to the image's own size | Area wrong unless images are exactly 640x640 *full* frames | GSD from each image's actual size; flag `IMAGES_ARE_FULL_FRAMES` |
| 7 | Flight height fixed at 20 m, camera = Raspberry Pi v2 (IMX219) | Area scales with H^2: a 10 % height error gives a 21 % area error | Left configurable; **use per-image AGL** |
| 8 | Mask-mean GLI mixes soil and canopy | GLI tracks canopy cover as much as plant condition | Reports `veg_fraction` and GLI over vegetation pixels only |
| 9 | GLI thresholds and yield factors have no source or validation | "Estimated yield" is reference yield x area x guess | Kept, renamed `indicative_*`; NaN gives `UNKNOWN`, not `MODERATE` |
| 10 | No evaluation | No accuracy number exists for the model | `model.val()` on the test split + a near-duplicate leakage check across splits |
| 11 | "NDVI" section ran on the IIT Tirupati paddy repo (`Set_1/Set_2`) with the same broken radiometry | Unrelated data, wrong method | Removed; Sequoia handled in notebook 2 |
| 12 | `cv2.putText` cannot draw `²` | Labels show `??` | `m2` |

The leakage check matters most. The old cell 17 expects 18 test images. If Roboflow split overlapping
frames from one flight at random, test mAP is optimistic. Five of the 14 papers in your review fail the
same way.

## 2. Notebook 2: Sequoia calibration, alignment and indices (`notebooks/02_sequoia_alignment_indices.ipynb`)

### 2.1 Radiometry was a no-op (evidence)

The old code read the exiftool keys `Black Level`, `Sensor Gain` and `Irradiance`. Sequoia TIFFs contain none of
them; the calibration lives in XMP (`Camera:SensorModel`, `Camera:VignettingPolynomial2D`,
`Camera:IrradianceList`, ...). Every band therefore used black = 64, gain = 1, irradiance = 1, which is visible
in the old printout. Real values from the test capture:

| Band | SensorModel A | B (dark, DN) | C | Min DN in frame | Vignetting, centre to corner |
|---|---|---|---|---|---|
| NIR | 3.006e8 | 4310 | 3422 | 4992 | 0.98 to 0.73 |
| RED | 2.758e8 | 4617 | 2830 | 4288 | 0.98 to 0.72 |
| REG | 0.870e8 | 4192 | 1121 | 4032 | 0.98 to 0.70 |

Parrot model: `I = f^2 (DN - B) / (A * t * ISO + C)`, then divide by the vignetting polynomial. Ablation on
the same registered capture (median over valid pixels):

| Radiometry | NDVI | NDRE |
|---|---|---|
| Old: DN - 64 | 0.418 | 0.392 |
| + dark offset B | 0.690 | 0.608 |
| + band sensitivity A, C, exposure, ISO, f-number | 0.667 | 0.085 |
| + vignetting (full model) | 0.662 | 0.073 |

The dark offset accounts for most of the NDVI error and the per-band sensitivity (A differs 3.5x between RED and REG)
for the NDRE error. Vignetting barely moves the ratio indices, because it is similar across bands, but it matters
for any band-level reflectance. Sanity check: after correction, NDRE > NDVI on 0.07 % of vegetation pixels; the
old outputs had NDRE > NDVI *on average*. The notebook now runs this check automatically and warns.

### 2.2 Alignment

* Bugs fixed: the affine composition for NIR -> REG -> RED was multiplied in the wrong order; NCC was scored over
  the zero-filled border; failures fell back to **identity** (about 20 px off) rather than the rig transform.
* The reference band is now REG, registered directly to NIR and RED. Every capture is checked against the
  **flight-median** transform (the rig is rigid), and outliers fall back to that median.
* **Lens undistortion was tested and dropped.** I expected per-band fisheye differences to defeat a single
  homography. The data says otherwise: on raw frames, residual misregistration measured at the four corners
  is 0.3 to 0.7 px (NIR) and 0.3 to 1.0 px (RED). My conversion of the Pix4D fisheye model made it worse
  (about 40 px), most likely a principal-point convention I could not verify. Undistortion is needed for
  orthorectification and area, not for band-to-band indices.
* RED -> REG remains the weak pair (ECC 0.45 vs 0.89 for NIR). Watch `alignment_summary.csv` over all 4 captures.

### 2.3 What the indices still are not

Without a reflectance panel, the outputs are **calibrated at-sensor irradiance ratios**, normalised to the
flight's Sunshine irradiance. They are consistent within a flight but are not surface reflectance, and NDVI
thresholds from the literature do not transfer. Photograph the panel before and after each flight and fill in
`PANEL`. Two further unverified points: the Sunshine gain-index-to-gain mapping is assumed (it only matters if
the index changes during a flight, and the notebook warns if it does); and the XMP camera roll reads -37 deg,
which needs checking against the mount before trusting any nadir assumption.

## 3. Consolidated notebook (`notebooks/03_combined_segmentation_indices.ipynb`)

The notebook runs part 1 then part 2 and packages both result sets (tested end to end). It does **not** mask the
indices by crop: no pixel correspondence exists between the RGB dataset and the Sequoia frames (different
cameras, no georeference, different scenes). Two routes would give per-crop NDVI:

1. Fly the Sequoia over the crop plots and segment its own 16 MP RGB, or the registered bands, so masks and
   indices share a pixel grid; or
2. build orthomosaics of both sensors on one CRS and rasterise the masks onto the index grid.

## 4. The proposal: a custom CNN under 50k parameters that beats existing models

**Claim as stated.** A CNN with < 50,000 parameters segments the five crops better than existing models.

**What is built.** `agriseg/agrisegnet.py` now has `AgriSegNetNano` (`--arch nano`); see section 7 for the
current, device-agnostic version (34,742 parameters, 0.25 GMACs at 640x640). The existing AgriSegNet width-scaled to 0.25 is 43,894 parameters and makes the natural second
candidate, because width scaling was the winning strategy in Menshchikov et al. For scale: YOLO11n-seg is 2.87M
parameters and 9.8 GFLOPs; YOLO26n-seg is the same class.

### Assumption audit

| Assumption | Status |
|---|---|
| Parameter count is the binding constraint on the device | **False for Orin.** At 640 px the nano model's INT8 weights are 32 KiB but peak activations are about 3.1 MiB. Going from 491k to 33k parameters cuts peak activation memory only from 3.5 to 3.1 MiB |
| A smaller model buys flight time | Weak. An octocopter hovers at hundreds of watts; Orin draws 7 to 15 W. Model-dependent power differences are a fraction of a percent of flight energy. Measure it before claiming it |
| Real-time needs a smaller model | Not on Orin. Your review cites 25 to 80 ms end-to-end for YOLO-class models on Orin Nano with TensorRT FP16; the camera delivers about 1 frame/s |
| "Better" is defined | No. Mask mAP (YOLO, instance) and mIoU (semantic) are not the same metric. `agriseg/evaluate.py` converts YOLO to semantic, but the result depends on the confidence threshold, which must be tuned on valid |
| The test set can resolve a difference | Doubtful. 18 test images, probably from a random split of one or few flights. Image-level bootstrap CIs will be several mIoU points wide |
| Existing models are the right comparison | Incomplete. Also needed: a colour/texture classical baseline (if GLI + texture separates the crops, a CNN "win" is trivial), a tiny U-Net at the same budget (Menshchikov W=4: about 0.05M parameters), and LR-ASPP-MobileNetV3 |

### Strongest counterargument

A COCO-pretrained YOLO26n-seg, fine-tuned and run as INT8 TensorRT, already exceeds the 1 Hz requirement on Orin with
a wide margin. A from-scratch 33k model has around 85x less capacity and no pretraining, so it should be
*worse* under domain shift (new field, growth stage, light, altitude). If it wins on the current test set, the
likeliest explanations are, in order: (a) the task is easy, with large homogeneous colour-separable patches;
(b) the instance-to-semantic conversion penalises YOLO (gaps between instances count as background); (c) split
leakage. None of these is an architectural contribution.

### Failure modes to test before claiming anything

Altitude/GSD change, white-balance and sun-angle change, early growth stage (gram, fenugreek and linseed look
alike before canopy closure), field boundaries and mixed pixels, and INT8 quantisation drop. Tiny models typically degrade
faster on all of these.

### Evidence that would make the claim publishable

1. **Split by field or date**, never by random image. Run the leakage check in notebook 1 first.
2. **Pareto, not a single number.** mIoU (crops) vs measured latency and J/frame on the target device, with
   nano (w = 0.75 and 1.0), AgriSegNet (w = 0.25, 0.5, 1.0), a tiny U-Net, LR-ASPP and YOLO26n/s-seg. Use 3 seeds
   each, paired bootstrap CIs, and per-class IoU.
3. **The device that motivates 50k.** If it is Orin, report FP16 and INT8 TensorRT latency and tegrastats power;
   the 50k constraint will probably not matter there. If it is an MCU or NPU (STM32N6, ESP32-S3, Hailo), name it:
   then peak activation memory, not parameters, is the budget, and the model must be designed for it (smaller
   input tiles, argmax at 1/8 instead of a full-resolution logit map).
4. **Domain shift.** Train on one field or date, test on another.

### A defensible version of the claim

> At <= 50k parameters and <= 0.5 GFLOPs, AgriSegNet-Nano comes within X mIoU (95 % CI) of YOLO26n-seg on
> held-out fields, while using Y x less energy per frame on [device], which makes onboard crop mapping possible
> on hardware where YOLO-class models do not fit.

That is a Pareto claim with a deployment reason. Your literature review already argues that the bigger gap is
in-flight evaluation (latency, watts, J/ha) rather than another architecture. A small model is worth building
only if it serves that measurement.

### Commands

```bash
# candidates under 50k (budget enforced; aborts above 50,000)
python agriseg/train.py --data <export> --imgsz 640 --arch nano --width 1.0 --seed 0 --out runs/nano_w1_s0
python agriseg/train.py --data <export> --imgsz 640 --arch agrisegnet --width 0.25 --max-params 50000 --seed 0 --out runs/agri_w025_s0
# compare on test against YOLO (tune YOLO conf on valid first, see agriseg/README.md)
python agriseg/evaluate.py --data <export> --split test --imgsz 640 \
    --custom runs/nano_w1_s*/best.pt --yolo v4_best_epoch78.pt --yolo-conf 0.25 --out results/compare
```

## 5. Code changes in `agriseg/`

* `train.py`: crashed with `--aux-w 0` (indexed a tensor with `"out"`); fixed. Worker seeds mixed in wall-clock
  time, so `--seed` did not reproduce a run; fixed. Added `--arch` and `--max-params` (defaults: 50k for nano, 1M
  for agrisegnet).
* `engine.py`: checkpoints record `arch`; old checkpoints still load as `agrisegnet`.
* `agrisegnet.py`: `AgriSegNetNano`, `build_model`, and a parameter/GMAC table (`python agriseg/agrisegnet.py`).

## 6. Not verified here

* The "(corrected)" Parrot notebook (15.7 MB, over the Drive connector's 10 MB limit). If it differs from the
  shared copy beyond outputs, tell me what changed.
* The YOLO weights and the Roboflow dataset. Inference paths were tested with `yolo11n-seg.pt` on Ultralytics
  sample images. I deliberately did not use the leaked key to download your data.
* Only 1 of the 4 Sequoia captures, so flight-median fallback was exercised only trivially.
* The repo's `agriedgenet.py` (874k parameters, 4-band, multitask) conflicts with the new 50k target. It is left
  untouched until you decide which direction to keep.

## 7. Round 2 (device-agnostic target, custom CNN in notebook 1, 12 Sequoia images)

**Requirement as clarified:** one model for any device (MCU, Raspberry Pi, Jetson Nano/Orin), light enough to share
the device with other scheduled tasks; it replaces YOLO26 in notebook 1.

**Model changes.** `AgriSegNetNano` was redesigned for operator portability, not just size:
* The dilated convolutions are gone (microcontroller kernels support dilation unevenly). Receptive field now
  comes from a 1/32 stage plus a global-average-pool gate.
* Upsampling uses fixed x2/x8 factors, so the exported graph is static. It contains only Conv, Add, Mul, Clip,
  GlobalAveragePool, Sigmoid and Resize; no Shape, Slice or Concat ops.
* There is an optional 1/8-resolution output (`--output-stride 8`) for hosts that upsample the label map themselves.

| | params | GMACs @640 | peak INT8 activations @256 / 320 / 640 |
|---|---|---|---|
| nano w=0.5 | 11,390 | 0.10 | 0.38 / 0.60 / 2.38 MiB |
| nano w=1.0 (default) | 34,742 | 0.25 | 0.44 / 0.68 / 2.73 MiB |

**What decides the device is the input size, not the parameter count.** At 640 px the input and first layer
alone need more than 2 MiB, which is beyond most microcontrollers. At 256 to 320 px the model fits in under
0.7 MiB. Notebook 1 therefore evaluates at the training size and at 320 px, and exports both. If 320 px costs
too much accuracy on your data, tile the frame rather than widen the model.

**Notebook 1 now:** downloads the Roboflow train/valid/test export, reports image sizes, checks for split leakage,
trains nano, scores test mIoU at two input sizes (with YOLO26 as an optional baseline if its weights are present),
exports FP32 and INT8 ONNX and scores those too, then runs the GLI and area analysis on connected regions of the
custom model's label map. On the synthetic set the INT8 graph's mIoU stayed within ~1 point of FP32 (it came out
0.9 to 1.3 points *higher*, which is noise); measure the drop on your data.

**Indices on all 12 images.** All four captures register with consistent transforms. Centre shift relative to REG:
NIR (-18.9, +3.2 to +3.5) px, RED (-12.3 to -12.5, +22.9 to +23.3) px; maximum deviation from the flight median
1.6 px. That is what a rigid rig should produce, so the flight-median fallback was not triggered. Median NDVI is
0.55 to 0.66 and median NDRE 0.03 to 0.08. NDRE > NDVI on at most 0.21 % of vegetation pixels in any capture.

**Still open.** I have not trained on your Roboflow data (the key must not be used from here), so there is no real
accuracy number yet. For any reported result, run 3 seeds and a field/date-held-out split. The MCU path
(ONNX -> TFLite-Micro) is not automated, and the repo's `agriedgenet.py` (874k params, 4-band) is not used by
this pipeline.
