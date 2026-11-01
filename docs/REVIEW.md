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

## 8. Round 3: baseline, early stopping, deployment benchmark, GSD

* **Baseline:** YOLO26n-seg **from scratch** (`yolo26n-seg.yaml`) only, matching Nano (no pretraining). It has 3.13M
  parameters and 10.6 GFLOPs at 640, against Nano's 34.7k. Both are trained on the same Roboflow split, input size,
  epoch cap (300) and seed.
* **Early stopping on valid**, patience 50 for both. Nano stops on valid mIoU (crops); Ultralytics stops on its own
  fitness (0.1 mAP50 + 0.9 mAP50-95). The criteria differ because the models' native outputs differ. A patience of 5
  stopped Nano at epoch 6 on a noisy early peak, hence 50.
* **Deployment benchmark (`agriseg/benchmark.py`):** both models are exported to static ONNX at 640, 480 and 320 px,
  in FP32 and INT8. INT8 uses the same recipe for both: ORT static QDQ, per-channel, Conv/MatMul, the same
  calibration images. Both are scored on test with ONNX Runtime; YOLO's confidence threshold comes from valid.
  Reported: mIoU (crops), per-class IoU, paired-bootstrap CI (Nano - YOLO) per size x precision, ONNX size, and CPU
  latency (1 thread). Checked: YOLO26n ONNX FP32 masks match PyTorch exactly (IoU 1.0); INT8 agrees at 0.987.
  **INT8 latency on ORT-CPU is not representative** (emulated and slower than FP32); INT8 speed must be measured on the
  target runtime.
* **GSD:** Pi Camera v2 (IMX219, 3.68 x 2.76 mm sensor, 3.04 mm lens, 3280 x 2464) at 25 m gives a 30.26 x 22.70 m
  footprint and 9.23 mm/px native GSD; a 640 x 640 (stretched) frame gives 4.73 x 3.55 cm/px and 16.8 cm2/px. Each
  overlay shows the GSD and per-class area in a header, and area (m2) and GLI on every patch. This is valid only if
  Roboflow resized whole frames; if it cropped or tiled them, set `IMAGES_ARE_FULL_FRAMES = False`. Area assumes flat
  ground, nadir view and a constant 25 m AGL: a 1 m height error is about 8 % in area.

## 9. Results of the first full run (Colab T4, 2026-11-01, seed 0)

Test split: 18 images (68 instances). Both models trained from scratch at 640 px, early stopping on valid with
patience 50 (Nano best epoch 173 of 223; YOLO26n-seg best epoch 245 of 295). YOLO's confidence threshold, 0.4, was
chosen on valid and sits at the top of the sweep.

| size | prec | model | params | ONNX KiB | CPU ms (1 thr) | mIoU crops | fenugreek | gram | linseed | potato | wheat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 | fp32 | Nano | 34,742 | 150 | 19.6 | 0.798 | 0.53 | 0.78 | 0.78 | 0.94 | 0.96 |
| 640 | fp32 | YOLO26n | 3,054,875 | 10,811 | 109.5 | 0.873 | 0.68 | 0.86 | 0.92 | 0.95 | 0.97 |
| 640 | int8 | Nano | | 125 | | 0.798 | 0.56 | 0.75 | 0.79 | 0.94 | 0.94 |
| 640 | int8 | YOLO26n | | 3,340 | | 0.870 | 0.68 | 0.86 | 0.92 | 0.95 | 0.95 |
| 480 | fp32 | Nano | | | 15.9 | 0.771 | 0.44 | 0.77 | 0.74 | 0.94 | 0.96 |
| 480 | fp32 | YOLO26n | | | 78.6 | 0.872 | 0.68 | 0.85 | 0.90 | 0.96 | 0.97 |
| 320 | fp32 | Nano | | | 7.3 | 0.740 | 0.41 | 0.73 | 0.69 | 0.93 | 0.94 |
| 320 | int8 | Nano | | | | 0.710 | 0.32 | 0.68 | 0.68 | 0.95 | 0.91 |
| 320 | fp32 | YOLO26n | | | 26.0 | 0.864 | 0.66 | 0.84 | 0.90 | 0.95 | 0.97 |
| 320 | int8 | YOLO26n | | | | 0.861 | 0.65 | 0.84 | 0.90 | 0.95 | 0.96 |

Paired bootstrap, Nano - YOLO (mIoU crops, 95 % CI): 640 fp32 -0.108 [-0.209, -0.030]; 640 int8 -0.117 [-0.218, -0.034];
480 fp32 -0.098 [-0.186, -0.028]; 320 fp32 -0.145 [-0.252, -0.051]; 320 int8 -0.148 [-0.265, -0.049].
INT8 latency on ORT-CPU is not meaningful (emulated) and is omitted.

**Reading.**
* The claim "smaller *and* better" is refuted on this data: YOLO26n-seg (scratch) is significantly better at every
  size and precision. The defensible claim is a trade-off: 88x fewer parameters, a 72x smaller FP32 file (27x for
  INT8) and 5.6x lower 1-thread CPU latency at 640 px, for -0.075 mIoU (-9 %).
* The gap sits almost entirely in the minority crops: fenugreek (-0.15), linseed (-0.14) and gram (-0.08). Wheat and
  potato are at parity (0.94 to 0.97). These are also the crops with the fewest images, so their IoUs are the least
  certain.
* Nano loses accuracy as input shrinks (0.798 -> 0.740 at 320 px); YOLO barely does (0.873 -> 0.864). Nano also loses
  3 points to INT8 at 320 px, while YOLO loses under 0.5 at every size.
* **On a CPU-class device YOLO26n at 320 px (0.864, 26 ms) beats Nano at 640 px (0.798, 19.6 ms)** for about 1.3x the
  latency. Nano wins only where weight or activation memory rules YOLO out (microcontrollers, a 10.8 MB FP32 / 3.3 MB INT8
  model), or where the segmentation must share a small CPU with other tasks.
* Limitations: single seed; 18 test images; validation has 1 image each of fenugreek and linseed; the YOLO threshold is
  at the edge of the sweep; random Roboflow split.

**Sequoia indices.** All 4 captures were registered by ECC (no fallback), within 1.65 px of the flight-median rig
transform. Median NDVI 0.55 to 0.66, median NDRE 0.03 to 0.08, NDRE > NDVI on at most 0.21 % of vegetation pixels.
These are identical to the values measured locally.

## 10. Per-field (instance) output for Nano

**Design.** Nano gets a field-boundary head: 2.3k parameters at 1/4 resolution, for 37,047 parameters total with 6
classes and 0.30 GMACs at 640 px. Its ONNX graph is still static; the only new op is one Concat. `instances.group_instances`
turns class probabilities + boundary probability into fields:
1. connected regions of each crop are candidate fields;
2. a region is split only when predicted boundaries cut it into >= 2 parts, each large enough to be a field
   (>= 10 % of the region and >= 0.2 % of the image), with boundary pixels reassigned by watershed;
3. the field's confidence is the mean probability of its class.

By definition this is **bottom-up (proposal-free) instance segmentation**: the network stays dense, the system outputs
instances. YOLO is top-down: it proposes objects, then masks them.

**What was tried and dropped.** A distance-to-field-edge output with distance-based seeding over-split fields. On
synthetic data the distance map followed crop texture rather than field shape, giving mask mAP50 0.40 vs 0.88 for
connected regions. Boundary-only grouping with the split rule above matches connected regions where fields do not touch,
and is the only variant that can separate touching fields of the same crop.

**Metric.** `instances.MaskAP` gives COCO-style mask AP (101-point, IoU 0.50:0.95). It matches pycocotools exactly on the
same predictions (mAP50 0.570, mAP75 0.496, mAP50-95 0.462). Ultralytics' own `val()` reported 0.498 / 0.449 / 0.376
on those predictions, because it scores masks at reduced resolution. Report the shared COCO-style number for both models
and Ultralytics' number only as a footnote.

**Synthetic check (256 px, not a real result).** Nano+boundary reached mIoU 0.897, mask mAP50 0.878, mAP50-95 0.615;
plain Nano with connected regions reached 0.868 / 0.833 / 0.578. INT8 cost nothing. Whether the boundary head separates
touching same-crop fields can only be measured on the Roboflow data.
