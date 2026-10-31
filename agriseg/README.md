# AgriSegNet – lightweight crop semantic segmentation vs YOLO26

Input: your Roboflow export (YOLO segmentation format: `data.yaml`, `train|valid|test/{images,labels}`).
Polygons are rasterised to semantic masks (0 = background, class_id+1). Directory layout is used,
not the paths inside `data.yaml`, so Roboflow's `../train/images` quirk does not matter here.

## Install
    pip install torch opencv-python-headless pyyaml numpy          # custom model
    pip install ultralytics                                        # only for the YOLO26 comparison
    pip install onnx onnxscript onnxruntime                        # only for export_onnx.py

## 1. Parameter / compute check (no data needed)
    python agrisegnet.py
Prints params and GMACs@640 for both architectures (6 classes):

| arch | width | params | GMACs @640 | peak INT8 activations @256 / 320 / 640 px |
|---|---|---|---|---|
| nano (`AgriSegNetNano`) | 0.5 | 11,390 | 0.10 | 0.38 / 0.60 / 2.38 MiB |
| nano | 1.0 | 34,742 | 0.25 | 0.44 / 0.68 / 2.73 MiB |
| agrisegnet | 0.25 | 43,894 | 0.12 | 0.38 / 0.60 / 2.38 MiB |
| agrisegnet | 1.0 | 491,102 | 0.83 | 0.44 / 0.68 / 2.73 MiB |

Nano is device-agnostic: its exported graph is static and uses only Conv (dense/depthwise, no dilation), Add, Mul,
Clip, GlobalAveragePool, Sigmoid and Resize (fixed x2/x8). Input sides must be multiples of 32.

## 1b. Deployment check (ONNX FP32 + INT8, accuracy of the exported graphs)
    python onnx_tools.py --ckpt runs/nano_w1_s0/best.pt --data /path/to/export --imgsz 640 --out deploy
    python onnx_tools.py --ckpt runs/nano_w1_s0/best.pt --data /path/to/export --imgsz 320 --out deploy/mcu_320
Writes FP32 and QDQ-INT8 ONNX (TensorRT / ONNX Runtime / OpenVINO) and their test mIoU. For microcontrollers,
convert the FP32 ONNX to TFLite INT8 (onnx2tf) with the same calibration images; not automated here.

`--arch nano` enforces a 50k budget, `--arch agrisegnet` 1M (override with `--max-params`).
At 640 px peak activations (~3 MiB INT8) dwarf the weights (32 KiB for nano): parameters are not the
memory budget on small devices. See `docs/REVIEW.md` section 4 before claiming anything about the sub-50k model.

## 2. Train (use the SAME --imgsz you trained YOLO26 with; use a multiple of 128: 512, 640, 768, 1024)
    python train.py --data /path/to/export --imgsz 640 --epochs 150 --bs 16 --width 1.0 --depth 1 --seed 0 --out runs/agri_w1d1_s0
    python train.py --data /path/to/export --imgsz 640 --epochs 150 --bs 16 --arch nano --width 1.0 --seed 0 --out runs/nano_w1_s0
Repeat for seeds 0,1,2, and for the width/depth sweep (width 0.5/0.75/1.0/1.5, depth 1/2).
Training aborts above the parameter budget. Checkpoints hold the EMA weights and record `arch`.
Split your data by field or date first: random splits of overlapping UAV frames leak (run the leakage check in
`notebooks/01_rgb_segmentation_gli.ipynb`).

## 3. Compare against YOLO26 (tune on valid, report on test)
YOLO26-seg is an instance model; its masks are converted to one semantic map and the result depends on
the confidence threshold. Choose that threshold on the **validation** split, then report **test** once:

    # a) pick YOLO's threshold on valid
    python evaluate.py --data /path/to/export --split valid --imgsz 640 --yolo yolo26n_seg.pt \
        --yolo-conf 0.05 0.1 0.25 0.4 --no-bench --viz 0 --out results/yolo_conf_sweep
    # b) final comparison on test, single threshold for YOLO, 3 seeds for AgriSegNet
    python evaluate.py --data /path/to/export --split test --imgsz 640 \
        --custom runs/agri_w1d1_s0/best.pt runs/agri_w1d1_s1/best.pt runs/agri_w1d1_s2/best.pt \
        --yolo yolo26n_seg.pt yolo26s_seg.pt --yolo-conf 0.25 --out results/compare

Outputs `results.json`, a printed table (params, GFLOPs, mIoU on crop classes, mIoU incl. background,
pixel accuracy, latency), per-class IoU, mean +/- std over AgriSegNet seeds, a paired-bootstrap CI for
(AgriSegNet - YOLO) over test images, and side-by-side montages `viz_*.jpg`
(image | ground truth | each model, in the order listed).

## 4. Edge benchmark (the comparison that matters for the paper)
    python export_onnx.py --ckpt runs/agri_w1d1_s0/best.pt --imgsz 640
    trtexec --onnx=agrisegnet.onnx --fp16 --saveEngine=agri_fp16.engine --warmUp=500 --iterations=200
    yolo export model=yolo26n-seg.pt format=engine half=True imgsz=640
Run both engines in the same loop on the Jetson, log power with `tegrastats --interval 200`, report J/frame.
Also build INT8 engines (calibrate on ~500 training images) for both and report the mIoU drop.

## Caveats
* mIoU(crops) excludes background and is the headline metric; mIoU(all) is reported too.
* Latency from `evaluate.py` is PyTorch / Ultralytics wall-clock with different pre/post-processing for the
  two models - use TensorRT numbers for any claim. fp16 latency is only measured on CUDA.
* GFLOPs: AgriSegNet = 2 x MACs at a square imgsz x imgsz input (non-square frames cost proportionally less);
  YOLO = Ultralytics' own count on the fused model at the same imgsz.
* ONNX export works at any size but yields a clean AveragePool graph only when imgsz is a multiple of 128
  (the exporter rounds up and tells you). Other sizes lower to Gather/Where ops that TensorRT/Edge-TPU dislike.
* Overlapping polygons in the ground truth are resolved by file order (later wins).
* No pretraining / distillation yet: the model trains from scratch. Add those only if the from-scratch
  baseline shows a gap to YOLO26 that more epochs and augmentation do not close.
