# Usage: python tools/draw_architectures.py docs/figures
"""Architecture diagrams for AgriSegNet-Nano (+ boundary head) and YOLO26n-seg, 640x640 input, 5 crops.
All shapes and parameter counts measured from the models (see session); drawn with matplotlib."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DK, TXT, MUT = "#1F3A2E", "#1C2B22", "#5E6E64"
NANO, NANO_L = "#D99A1E", "#FBF1DC"
YOLO, YOLO_L = "#4E6A7E", "#E6EDF2"
GRN, GRN_L = "#2F6B4A", "#E6F0E8"
GREY_L = "#F1F3F0"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def fig_canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(0.35, 8.55, title, fontsize=22, fontweight="bold", color=DK, family="DejaVu Serif", va="center")
    ax.text(0.35, 8.08, subtitle, fontsize=12.5, color=MUT, va="center")
    return fig, ax


def box(ax, x, y, w, h, title, lines, edge, fill, tsize=12, lsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6,
                                edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=TXT)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.3, l, ha="center", va="center", fontsize=lsize, color=TXT)
    return (x, y, w, h)


def arrow(ax, p, q, color=MUT, rad=0.0, label=None, lpos=0.5, ls="-"):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=color,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        lx, ly = p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos
        if abs(q[0] - p[0]) < 0.05:          # vertical arrow: label to the right of the line
            ax.text(lx + 0.1, ly, label, fontsize=9.5, color=color, ha="left", va="center", fontweight="bold")
        else:
            ax.text(lx, ly + 0.14, label, fontsize=9.5, color=color, ha="center", va="bottom", fontweight="bold")


R = lambda b: (b[0] + b[2], b[1] + b[3] / 2)
L = lambda b: (b[0], b[1] + b[3] / 2)
T = lambda b, f=0.5: (b[0] + b[2] * f, b[1] + b[3])
B = lambda b, f=0.5: (b[0] + b[2] * f, b[1])


def band_label(ax, y, text, color):
    ax.text(0.35, y, text, fontsize=11, fontweight="bold", color=color, rotation=90, va="center", ha="center")


# ======================================================================= NANO
fig, ax = fig_canvas("AgriSegNet-Nano with field-boundary head",
                     "37,047 parameters  |  0.30 GMACs @ 640x640  |  bottom-up instance segmentation  |  "
                     "ops: Conv, depthwise Conv, Add, Mul, Clip (ReLU6), GlobalAvgPool, Sigmoid, Resize, Concat")
w, h, y0 = 2.2, 1.55, 6.05
band_label(ax, y0 + h / 2, "ENCODER", NANO)
xs = [0.7, 3.15, 5.6, 8.05, 10.5, 12.95]
inp = box(ax, xs[0], y0, w, h, "Input", ["RGB 3 x 640 x 640", "resize long side, pad /32"], DK, GREY_L)
stem = box(ax, xs[1], y0, w, h, "Stem  1/2", ["Conv 3x3 s2, BN, ReLU6", "16 x 320 x 320", "464 params"], NANO, NANO_L)
s4 = box(ax, xs[2], y0, w, h, "Stage 1/4", ["DW-sep s2 + InvRes(e=2)", "24 x 160 x 160", "3,584 params"], NANO, NANO_L)
s8 = box(ax, xs[3], y0, w, h, "Stage 1/8", ["InvRes s2 + InvRes", "32 x 80 x 80", "8,368 params"], NANO, NANO_L)
s16 = box(ax, xs[4], y0, w, h, "Stage 1/16", ["InvRes s2 + DW block", "48 x 40 x 40", "8,976 params"], NANO, NANO_L)
s32 = box(ax, xs[5], y0, w, h, "Stage 1/32", ["DW-sep s2 + DW block", "48 x 20 x 20", "5,856 params"], NANO, NANO_L)
for a, b in [(inp, stem), (stem, s4), (s4, s8), (s8, s16), (s16, s32)]:
    arrow(ax, R(a), L(b))

y1 = 3.75
band_label(ax, y1 + h / 2, "CONTEXT + DECODER", GRN)
gate = box(ax, 12.95, y1, w, h, "Global context", ["GAP -> 1x1 -> sigmoid gate", "1x1 conv on f32 (32 ch)", "3,168 params"], GRN, GRN_L)
fuse = box(ax, 10.5, y1, w, h, "Context fusion 1/16", ["1x1(f16) x gate", "+ upsample x2 (f32)", "32 x 40 x 40  |  1,600"], GRN, GRN_L)
dec = box(ax, 8.05, y1, w, h, "Decoder 1/8", ["up x2 + 1x1 skip (f8)", "DW block -> 1x1 head", "6 x 80 x 80  |  2,726"], GRN, GRN_L)
arrow(ax, B(s32), T(gate), label="f32")
arrow(ax, B(s16), T(fuse), label="f16")
arrow(ax, L(gate), R(fuse))
arrow(ax, L(fuse), R(dec))
arrow(ax, B(s8), T(dec), label="f8")

y2 = 1.25
band_label(ax, y2 + h / 2, "OUTPUTS", DK)
bnd = box(ax, 5.6, y2, w, h, "Boundary head 1/4", ["up x2 (decoder) + 1x1 (f4)", "DW block -> 1x1", "1 x 160 x 160  |  2,305"], NANO, "#FFFFFF")
cls = box(ax, 8.05, y2, w, h, "Class logits", ["upsample x8 (bilinear)", "6 x 640 x 640", "background + 5 crops"], GRN, "#FFFFFF")
outp = box(ax, 3.15, y2, w, h, "Network output", ["concat: 7 x 640 x 640", "6 class + 1 boundary", "(one static ONNX graph)"], DK, GREY_L)
post = box(ax, 0.7, y2, w, h, "Grouping (CPU)", ["argmax -> crop regions", "split by boundaries", "-> fields: class, conf, mask"], DK, GREY_L)
arrow(ax, B(dec), T(cls))
arrow(ax, B(s4), T(bnd), label="f4")
arrow(ax, (dec[0], dec[1] + 0.25), (bnd[0] + bnd[2], bnd[1] + bnd[3] - 0.15), rad=-0.15, label="d8")
arrow(ax, L(bnd), R(outp))
arrow(ax, B(cls, 0.3), (outp[0] + outp[2] * 0.7, outp[1]), rad=-0.35)
arrow(ax, L(outp), R(post))

ax.add_patch(FancyBboxPatch((10.5, 0.55), 4.65, 2.35, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.0,
                            edgecolor="#C9D2C5", facecolor="#FAFBF9"))
rows = [("Parameters by part", ""), ("Encoder", "27,248"), ("Context", "4,768"), ("Decoder + class head", "2,726"),
        ("Boundary head", "2,305"), ("Total", "37,047")]
for i, (k, v) in enumerate(rows):
    bold = i in (0, len(rows) - 1)
    ax.text(10.75, 2.6 - i * 0.36, k, fontsize=11, color=TXT, fontweight="bold" if bold else "normal", va="center")
    ax.text(14.95, 2.6 - i * 0.36, v, fontsize=11, color=TXT, fontweight="bold" if bold else "normal", va="center", ha="right")
ax.text(0.35, 0.3, "InvRes = inverted residual (1x1 expand x2, depthwise 3x3, 1x1 project); DW block = depthwise 3x3 + pointwise 1x1, "
        "residual; DW-sep s2 = depthwise 3x3 stride 2 + pointwise. BN is folded into the convolutions at export.",
        fontsize=9.5, color=MUT, va="center")
fig.savefig(f"{OUT}/nano_architecture.png"); fig.savefig(f"{OUT}/nano_architecture.svg"); plt.close(fig)

# ======================================================================= YOLO26n-seg
fig, ax = fig_canvas("YOLO26n-seg (baseline)",
                     "3,054,875 parameters with 5 classes (2.69M fused for inference)  |  9.1 GFLOPs (fused) @ 640x640  |  "
                     "top-down instance segmentation, NMS-free (end-to-end) head")
band_label(ax, y0 + h / 2, "BACKBONE", YOLO)
inp = box(ax, xs[0], y0, w, h, "Input", ["RGB 3 x 640 x 640", "letterbox"], DK, GREY_L)
b1 = box(ax, xs[1], y0, w, h, "1/2 , 1/4", ["Conv s2 (16) -> Conv s2 (32)", "-> C3k2 x1 (64)", "11,776 params"], YOLO, YOLO_L)
b3 = box(ax, xs[2], y0, w, h, "P3  1/8", ["Conv s2 -> C3k2", "128 x 80 x 80", "63,072 params"], YOLO, YOLO_L)
b4 = box(ax, xs[3], y0, w, h, "P4  1/16", ["Conv s2 -> C3k2", "128 x 40 x 40", "234,752 params"], YOLO, YOLO_L)
b5 = box(ax, xs[4], y0, w, h, "1/32", ["Conv s2 -> C3k2", "256 x 20 x 20", "641,536 params"], YOLO, YOLO_L)
b6 = box(ax, xs[5], y0, w, h, "P5  1/32", ["SPPF (pooling pyramid)", "C2PSA (self-attention)", "414,336 params"], YOLO, YOLO_L)
for a, b in [(inp, b1), (b1, b3), (b3, b4), (b4, b5), (b5, b6)]:
    arrow(ax, R(a), L(b))

band_label(ax, y1 + h / 2, "NECK (FPN + PAN)", GRN)
td4 = box(ax, 10.5, y1, w, h, "Top-down 1/16", ["up x2 (P5) + concat P4", "C3k2 -> 128 x 40 x 40", "119,808 params"], GRN, GRN_L)
n3 = box(ax, 8.05, y1, w, h, "N3  1/8", ["up x2 + concat P3", "C3k2 -> 64 x 80 x 80", "34,304 params"], GRN, GRN_L)
n4 = box(ax, 5.6, y1, w, h, "N4  1/16", ["Conv s2 + concat", "C3k2 -> 128 x 40 x 40", "132,224 params"], GRN, GRN_L)
n5 = box(ax, 3.15, y1, w, h, "N5  1/32", ["Conv s2 + concat P5", "C3k2 -> 256 x 20 x 20", "610,816 params"], GRN, GRN_L)
arrow(ax, B(b6), T(td4), label="P5")
arrow(ax, B(b4), (td4[0] + 0.25, td4[1] + td4[3]), rad=0.2, label="P4", lpos=0.35)
arrow(ax, L(td4), R(n3))
arrow(ax, B(b3), T(n3), label="P3", lpos=0.25)
arrow(ax, L(n3), R(n4))
arrow(ax, L(n4), R(n5))

band_label(ax, y2 + h / 2, "HEAD (Segment26)", DK)
det = box(ax, 3.15, y2, 4.65, h, "Detection branches on N3, N4, N5", ["8,400 anchors (6,400 + 1,600 + 400)",
          "per anchor: box (4) + class scores (5) + mask coefficients (32)",
          "one-to-one head: no NMS (one-to-many branch: training only)"], YOLO, "#FFFFFF", lsize=10)
proto = box(ax, 8.05, y2, w, h, "Proto26 (from N3)", ["32 mask prototypes", "32 x 160 x 160"], YOLO, "#FFFFFF")
inst = box(ax, 0.7, y2, w, h, "Instances", ["top-k by score", "mask = sigmoid(coef . protos)", "crop to box, upsample"], DK, GREY_L)
for b in (n3, n4, n5):
    arrow(ax, B(b), (min(max(b[0] + b[2] / 2, det[0] + 0.3), det[0] + det[2] - 0.3), det[1] + det[3]))
arrow(ax, (n3[0] + n3[2] * 0.8, n3[1]), T(proto, 0.4))
arrow(ax, L(det), R(inst))
arrow(ax, B(proto, 0.5), (inst[0] + inst[2] * 0.6, inst[1]), rad=-0.22)

ax.add_patch(FancyBboxPatch((10.5, 0.55), 4.65, 2.35, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.0,
                            edgecolor="#C9D2C5", facecolor="#FAFBF9"))
rows = [("Parameters by part (5 classes)", ""), ("Backbone", "1,365,472"), ("Neck", "897,152"),
        ("Head (incl. training-only branch)", "792,251"), ("Total", "3,054,875")]
for i, (k, v) in enumerate(rows):
    bold = i in (0, len(rows) - 1)
    ax.text(10.75, 2.6 - i * 0.42, k, fontsize=11, color=TXT, fontweight="bold" if bold else "normal", va="center")
    ax.text(14.95, 2.6 - i * 0.42, v, fontsize=11, color=TXT, fontweight="bold" if bold else "normal", va="center", ha="right")
ax.text(0.35, 0.3, "C3k2 = CSP block with two-conv bottlenecks; SPPF = spatial pyramid pooling (fast); C2PSA = CSP block with "
        "position-sensitive self-attention. Channel counts are the n-scale values (width 0.25).",
        fontsize=9.5, color=MUT, va="center")
fig.savefig(f"{OUT}/yolo26n_seg_architecture.png"); fig.savefig(f"{OUT}/yolo26n_seg_architecture.svg"); plt.close(fig)
print("ok")
