"""AgriSegNet: lightweight semantic segmentation network for UAV crop mapping.

Design rationale (see README):
  * crop-type regions are large and boundaries are coarse -> spend capacity in the
    encoder / receptive field, predict at 1/8 resolution, bilinear-upsample.
  * ReLU6 only, no SE / hard-swish -> stable INT8 and Edge-TPU / TensorRT friendly.
  * depthwise-separable blocks everywhere except the first fused block.

Width (`width`) and depth (`depth`) multipliers let you sweep the Pareto front the way
the hogweed U-Net paper did with W and D.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _c(ch: int, width: float, divisor: int = 8) -> int:
    """Scale channels, keep them a multiple of `divisor` (tensor-core / NPU friendly)."""
    v = max(divisor, int(ch * width + divisor / 2) // divisor * divisor)
    if v < 0.9 * ch * width:
        v += divisor
    return v


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        pad = k // 2
        layers: List[nn.Module] = [
            nn.Conv2d(cin, cout, k, s, pad, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class FusedMBConv(nn.Module):
    """3x3 conv expand -> 1x1 linear project. Used at high resolution where depthwise
    convs are memory-bound and slow on GPUs."""

    def __init__(self, cin, cout, s=1, expand=2):
        super().__init__()
        mid = cin * expand
        self.use_res = s == 1 and cin == cout
        self.block = nn.Sequential(
            ConvBNAct(cin, mid, 3, s),
            ConvBNAct(mid, cout, 1, 1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, s=1, expand=3, k=3):
        super().__init__()
        mid = cin * expand
        self.use_res = s == 1 and cin == cout
        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(cin, mid, 1, 1))
        layers += [
            ConvBNAct(mid, mid, k, s, groups=mid),
            ConvBNAct(mid, cout, 1, 1, act=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class DWSep(nn.Sequential):
    """depthwise 3x3 + pointwise 1x1 (both with BN, ReLU6 after the pointwise)."""

    def __init__(self, cin, cout):
        super().__init__(
            ConvBNAct(cin, cin, 3, 1, groups=cin),
            ConvBNAct(cin, cout, 1, 1),
        )


class ContextModule(nn.Module):
    """Tiny pyramid pooling: injects global / regional context (several crop rows to a
    whole plot) at 1/32 resolution where it is almost free."""

    def __init__(self, cin, cout, bins: Sequence[int] = (1, 2, 4)):
        super().__init__()
        mid = max(8, cout // 4)
        self.bins = tuple(bins)
        self.reduce = nn.ModuleList([ConvBNAct(cin, mid, 1, 1) for _ in bins])
        self.fuse = ConvBNAct(cin + mid * len(bins), cout, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for b, red in zip(self.bins, self.reduce):
            p = F.adaptive_avg_pool2d(x, b)
            outs.append(F.interpolate(red(p), size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(outs, dim=1))


class AgriSegNet(nn.Module):
    """
    Input : (B, 3, H, W), H and W divisible by 32; use multiples of 128 for a clean ONNX/TensorRT graph (the eval code pads for you)
    Output: (B, num_classes, H, W) logits.  In training mode with aux=True returns a dict
            {"out": ..., "aux": ...}; the aux head is dropped at inference/export.
    """

    def __init__(self, num_classes: int, width: float = 1.0, depth: int = 1,
                 decoder_ch: int = 64, aux: bool = True, drop: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = aux
        c0, c1, c2, c3, c4 = (_c(c, width) for c in (16, 24, 48, 96, 160))
        d = _c(decoder_ch, width)

        self.stem = ConvBNAct(3, c0, 3, 2)                                   # 1/2
        s1 = [FusedMBConv(c0, c1, s=2, expand=2)]                            # 1/4
        s1 += [FusedMBConv(c1, c1, s=1, expand=2) for _ in range(depth - 1)]
        self.stage1 = nn.Sequential(*s1)
        self.stage2 = nn.Sequential(                                         # 1/8
            InvertedResidual(c1, c2, 2, 3, 3),
            *[InvertedResidual(c2, c2, 1, 3, 3) for _ in range(depth)])
        self.stage3 = nn.Sequential(                                         # 1/16
            InvertedResidual(c2, c3, 2, 3, 5),
            *[InvertedResidual(c3, c3, 1, 3, 5) for _ in range(depth + 1)])
        self.stage4 = nn.Sequential(                                         # 1/32
            InvertedResidual(c3, c4, 2, 3, 5),
            *[InvertedResidual(c4, c4, 1, 3, 5) for _ in range(depth)])

        self.context = ContextModule(c4, c3)
        self.lat16 = ConvBNAct(c3, d, 1, 1, act=False)
        self.lat8 = ConvBNAct(c2, d, 1, 1, act=False)
        self.ctx_proj = ConvBNAct(c3, d, 1, 1, act=False)
        self.dec16 = DWSep(d, d)
        self.dec8 = DWSep(d, d)
        self.drop = nn.Dropout2d(drop)
        self.head = nn.Conv2d(d, num_classes, 1)
        self.aux_head = nn.Sequential(ConvBNAct(c3, d, 1, 1), nn.Conv2d(d, num_classes, 1)) if aux else None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        H, W = x.shape[-2:]
        x = self.stem(x)
        x = self.stage1(x)
        f8 = self.stage2(x)
        f16 = self.stage3(f8)
        f32 = self.stage4(f16)

        ctx = self.ctx_proj(self.context(f32))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d16 = self.dec16(up(ctx, f16) + self.lat16(f16))
        d8 = self.dec8(up(d16, f8) + self.lat8(f8))
        out = self.head(self.drop(d8))
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        if self.training and self.use_aux:
            aux = F.interpolate(self.aux_head(f16), size=(H, W), mode="bilinear", align_corners=False)
            return {"out": out, "aux": aux}
        return out


def count_params(model: nn.Module, include_aux: bool = False) -> int:
    n = 0
    for name, p in model.named_parameters():
        if not include_aux and name.startswith("aux_head"):
            continue
        if p.requires_grad:
            n += p.numel()
    return n


def count_flops(model: nn.Module, imgsz=(512, 512), device="cpu") -> float:
    """Returns GMACs for one forward pass (aux head excluded because eval() drops it)."""
    import copy
    from torch.utils.flop_counter import FlopCounterMode
    model = copy.deepcopy(model).eval().to(device)   # copy: never move the caller's model off its device
    x = torch.zeros(1, 3, *imgsz, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9   # FLOPs -> MACs -> G


# ----------------------------------------------------------------------------- sub-50k variant
class DWBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, residual when shapes match. Stride 2 downsamples."""

    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.use_res = s == 1 and cin == cout
        self.dw = ConvBNAct(cin, cin, 3, s, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1, 1, act=not self.use_res)

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.use_res else y


class AgriSegNetNano(nn.Module):
    """Device-agnostic < 50k-parameter semantic segmentation network (crop-type mapping, UAV RGB).

    Operator set (after BN folding): Conv (dense + depthwise, no dilation), Add, Mul, Clip (ReLU6),
    GlobalAveragePool, Sigmoid, Resize (bilinear, integer factors). These map to TensorRT, ONNX Runtime,
    TFLite / TFLite-Micro, OpenVINO and most NPUs; nothing dynamic, no dilation, no attention.

    Capacity goes to receptive field (stages down to 1/32 + a global-context gate) rather than channels:
    crop-type regions are large and colour/texture driven. Prediction at 1/8.

    `output_stride=8` in forward() skips the final upsample: the deployed graph then emits logits at 1/8
    resolution (argmax there, nearest-upsample the label map on the host). At 640 px this removes the
    largest activation in the network (640*640*num_classes).
    """

    def __init__(self, num_classes: int, width: float = 1.0, aux: bool = True, drop: float = 0.05):
        super().__init__()
        self.num_classes, self.use_aux = num_classes, aux
        self.output_stride = 1
        c0, c1, c2, c3 = (_c(c, width) for c in (16, 24, 32, 48))
        self.stem = ConvBNAct(3, c0, 3, 2)                                    # 1/2
        self.stage4 = nn.Sequential(DWBlock(c0, c1, 2), InvertedResidual(c1, c1, 1, 2, 3))   # 1/4
        self.stage8 = nn.Sequential(InvertedResidual(c1, c2, 2, 2, 3),         # 1/8
                                    InvertedResidual(c2, c2, 1, 2, 3))
        self.stage16 = nn.Sequential(InvertedResidual(c2, c3, 2, 2, 3), DWBlock(c3, c3))   # 1/16
        self.stage32 = nn.Sequential(DWBlock(c3, c3, 2), DWBlock(c3, c3))      # 1/32
        self.ctx16 = ConvBNAct(c3, c2, 1, 1)
        self.ctx32 = ConvBNAct(c3, c2, 1, 1, act=False)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c3, c2, 1), nn.Sigmoid())
        self.low = ConvBNAct(c2, c2, 1, 1, act=False)
        self.refine = DWBlock(c2, c2)
        self.drop = nn.Dropout2d(drop)
        self.head = nn.Conv2d(c2, num_classes, 1)
        self.aux_head = nn.Conv2d(c3, num_classes, 1) if aux else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # input H, W must be divisible by 32; fixed integer scale factors keep the exported graph static
        # (no Shape/Slice/Concat ops), which microcontroller and NPU compilers require
        f8 = self.stage8(self.stage4(self.stem(x)))
        f16 = self.stage16(f8)
        f32 = self.stage32(f16)
        up = lambda t, k: F.interpolate(t, scale_factor=k, mode="bilinear", align_corners=False)
        ctx = self.ctx16(f16) * self.gate(f32) + up(self.ctx32(f32), 2)
        out = self.head(self.drop(self.refine(up(ctx, 2) + self.low(f8))))
        if self.output_stride == 1:
            out = up(out, 8)
        if self.training and self.use_aux:
            return {"out": out, "aux": up(self.aux_head(f16), 16)}
        return out


def peak_activation_bytes(model: nn.Module, imgsz=(640, 640), bytes_per_value=1) -> int:
    """Largest (input + output) of any conv, plus the final upsample if present. BN and activations are
    folded or in-place at deployment, so they are not counted. A lower bound on the runtime arena size."""
    peak = [0]

    def hook(_m, inp, out):
        peak[0] = max(peak[0], out.numel() + inp[0].numel())

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        out = model.eval()(torch.zeros(1, 3, *imgsz))
    for h in hs:
        h.remove()
    if getattr(model, "output_stride", 1) == 1:      # final bilinear upsample: 1/8 logits -> full res
        peak[0] = max(peak[0], out.numel() + out.numel() // 64)
    return peak[0] * bytes_per_value


ARCHS = {"agrisegnet": AgriSegNet, "nano": AgriSegNetNano}
PARAM_BUDGET = {"agrisegnet": 1_000_000, "nano": 50_000}


def build_model(arch: str, num_classes: int, width: float = 1.0, depth: int = 1, aux: bool = True) -> nn.Module:
    if arch == "nano":
        return AgriSegNetNano(num_classes, width=width, aux=aux)
    return AgriSegNet(num_classes, width=width, depth=depth, aux=aux)


if __name__ == "__main__":
    print(f"{'arch':<11}{'width':>6}{'params':>10}{'GMACs@640':>11}   peak INT8 activations (MiB) @256 / 320 / 640")
    for arch, w in [("nano", 0.5), ("nano", 0.75), ("nano", 1.0), ("agrisegnet", 0.25), ("agrisegnet", 1.0)]:
        m = build_model(arch, num_classes=6, width=w, aux=False)
        acts = [peak_activation_bytes(m, (s, s)) / 2 ** 20 for s in (256, 320, 640)]
        line = f"{arch:<11}{w:>6}{count_params(m):>10,d}{count_flops(m, (640, 640)):>11.3f}   " + " / ".join(f"{a:.2f}" for a in acts)
        if arch == "nano":
            m.output_stride = 8
            line += "   | stride-8 output @640: %.2f" % (peak_activation_bytes(m, (640, 640)) / 2 ** 20)
        print(line)
