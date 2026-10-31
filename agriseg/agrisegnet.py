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
    from torch.utils.flop_counter import FlopCounterMode
    model = model.eval().to(device)
    x = torch.zeros(1, 3, *imgsz, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9   # FLOPs -> MACs -> G


# ----------------------------------------------------------------------------- sub-50k variant
class DilatedDW(nn.Module):
    """Residual depthwise-separable block with a dilated 3x3 depthwise conv: grows the receptive
    field at 1/16 for ~2.5k parameters instead of adding a 1/32 stage (~12k)."""

    def __init__(self, ch, dilation):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, dilation, dilation=dilation, groups=ch, bias=False),
                                nn.BatchNorm2d(ch), nn.ReLU6(inplace=True))
        self.pw = ConvBNAct(ch, ch, 1, 1, act=False)

    def forward(self, x):
        return x + self.pw(self.dw(x))


class AgriSegNetNano(nn.Module):
    """< 50k-parameter semantic segmentation network (crop-type mapping from UAV RGB).

    Budget logic: crop-type regions are large and colour/texture driven, so capacity goes to receptive
    field (dilated 1/16 stage + global-context gate) rather than to channels; prediction at 1/8.
    Head is LR-ASPP style (MobileNetV3): global average pool -> sigmoid gate on 1/16 features, fused
    with a 1/8 skip. GlobalAveragePool exports cleanly to ONNX at any input size divisible by 16.
    ReLU6 / BN / depthwise only -> INT8- and NPU-friendly.
    """

    def __init__(self, num_classes: int, width: float = 1.0, aux: bool = True, drop: float = 0.05):
        super().__init__()
        self.num_classes, self.use_aux = num_classes, aux
        c0, c1, c2, c3 = (_c(c, width) for c in (16, 24, 32, 48))
        self.stem = ConvBNAct(3, c0, 3, 2)                                            # 1/2
        self.down4 = nn.Sequential(ConvBNAct(c0, c0, 3, 2, groups=c0), ConvBNAct(c0, c1, 1, 1))  # 1/4
        self.stage4 = InvertedResidual(c1, c1, 1, 2, 3)
        self.stage8 = nn.Sequential(InvertedResidual(c1, c2, 2, 2, 3),                 # 1/8
                                    InvertedResidual(c2, c2, 1, 2, 3))
        self.stage16 = nn.Sequential(InvertedResidual(c2, c3, 2, 2, 3),                # 1/16
                                     DilatedDW(c3, 2), DilatedDW(c3, 4), DilatedDW(c3, 8))
        self.ctx_feat = ConvBNAct(c3, c2, 1, 1)
        self.ctx_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c3, c2, 1), nn.Sigmoid())
        self.low = ConvBNAct(c2, c2, 1, 1, act=False)
        self.refine = DWSep(c2, c2)
        self.drop = nn.Dropout2d(drop)
        self.head = nn.Conv2d(c2, num_classes, 1)
        self.aux_head = nn.Conv2d(c3, num_classes, 1) if aux else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        f8 = self.stage8(self.stage4(self.down4(self.stem(x))))
        f16 = self.stage16(f8)
        ctx = self.ctx_feat(f16) * self.ctx_gate(f16)
        ctx = F.interpolate(ctx, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(self.drop(self.refine(ctx + self.low(f8))))
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        if self.training and self.use_aux:
            aux = F.interpolate(self.aux_head(f16), size=(H, W), mode="bilinear", align_corners=False)
            return {"out": out, "aux": aux}
        return out


ARCHS = {"agrisegnet": AgriSegNet, "nano": AgriSegNetNano}
PARAM_BUDGET = {"agrisegnet": 1_000_000, "nano": 50_000}


def build_model(arch: str, num_classes: int, width: float = 1.0, depth: int = 1, aux: bool = True) -> nn.Module:
    if arch == "nano":
        return AgriSegNetNano(num_classes, width=width, aux=aux)
    return AgriSegNet(num_classes, width=width, depth=depth, aux=aux)


if __name__ == "__main__":
    for arch, w, d in [("nano", 0.75, 1), ("nano", 1.0, 1), ("nano", 1.25, 1),
                       ("agrisegnet", 0.25, 1), ("agrisegnet", 0.5, 1), ("agrisegnet", 1.0, 1)]:
        m = build_model(arch, num_classes=6, width=w, depth=d)
        n = count_params(m)
        print(f"{arch:<10s} width={w:<4} depth={d}  params={n:>9,d}  "
              f"{'<50k' if n < 50_000 else '    '}  GMACs@640={count_flops(m, (640, 640)):6.3f}")
