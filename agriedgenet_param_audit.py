"""
Exact analytical parameter + MAC counter for AgriEdgeNet.
Mirrors the PyTorch reference implementation module-for-module.
Conv layers use bias=False followed by BatchNorm (2 params per output channel).
Channel attention = ECA (Efficient Channel Attention): a single 1-D conv over the
channel descriptor, k params total, instead of SE's two FC layers (~100k each here).
"""

class Counter:
    def __init__(self):
        self.p = 0; self.m = 0; self.log = []

    def conv_bn(self, cin, cout, k, res_out, name, groups=1, bn=True):
        p = k*k*(cin//groups)*cout
        if bn: p += 2*cout
        macs = k*k*(cin//groups)*cout*res_out*res_out
        self.p += p; self.m += macs
        self.log.append((name, p, macs/1e6)); return p

    def linear(self, cin, cout, name):
        p = cin*cout + cout
        self.p += p; self.m += cin*cout
        self.log.append((name, p, cin*cout/1e6)); return p

    def eca(self, ch, name, k=5):
        # Conv1d(1,1,kernel=k,bias=False) applied to the C-length pooled descriptor
        p = k
        self.p += p; self.m += ch*k
        self.log.append((name, p, ch*k/1e6)); return p

    def mbconv(self, cin, cout, k, res_out, e, name, attn=False):
        hidden = cin*e
        if e != 1:
            self.conv_bn(cin, hidden, 1, res_out, name+".expand")
        self.conv_bn(hidden, hidden, k, res_out, name+".dw", groups=hidden)
        if attn:
            self.eca(hidden, name+".eca")
        self.conv_bn(hidden, cout, 1, res_out, name+".project")

    def dwsep(self, cin, cout, k, res_out, name):
        self.conv_bn(cin, cin, k, res_out, name+".dw", groups=cin)
        self.conv_bn(cin, cout, 1, res_out, name+".pw")


def build(cfg, R=512, K=6, verbose=False):
    S, C1, C2, C3, C4, D, n3, n4, e3, e4 = cfg
    c = Counter()
    r2, r4, r8, r16 = R//2, R//4, R//8, R//16

    c.conv_bn(4, S, 1, R, "spectral_stem")     # learned band mixing (index-like)
    c.conv_bn(S, C1, 3, r2, "stem1")           # 1/2
    c.conv_bn(C1, C2, 3, r4, "stem2")          # 1/4  -> decoder lateral
    c.mbconv(C2, C3, 3, r8, e3, "s3.0")        # 1/8
    for i in range(n3-1):
        c.mbconv(C3, C3, 3, r8, e3, f"s3.{i+1}", attn=True)
    c.mbconv(C3, C4, 5, r16, e4, "s4.0")       # 1/16
    for i in range(n4-1):
        c.mbconv(C4, C4, 5, r16, e4, f"s4.{i+1}", attn=True)
    c.conv_bn(C4, C4, 1, 1, "ctx.gap_proj")    # global context branch
    c.conv_bn(C4, C4, 3, r16, "ctx.dil_dw", groups=C4)
    c.conv_bn(C4, C4, 1, r16, "ctx.merge")
    c.conv_bn(C4, D, 1, r16, "dec.lat4")
    c.conv_bn(C3, D, 1, r8,  "dec.lat3")
    c.conv_bn(C2, D, 1, r4,  "dec.lat2")
    c.dwsep(D, D, 3, r8, "dec.fuse3")
    c.dwsep(D, D, 3, r4, "dec.fuse2")
    c.dwsep(D, D, 3, r4, "head.pre")
    c.conv_bn(D, K, 1, r4, "head.cls", bn=False); c.p += K  # conv bias
    c.linear(C4, 32, "aux.fc1")
    c.linear(32, 1, "aux.fc2")

    if verbose:
        print(f"{'layer':<22}{'params':>12}{'MMACs':>12}")
        for n,p,m in c.log:
            print(f"{n:<22}{p:>12,}{m:>12.1f}")
    return c

#         S   C1  C2   C3   C4   D  n3 n4 e3 e4
configs = {
  "AgriEdgeNet-S": (12, 24, 32,  64, 112, 48, 1, 2, 3, 3),
  "AgriEdgeNet-M": (16, 32, 48, 112, 192, 72, 2, 3, 4, 3),
}

print(f"{'variant':<16}{'params':>11}{'MB fp16':>9}{'GMACs@512':>11}{'GMACs@384':>11}{'GMACs@320':>11}")
for name, cfg in configs.items():
    a = build(cfg, R=512); b = build(cfg, R=384); d = build(cfg, R=320)
    flag = "  <-- OVER 1M" if a.p > 1_000_000 else ""
    print(f"{name:<16}{a.p:>11,}{a.p*2/1e6:>9.2f}{a.m/1e9:>11.2f}{b.m/1e9:>11.2f}{d.m/1e9:>11.2f}{flag}")

print("\n=== AgriEdgeNet-M layer breakdown @512x512, K=6 ===")
m = build(configs["AgriEdgeNet-M"], R=512, verbose=True)
print(f"{'TOTAL':<22}{m.p:>12,}{m.m/1e6:>12.1f}")
