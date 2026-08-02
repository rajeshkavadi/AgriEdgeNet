"""
Operating-envelope analysis for the ACTUAL hardware:
Parrot Sequoia multispectral head + NVIDIA Jetson Nano / Orin Nano.

Sequoia multispectral optics (per band, 4 separate global-shutter monochrome cameras):
  resolution 1280 x 960 px, focal length 3.98 mm, pixel pitch 3.75 um
  bands: Green 550/40, Red 660/40, RedEdge 735/10, NIR 790/40
  max capture rate ~1 Hz (all four bands)
"""
PX, PY = 1280, 960
PITCH = 3.75e-6
FOCAL = 3.98e-3
SEQUOIA_MAX_HZ = 1.0

def gsd(alt):                      # metres per pixel
    return alt * PITCH / FOCAL

def envelope(alt, overlap):
    g = gsd(alt)
    swath = PX * g                 # across-track footprint
    along = PY * g                 # along-track footprint
    advance = along * (1 - overlap)
    v_max = advance * SEQUOIA_MAX_HZ          # sensor-limited ground speed
    ha_h = swath * v_max * 3600 / 1e4
    return g, swath, v_max, ha_h

print("A. SENSOR-LIMITED FLIGHT ENVELOPE  (Sequoia capped at 1 Hz, 75% forward overlap)")
print(f"{'alt m':>6}{'GSD cm':>9}{'swath m':>9}{'max speed m/s':>15}{'coverage ha/h':>15}")
for alt in (5, 10, 20, 30, 40, 60, 100):
    g, sw, v, ha = envelope(alt, 0.75)
    print(f"{alt:>6}{g*100:>9.2f}{sw:>9.1f}{v:>15.2f}{ha:>15.1f}")

print("\nB. TARGET DETECTABILITY  (pixels across a target; ~5-10 px needed to segment)")
targets = [("weed rosette 5 cm", 0.05), ("weed patch 20 cm", 0.20), ("crop plant 40 cm", 0.40)]
print(f"{'alt m':>6}" + "".join(f"{n:>20}" for n, _ in targets))
for alt in (5, 10, 20, 30, 60):
    g = gsd(alt)
    print(f"{alt:>6}" + "".join(f"{s/g:>20.1f}" for _, s in targets))

print("\nC. WHERE THE BOTTLENECK SITS  (mission wants a constant 5 m/s survey speed)")
print(f"{'alt m':>6}{'v_max m/s':>11}{'req Hz @5m/s':>14}{'binding constraint':>26}")
for alt in (5, 10, 20, 30, 60):
    g = gsd(alt); advance = PY*g*0.25
    vmax = advance*SEQUOIA_MAX_HZ
    req = 5.0/advance
    bind = "SEQUOIA (must slow down)" if req > SEQUOIA_MAX_HZ else "neither - headroom"
    print(f"{alt:>6}{vmax:>11.2f}{req:>14.2f}{bind:>26}")
print("  -> below ~22 m altitude the Sequoia, not the Jetson, caps mission speed.")

print("\nD. COMPUTE HEADROOM AT NATIVE SEQUOIA RESOLUTION (1280x960 = 1.23 MP)")
# GMACs scale linearly with pixel count; counter values are for square inputs
base = {"AgriEdgeNet-S @512": 0.62, "AgriEdgeNet-M @512": 2.07}
scale = (PX*PY)/(512*512)
print(f"  pixel-count scale vs 512x512: {scale:.2f}x")
for k, v in base.items():
    name = k.split(" @")[0]
    print(f"  {name:<16} native 1280x960 -> {v*scale:6.2f} GMACs  ({v*scale*2:5.1f} GFLOPs)")
print("\n  Budget available per frame at 1 Hz: 1000 ms")
print("  Even at a pessimistic 10% of peak throughput, both variants finish")
print("  well inside 1 s on Nano-class hardware -> compute is NOT the constraint.")
