# Required onboard processing rate derived from flight geometry, not from hardware capability.
# Camera model: 1" sensor, 13.2 mm width, 5472x3648 px, 8.8 mm focal length.
SW, PW, PH, F = 13.2, 5472, 3648, 8.8

def rate(alt, speed, fwd_overlap):
    gsd = alt * SW / (F * PW)            # m per pixel
    swath = PW * gsd                      # across-track footprint (m)
    fwd   = PH * gsd                      # along-track footprint (m)
    advance = fwd * (1 - fwd_overlap)     # ground distance between consecutive frames
    hz = speed / advance                  # required capture/processing rate
    ha_per_h = swath * speed * 3600 / 10000
    return gsd*100, swath, fwd, hz, ha_per_h

print(f"{'regime':<34}{'GSD cm':>8}{'swath m':>9}{'req Hz':>8}{'ha/h*':>8}")
rows = [
 ("2 m / 1 m/s  (weed species ID)",   2,  1.0, 0.75),
 ("6 m / 2 m/s  (Doc 8 rice)",        6,  2.0, 0.75),
 ("12 m / 3.5 m/s (Doc 11 rice)",    12,  3.5, 0.60),
 ("30 m / 5 m/s  (survey)",          30,  5.0, 0.75),
 ("60 m / 8 m/s  (broad survey)",    60,  8.0, 0.75),
]
for name, a, v, o in rows:
    g, sw, fw, hz, ha = rate(a, v, o)
    print(f"{name:<34}{g:>8.2f}{sw:>9.1f}{hz:>8.2f}{ha:>8.1f}")
print("\n* theoretical strip-coverage rate, excludes turns/climb/battery swaps (derate ~40-50%)")
print("\nHeadroom check: AgriEdgeNet-M projected 20-35 fps @384 on Orin Nano FP16")
for name,a,v,o in rows:
    hz = rate(a,v,o)[3]
    print(f"  {name:<34} required {hz:5.2f} Hz -> headroom vs 20 fps = {20/hz:5.1f}x")
