"""Which TRUE separation does the 52 hpf tissue match, once measured through
the same skeleton route? Calibrate against synthetic fields of known separation."""
import sys, glob, os
sys.path.insert(0, "/home/claude/xcheck/bundle/A_traced")
import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.draw import line as draw_line
from bias_check import real_crossings
from traced_dominance import load_trace, SCALE_UM, FIELD_UM, WIDTH_PX
N = int(round(FIELD_UM / SCALE_UM))

def synth(seed, sep_deg, jitter_deg=25.0, pitch_um=2.80):
    rng = np.random.default_rng(seed); img = np.zeros((N,N), bool)
    j = np.deg2rad(jitter_deg)
    for fam in (0.0, np.deg2rad(sep_deg)):
        for o in np.arange(-N, 2*N, pitch_um/SCALE_UM):
            a = fam + rng.uniform(-j, j); c,s = np.cos(a), np.sin(a)
            px, py = o*(-s)+N/2, o*c+N/2; L = 3*N
            rr,cc = draw_line(int(py-L*s), int(px-L*c), int(py+L*s), int(px+L*c))
            ok = (rr>=0)&(rr<N)&(cc>=0)&(cc<N)
            if ok.sum()<10: continue
            img[rr[ok], cc[ok]] = True
    return binary_dilation(img, disk(WIDTH_PX//2))

print("CALIBRATION: synthetic fields of KNOWN separation, +-25 deg jitter, 5 seeds")
print(f"{'true separation':>18s} {'n':>6s} {'median':>8s} {'>60deg':>8s} {'<30deg':>8s}")
for sep in (0, 30, 45, 60, 75, 90):
    a = np.concatenate([real_crossings(synth(s, sep)) for s in range(5)])
    print(f"{sep:>15d} deg {len(a):>6d} {np.median(a):>8.1f} {np.mean(a>60):>8.2f} {np.mean(a<30):>8.2f}")

print("\nTRACED TISSUE, same measure")
print(f"{'':>18s} {'n':>6s} {'median':>8s} {'>60deg':>8s} {'<30deg':>8s}")
for hpf in (32,36,40,44,48,52):
    fs = sorted(glob.glob(f"/mnt/user-data/uploads/analysis/{hpf}hpf/*- Copy.tif"))
    if not fs: continue
    per=[]; acc=[]
    for f in fs:
        m,*_ = load_trace(f); x = real_crossings(m); acc.append(x)
        per.append((np.median(x) if len(x) else np.nan, np.mean(x>60) if len(x) else np.nan))
    a = np.concatenate(acc)
    meds = [p[0] for p in per]; f60=[p[1] for p in per]
    print(f"{hpf:>15d} hpf {len(a):>6d} {np.median(a):>8.1f} {np.mean(a>60):>8.2f} {np.mean(a<30):>8.2f}"
          f"   per-heart median {np.nanmin(meds):.0f}-{np.nanmax(meds):.0f}"
          f" | per-heart >60 {np.nanmin(f60):.2f}-{np.nanmax(f60):.2f}")
