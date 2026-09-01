"""res_tan.json — the same quantities as full.py, but with JUNCTION-LOCAL TANGENT angles.

full.py gives each branch one PCA principal axis over its whole length. The model's
+-90 deg rule is defined AT the branch point, so the like-for-like branch direction is the
local tangent there. Everything else is identical: same junctions, same disk(2) pairing,
same min(d, 180-d) fold, same synthetic ladder, same traced files.
"""
import sys, glob, json, warnings
sys.path.insert(0, "/mnt/user-data/uploads/thesis--thesis/pipeline")
warnings.filterwarnings("ignore")
import logging; logging.getLogger("tifffile").setLevel(logging.ERROR)

import numpy as np
from skimage.morphology import dilation, disk
from skimage.draw import line as draw_line
from traced_dominance import load_trace, SCALE_UM, FIELD_UM, WIDTH_PX
from datapaths import ROOT as DATA_ROOT
from tangent_crossings import tangent_crossings

N = int(round(FIELD_UM / SCALE_UM))


def synth(seed, sep, jit=25.0, pitch=2.80):
    rng = np.random.default_rng(seed); img = np.zeros((N, N), bool); j = np.deg2rad(jit)
    for fam in (0.0, np.deg2rad(sep)):
        for o in np.arange(-N, 2 * N, pitch / SCALE_UM):
            a = fam + rng.uniform(-j, j); c, s = np.cos(a), np.sin(a)
            px, py = o * (-s) + N / 2, o * c + N / 2; L = 3 * N
            rr, cc = draw_line(int(py - L * s), int(px - L * c), int(py + L * s), int(px + L * c))
            ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
            if ok.sum() < 10:
                continue
            img[rr[ok], cc[ok]] = True
    return dilation(img, disk(WIDTH_PX // 2))


cal = {}
for sep in (0, 30, 45, 60, 75, 90):
    a = np.concatenate([tangent_crossings(synth(s, sep)) for s in range(5)])
    cal[sep] = dict(n=len(a), med=float(np.median(a)), f60=float(np.mean(a > 60)), ang=a.tolist())
    print("  ladder %2d deg  n=%5d  f60=%.3f" % (sep, len(a), cal[sep]["f60"]), flush=True)

real = {}
for hpf in (32, 36, 40, 44, 48, 52):
    per, acc = [], []
    for f in sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif")):
        x = tangent_crossings(load_trace(f)[0]); acc.append(x)
        per.append(dict(med=float(np.median(x)) if len(x) else None,
                        f60=float(np.mean(x > 60)) if len(x) else None, n=len(x)))
    a = np.concatenate(acc)
    real[hpf] = dict(n=len(a), med=float(np.median(a)), f60=float(np.mean(a > 60)),
                     per=per, ang=a.tolist())
    print("  traced %2d hpf  n=%5d  f60=%.3f  per-heart %s"
          % (hpf, len(a), real[hpf]["f60"], ", ".join("%.2f" % p["f60"] for p in per)), flush=True)

json.dump(dict(cal={str(k): v for k, v in cal.items()},
               real={str(k): v for k, v in real.items()}), open("res_tan.json", "w"))
print("wrote res_tan.json")
