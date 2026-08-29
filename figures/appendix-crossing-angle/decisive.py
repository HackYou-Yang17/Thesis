"""
THE DECISIVE TEST, using Luka's own real_crossings() and load_trace().

Feed the measure a field we KNOW is orthogonal, built in the same
representation load_trace() produces (0.40 um/px, dilated to WIDTH_PX),
and see what median it returns.

  ~35 deg back  -> the traced median is a pipeline artefact; tissue is orthogonal
  ~70 deg back  -> the pipeline is fine; the tissue really does cross shallowly
"""
import sys, glob, re, os
import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.draw import line as draw_line

from bias_check import real_crossings
from traced_dominance import load_trace, SCALE_UM, FIELD_UM, WIDTH_PX
from datapaths import ROOT as DATA_ROOT

N = int(round(FIELD_UM / SCALE_UM))

def synth_orthogonal(seed, jitter_deg=25.0, pitch_um=2.80):
    """Two families at EXACTLY 90 deg, per-fibre jitter, in load_trace's output format."""
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N), bool)
    j = np.deg2rad(jitter_deg)
    for fam in (0.0, np.pi/2):
        step = pitch_um / SCALE_UM
        for o in np.arange(-N, 2*N, step):
            a = fam + rng.uniform(-j, j)
            c, s = np.cos(a), np.sin(a)
            px, py = o*(-s) + N/2, o*c + N/2
            L = 3*N
            rr, cc = draw_line(int(py-L*s), int(px-L*c), int(py+L*s), int(px+L*c))
            ok = (rr>=0)&(rr<N)&(cc>=0)&(cc<N)
            if ok.sum() < 10: continue
            img[rr[ok], cc[ok]] = True
    return binary_dilation(img, disk(WIDTH_PX // 2))

def rep(name, a):
    if len(a)==0: print(f"  {name:44s} no pairs"); return
    print(f"  {name:44s} n={len(a):5d}   median={np.median(a):5.1f}   >60deg={np.mean(a>60):.2f}   <30deg={np.mean(a<30):.2f}")

print("SYNTHETIC CONTROL — ground truth is TWO FAMILIES AT EXACTLY 90 DEG")
print("run through Luka's real_crossings(), same grid and width as the traces\n")
for jit in (10.0, 25.0, 40.0):
    acc=[]
    for seed in range(5):
        acc.append(real_crossings(synth_orthogonal(seed, jitter_deg=jit)))
    rep(f"orthogonal, per-fibre jitter +-{jit:.0f} deg", np.concatenate(acc))

print("\nREAL TRACES — same function, reproducing the published numbers\n")
for hpf in (32, 52):
    acc=[]
    for f in sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif")):
        m, *_ = load_trace(f)
        a = real_crossings(m)
        acc.append(a)
        rep(f"{os.path.basename(f).replace(' - Copy.tif','')}", a)
    rep(f"--> {hpf} hpf, pooled", np.concatenate(acc))
