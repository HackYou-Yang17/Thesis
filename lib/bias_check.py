"""Is the traced-crop dominance measure biased toward NEMATIC?

Five tests. The first four are synthetics where the truth is known; the fifth
measures the real traces directly, because the biggest suspected bias -- the
mixture model assumes the two families are 90 deg apart -- depends on a property
of the tissue, not of the code.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.draw import line as dline
from skimage.morphology import skeletonize, dilation, disk

import carma_stats as cs
from traced_dominance import load_trace, SCALE_UM, WINDOW_UM, WIDTH_PX

S = SCALE_UM
N = 154
RNG = np.random.default_rng(7)


def draw(angles, n=110, spread=8.0, length_um=(8, 25), width=WIDTH_PX, rng=RNG):
    m = np.zeros((N, N), bool)
    for _ in range(n):
        a = np.radians(rng.choice(angles) + rng.normal(0, spread))
        L = rng.uniform(*length_um) / S
        y0, x0 = rng.uniform(0, N, 2)
        y1, x1 = y0 + L * np.sin(a), x0 + L * np.cos(a)
        rr, cc = dline(int(np.clip(y0, 0, N - 1)), int(np.clip(x0, 0, N - 1)),
                       int(np.clip(y1, 0, N - 1)), int(np.clip(x1, 0, N - 1)))
        m[rr, cc] = True
    return dilation(m, disk(width // 2))


def p_of(mask, window=WINDOW_UM):
    dm = cs.dominance_map(np.asarray(mask, float), window=window, scale=S,
                          band=(0.05, 0.8), n_surrogates=16)
    r = cs.dominance_summary(dm)
    return r["nematic_frac"], r["frac_gated"], len(dm)


# ── 1 · isotropic control ───────────────────────────────────────────────────
def test_isotropic():
    print("\n1. ISOTROPIC CONTROL — random angles, no family structure at all.")
    print("   An unbiased measure should gate out or sit low; a nematic-biased "
          "one reads high.")
    for n in (60, 110, 200):
        ps = []
        for s in range(5):
            rng = np.random.default_rng(100 + s)
            m = draw(list(range(0, 180, 5)), n=n, spread=0.0, rng=rng)
            p, g, _ = p_of(m)
            ps.append((p, g))
        p = np.nanmean([a for a, _ in ps]); g = np.mean([b for _, b in ps])
        print(f"   n_lines={n:4d}  cov={m.mean():.2f}   p = {p:.3f}   gated {g:.2f}")


# ── 2 · crossing angle ──────────────────────────────────────────────────────
def test_crossing_angle():
    print("\n2. CROSSING ANGLE — two equal families, separation swept.")
    print("   The mixture model assumes 90 deg. Anything shallower is read as "
          "partly nematic BY CONSTRUCTION.")
    rows = []
    for sep in (90, 80, 70, 60, 50, 40, 30):
        m = draw([0, sep], n=110, rng=np.random.default_rng(11))
        p, g, _ = p_of(m)
        rows.append((sep, p, g))
        print(f"   separation {sep:3d} deg   p = {p:.3f}   gated {g:.2f}")
    return pd.DataFrame(rows, columns=["sep_deg", "p", "gated"])


# ── 3 · how many lines are in a window ──────────────────────────────────────
def test_sparsity():
    print("\n3. SPARSITY — a window holding one or two lines cannot show two "
          "families.")
    print("   True structure is two orthogonal families throughout; only the "
          "line count changes.")
    for n in (25, 50, 110, 200, 320):
        m = draw([0, 90], n=n, rng=np.random.default_rng(23))
        p, g, nw = p_of(m)
        skel = skeletonize(m)
        wpx = int(round(WINDOW_UM / S))
        per = np.mean([skel[y:y + wpx, x:x + wpx].sum() * S / 15.0
                       for y in range(0, N - wpx + 1, wpx // 2)
                       for x in range(0, N - wpx + 1, wpx // 2)])
        print(f"   n_lines={n:4d}  cov={m.mean():.2f}  ~{per:4.1f} line-lengths "
              f"per window   p = {p:.3f}   gated {g:.2f}")


# ── 4 · window size on a known two-family field ─────────────────────────────
def test_window():
    print("\n4. WINDOW SIZE — same field, aperture swept. Truth is two "
          "orthogonal families (p = 0).")
    m = draw([0, 90], n=110, rng=np.random.default_rng(31))
    for w in (10.0, 12.5, 15.0, 20.0, 25.0, 40.0):
        try:
            p, g, nw = p_of(m, window=w)
            print(f"   window {w:5.1f} um  ({nw:3d} windows)   p = {p:.3f}   "
                  f"gated {g:.2f}")
        except ValueError as e:
            print(f"   window {w:5.1f} um  -- {e}")


# ── 5 · the real crossing-angle distribution ────────────────────────────────
def branch_angles(mask, min_len_px=8):
    """Split the traced skeleton at junctions; PCA angle per branch."""
    sk = skeletonize(mask)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8),
                      mode="constant") - sk
    junc = sk & (nb >= 3)
    branches, nlab = ndi.label(sk & ~junc, structure=np.ones((3, 3)))
    ang, keep = {}, []
    for lab in range(1, nlab + 1):
        ys, xs = np.nonzero(branches == lab)
        if ys.size < min_len_px:
            continue
        c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
        w, v = np.linalg.eigh(c)
        vx, vy = v[:, -1]
        ang[lab] = np.degrees(np.arctan2(vy, vx)) % 180.0
        keep.append(lab)
    return branches, junc, ang


def real_crossings(mask):
    """Angle at every detected intersection. EACH PAIR COUNTED ONCE."""
    branches, junc, ang = branch_angles(mask)
    jl, njl = ndi.label(junc, structure=np.ones((3, 3)))
    seen, out = set(), []
    for j in range(1, njl + 1):
        near = dilation(jl == j, disk(2))
        labs = sorted({int(v) for v in np.unique(branches[near])
                       if v > 0 and int(v) in ang})
        for i in range(len(labs)):
            for k in range(i + 1, len(labs)):
                key = (labs[i], labs[k])
                if key in seen:
                    continue
                seen.add(key)
                d = abs(ang[labs[i]] - ang[labs[k]]) % 180.0
                out.append(min(d, 180.0 - d))
    return np.array(out)


def test_real_crossings():
    print("\n5. REAL CROSSING ANGLES — measured on the traces themselves.")
    print("   Each intersecting pair counted ONCE (weighting by shared pixels "
          "biases hard toward parallel).")
    files = sorted(glob.glob("/mnt/user-data/uploads/analysis/*/*Copy.tif"),
                   key=lambda f: (int(re.search(r"(\d+)hpf", f).group(1)), f))
    rows = []
    for f in files:
        hpf = float(re.search(r"(\d+)hpf", f).group(1))
        name = os.path.basename(f).replace(" - Copy.tif", "")
        m, _, _ = load_trace(f)
        a = real_crossings(m)
        rows.append(dict(file=name, hpf=hpf, n_pairs=len(a),
                         median=np.median(a) if len(a) else np.nan,
                         frac_gt60=float((a > 60).mean()) if len(a) else np.nan,
                         frac_lt30=float((a < 30).mean()) if len(a) else np.nan))
    df = pd.DataFrame(rows)
    tp = df.groupby("hpf").agg(n_img=("file", "size"), pairs=("n_pairs", "mean"),
                               median=("median", "mean"),
                               frac_gt60=("frac_gt60", "mean"),
                               frac_lt30=("frac_lt30", "mean")).reset_index()
    print(tp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv("crossing_angles_per_image.csv", index=False)
    tp.to_csv("crossing_angles_per_timepoint.csv", index=False)
    return df, tp


if __name__ == "__main__":
    test_isotropic()
    sweep = test_crossing_angle()
    sweep.to_csv("crossing_sweep.csv", index=False)
    test_sparsity()
    test_window()
    test_real_crossings()
