"""Does whole-branch PCA vs junction-local tangent change the crossing-angle result?

bias_check.real_crossings gives each branch ONE principal axis over its whole length, then
takes the acute angle between branches meeting at a junction. If bundles curve, that axis is
not the direction the branch actually has AT the crossing -- which is where the model's +-90
deg rule is defined.

Here the same junctions are used, but each branch's angle is its LOCAL TANGENT at the pixel
nearest the junction, via spacing.tangent_angles (arc-local PCA over +-6 px, which follows
curvature instead of averaging it away).

Both the traced images and the synthetic calibration ladder are remeasured, so the equivalent
separation stays like-for-like.
"""
import sys, glob, os, warnings, json
sys.path.insert(0, "/mnt/user-data/uploads/thesis--thesis/pipeline")
warnings.filterwarnings("ignore")
import logging; logging.getLogger("tifffile").setLevel(logging.ERROR)

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, dilation, disk
from skimage.draw import line as draw_line

import spacing as sp
from bias_check import real_crossings
from traced_dominance import load_trace, SCALE_UM, FIELD_UM, WIDTH_PX
from datapaths import ROOT as DATA_ROOT

N = int(round(FIELD_UM / SCALE_UM))


def tangent_crossings(mask, near_px=2, arc_px=6):
    """Same junctions as real_crossings; branch angle = local tangent at the junction."""
    sk = skeletonize(mask)
    ang = sp.tangent_angles(sk, arc_px=arc_px)                 # radians, mod pi, per pixel
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    junc = sk & (nb >= 3)
    branches, nlab = ndi.label(sk & ~junc, structure=np.ones((3, 3)))
    jl, njl = ndi.label(junc, structure=np.ones((3, 3)))

    out, seen = [], set()
    for j in range(1, njl + 1):
        jmask = jl == j
        jy, jx = np.nonzero(jmask)
        cy, cx = jy.mean(), jx.mean()
        near = dilation(jmask, disk(near_px))
        labs = sorted({int(v) for v in np.unique(branches[near]) if v > 0})
        local = {}
        for lb in labs:
            ys, xs = np.nonzero((branches == lb) & near)
            if ys.size == 0:
                continue
            d2 = (ys - cy) ** 2 + (xs - cx) ** 2
            order = np.argsort(d2)
            for idx in order:                                   # nearest pixel WITH a tangent
                a = ang[ys[idx], xs[idx]]
                if np.isfinite(a):
                    local[lb] = np.degrees(a) % 180.0
                    break
        ks = sorted(local)
        for i in range(len(ks)):
            for k in range(i + 1, len(ks)):
                key = (ks[i], ks[k])
                if key in seen:
                    continue
                seen.add(key)
                d = abs(local[ks[i]] - local[ks[k]]) % 180.0
                out.append(min(d, 180.0 - d))
    return np.array(out)


def synth(seed, sep, jit=25.0, pitch=2.80):
    rng = np.random.default_rng(seed); img = np.zeros((N, N), bool); jr = np.deg2rad(jit)
    for fam in (0.0, np.deg2rad(sep)):
        for o in np.arange(-N, 2 * N, pitch / SCALE_UM):
            a = fam + rng.uniform(-jr, jr); c, s = np.cos(a), np.sin(a)
            px, py = o * (-s) + N / 2, o * c + N / 2; L = 3 * N
            rr, cc = draw_line(int(py - L * s), int(px - L * c), int(py + L * s), int(px + L * c))
            ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
            if ok.sum() < 10:
                continue
            img[rr[ok], cc[ok]] = True
    return dilation(img, disk(WIDTH_PX // 2))


SEPS = [0, 30, 45, 60, 75, 90]


def equiv(f, ladder):
    for i in range(len(SEPS) - 1):
        a, b = ladder[i], ladder[i + 1]
        if (a <= f <= b) or (b <= f <= a):
            return SEPS[i] if b == a else SEPS[i] + (f - a) / (b - a) * (SEPS[i + 1] - SEPS[i])
    return np.nan


if __name__ == "__main__":
    print("=" * 78)
    print("CALIBRATION LADDER, both definitions (fraction of pairs > 60 deg)")
    print("=" * 78)
    print("%6s %10s %10s" % ("sep", "PCA", "tangent"))
    lad_p, lad_t = [], []
    for sep in SEPS:
        ap = np.concatenate([real_crossings(synth(s, sep)) for s in range(5)])
        at = np.concatenate([tangent_crossings(synth(s, sep)) for s in range(5)])
        lad_p.append(float((ap > 60).mean())); lad_t.append(float((at > 60).mean()))
        print("%6d %10.3f %10.3f" % (sep, lad_p[-1], lad_t[-1]))

    print()
    print("=" * 78)
    print("TRACED IMAGES, both definitions")
    print("=" * 78)
    print("%6s %8s %8s %10s %10s" % ("hpf", "nPCA", "nTan", "PCA f>60", "tan f>60"))
    res = {}
    for hpf in (32, 36, 40, 44, 48, 52):
        ap, at = [], []
        for f in sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif")):
            m, *_ = load_trace(f)
            ap.append(real_crossings(m)); at.append(tangent_crossings(m))
        ap = np.concatenate(ap); at = np.concatenate(at)
        res[hpf] = (float((ap > 60).mean()), float((at > 60).mean()))
        print("%6d %8d %8d %10.3f %10.3f" % (hpf, ap.size, at.size, res[hpf][0], res[hpf][1]))

    print()
    print("=" * 78)
    print("EQUIVALENT SEPARATION at 52 hpf")
    print("=" * 78)
    print("  whole-branch PCA : f = %.3f -> %s" %
          (res[52][0], ("%.0f deg" % equiv(res[52][0], lad_p)) if equiv(res[52][0], lad_p) == equiv(res[52][0], lad_p) else "off ladder"))
    print("  junction tangent : f = %.3f -> %s" %
          (res[52][1], ("%.0f deg" % equiv(res[52][1], lad_t)) if equiv(res[52][1], lad_t) == equiv(res[52][1], lad_t) else "off ladder"))
    print()
    print("  ladder discrimination, tangent definition:")
    for i in range(len(SEPS) - 1):
        print("     %2d -> %2d deg : %+.3f" % (SEPS[i], SEPS[i + 1], lad_t[i + 1] - lad_t[i]))
    json.dump(dict(lad_pca=lad_p, lad_tan=lad_t, traced={str(k): v for k, v in res.items()}),
              open("tangent_vs_pca.json", "w"))
