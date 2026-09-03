"""Crossing angle at skeleton junctions -- the DECLARED definition (thesis Section 2.2, Appendix A.3).

Each branch's direction is its LOCAL TANGENT at the pixel nearest the junction, via
spacing.tangent_angles (arc-local PCA over +-6 px, which follows curvature instead of averaging
it away). The model's +-90 deg branch rule is defined AT the branch point, so this is the
like-for-like quantity. Junctions are skeleton pixels with >= 3 of 8 neighbours occupied; each
junction is dilated by `near_px` and every pair of branches touching it is counted ONCE.

No minimum branch length is applied. The route has two constants: near_px (2) and arc_px (6).

Driver: full_tan.py (synthetic ladder + traced images -> res_tan.json); figure: fig_tangent.py.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, dilation, disk

import spacing as sp


def tangent_crossings(mask, near_px=2, arc_px=6):
    """Acute angle (deg, 0-90) at every junction; branch angle = local tangent there.
    EACH PAIR COUNTED ONCE."""
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
