"""
nn.py -- window-free perpendicular nearest-neighbour fibre spacing.

WHY THIS REPLACES THE SCAN-LINE VERSION
---------------------------------------
The first implementation cast scan lines across an analysis window and took every
consecutive gap along them. Measured spacing then GREW WITH THE WINDOW (2.70 um at a
10 um aperture, 3.13 at 15, 3.37 at 20, 3.83 at 25) because a long scan line crosses
regions holding no traced fibre at all, and those empty crossings enter the gap
distribution. A spacing that depends on the aperture is not a length scale, and the
early/late difference it produced appeared only at the largest aperture -- exactly the
signature of an artefact.

THE MEASURE HERE HAS NO APERTURE.
For every traced fibre pixel: step perpendicular to that pixel's OWN local tangent, in
both directions, and record the distance to the first OTHER fibre running within
ANGLE_TOL of the same direction. That distance is the centre-to-centre spacing to the
neighbouring fibre of the same family. It is defined pixel by pixel, so there is no
window, no window director, and nothing for a window size to change.

Rules, each with its reason:
  - hits on the SAME traced branch are skipped: a curving fibre can re-enter its own
    perpendicular ray and would otherwise report its own curvature as a spacing
  - a fibre crossing at more than ANGLE_TOL does NOT stop the ray. It is not a neighbour
    in the family; stopping there would measure pore size, and pore size shrinks
    automatically whenever a second family appears -- i.e. along the axis under test
  - rays finding nothing within MAX_UM are CENSORED, not counted as a large spacing.
    The censored fraction is reported with every number
"""
import numpy as np
from scipy import ndimage as ndi
import spacing as sp

ANGLE_TOL_DEG = 20.0
MAX_UM = 20.0
STEP_PX = 0.5


def branch_labels(sk):
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - sk
    junction = sk & (nb >= 3)
    lab, _ = ndi.label(sk & ~junction, structure=np.ones((3, 3)))
    return lab


def _aligned(ang, r, c, a0, tol):
    """Is the fibre at (r, c) running within tol of a0?

    Junction pixels are split out of every branch and so carry NO tangent. Wherever two
    fibres cross, the neighbouring fibre is represented at that point only by junction
    pixels, and a ray would pass straight through it and report double the true spacing.
    Measured on a regular synthetic lattice: half of all rays returned 7.6 um against a
    4.0 um truth at 45-70 deg crossing. A junction pixel is therefore resolved by looking
    at the tangents of the branch pixels touching it.
    """
    a = ang[r, c]
    if np.isfinite(a):
        return abs(((a - a0 + np.pi / 2) % np.pi) - np.pi / 2) <= tol
    h, w = ang.shape
    sub = ang[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)]
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return False
    return bool((np.abs(((sub - a0 + np.pi / 2) % np.pi) - np.pi / 2) <= tol).any())


def nn_spacings(sk, um_per_px=sp.TARGET_UM_PER_PX, angle_tol_deg=ANGLE_TOL_DEG,
                max_um=MAX_UM, step_px=STEP_PX, subsample=1):
    """Return (both_sides_um, nearest_um, censored_frac).

    both_sides_um : every perpendicular distance found, both directions pooled -- this is
                    the centre-to-centre PITCH distribution
    nearest_um    : per source pixel, the smaller of the two sides -- nearest neighbour
    """
    ang = sp.tangent_angles(sk)
    lab = branch_labels(sk)
    h, w = sk.shape
    max_px = max_um / um_per_px
    tol = np.deg2rad(angle_tol_deg)

    src = np.argwhere(np.isfinite(ang))
    if subsample > 1:
        src = src[::subsample]

    both, nearest, censored = [], [], 0
    ts = np.arange(step_px, max_px + 1e-9, step_px)
    for (r0, c0) in src:
        a0 = ang[r0, c0]
        l0 = lab[r0, c0]
        u = np.array([np.cos(a0), -np.sin(a0)])
        sides = []
        for sgn in (+1, -1):
            hit = np.nan
            rr = r0 + sgn * ts * u[0]
            cc = c0 + sgn * ts * u[1]
            ri = np.rint(rr).astype(int)
            ci = np.rint(cc).astype(int)
            ok = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
            if not ok.any():
                sides.append(np.nan)
                continue
            ri, ci, tt = ri[ok], ci[ok], ts[ok]
            cand = np.where(sk[ri, ci] & (lab[ri, ci] != l0))[0]
            for k in cand:
                if _aligned(ang, ri[k], ci[k], a0, tol):
                    hit = tt[k] * um_per_px
                    break
            sides.append(hit)
        good = [s for s in sides if np.isfinite(s)]
        if not good:
            censored += 1
            continue
        both.extend(good)
        nearest.append(min(good))
    n = len(src)
    return (np.array(both), np.array(nearest),
            censored / n if n else np.nan)
