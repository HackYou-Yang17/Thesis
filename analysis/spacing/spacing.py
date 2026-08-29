"""
spacing.py -- perpendicular nearest-neighbour fibre spacing on the traced crops.

WHAT IS MEASURED
----------------
For each analysis window, the local dominant fibre direction theta is the MODE of the
window's local skeleton tangent distribution (see mode_axis). Scan lines are then cast
PERPENDICULAR to theta and the distance between consecutive fibre crossings along each
scan line is recorded. That distance is the spacing between neighbouring fibres, in um.

WHY THIS DEFINITION
-------------------
It is the direct analogue of the model's MESH_SPACING / exclusion pitch: the centre-to-
centre distance between adjacent fibres measured across them, not along them. Chosen over
(a) the empty-space distance transform, which mixes crossing angle into the number, and
(b) the radial FFT peak, which yields one value per image and so has no within-image
distribution to put an error bar on.

WHY IT IS MEASURED ON THE SKELETON
----------------------------------
Traced pen width is 1 px at the INPUT scale, and input scale is correlated with timepoint
(0.361 um/px early, 0.5675 um/px late). A width-sensitive measure would therefore change
systematically along the axis under test. Skeletonising removes width entirely; resampling
to a common 0.40 um/px grid makes the measurement floor identical at every timepoint.
Both steps are the ones already used by the dominance pipeline, unchanged.

FAMILY RESTRICTION
------------------
A crossing fibre also intersects the scan line, so an unrestricted scan measures pore size,
not fibre spacing, and would shrink automatically as a second family appears. Hits are
therefore restricted to skeleton pixels whose local tangent lies within ANGLE_TOL of the
window's dominant direction, which measures the SAME physical quantity -- spacing within
the dominant family -- at every timepoint. The unrestricted (pore) version is computed
alongside as a declared secondary.

ANGLE_TOL = 20 deg is NOT tuned to the answer. It is the largest tolerance that still
recovers a known 4.00 um pitch exactly when a second family is present 30 deg away;
at 30 deg tolerance the same case reads 2.40 um. Residual limit, stated: two families
only 20 deg apart are not separable at any tolerance (reads 2.50 vs 4.00 truth).
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# ---- fixed pipeline constants (inherited from the dominance pipeline, not refitted) ----
TARGET_UM_PER_PX = 0.40     # common grid
FIELD_UM = 61.4             # common centre-cropped field
RED = (237, 28, 36)         # ImageJ red, anti-aliasing off
RED_MARGIN = 60             # red - max(green, blue) > 60


# ----------------------------------------------------------------- loading / normalising
def read_scale(path):
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        xr = p.tags['XResolution'].value
        return xr[1] / xr[0]


def extract_trace(path):
    """Exact recovery of the red annotation layer. No threshold parameter is fitted."""
    a = tifffile.imread(path)
    if a.ndim == 2:
        raise ValueError('no colour channels in %s' % path)
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    return (r - np.maximum(g, b)) > RED_MARGIN


def normalise(mask, um_per_px, target=TARGET_UM_PER_PX, field_um=FIELD_UM):
    """skeletonise -> resample to a common grid -> re-skeletonise -> centre crop."""
    sk = skeletonize(mask)
    zoom = um_per_px / target
    if abs(zoom - 1.0) > 1e-6:
        sk = ndi.zoom(sk.astype(np.uint8), zoom, order=0) > 0
        sk = skeletonize(sk)
    n = int(round(field_um / target))
    h, w = sk.shape
    if h < n or w < n:
        pad = ((max(0, (n - h + 1) // 2), max(0, n - h - (n - h + 1) // 2)),
               (max(0, (n - w + 1) // 2), max(0, n - w - (n - w + 1) // 2)))
        sk = np.pad(sk, pad)
        h, w = sk.shape
    i0, j0 = (h - n) // 2, (w - n) // 2
    return sk[i0:i0 + n, j0:j0 + n]


# ----------------------------------------------------------------- local tangent angles
def tangent_angles(sk, arc_px=6):
    """Local tangent angle (radians, mod pi) at every skeleton pixel.

    Skeleton is split at junctions into branches; each branch is walked and the tangent
    at a pixel is the PCA axis of the +-arc_px pixels around it ALONG THE ARC, so curvature
    is followed instead of being averaged away. Isolated fragments shorter than 3 px carry
    no reliable direction and are dropped.
    """
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - sk
    junction = sk & (nb >= 3)
    branches = sk & ~junction
    lab, nlab = ndi.label(branches, structure=np.ones((3, 3)))

    ang = np.full(sk.shape, np.nan)
    for sl, idx in zip(ndi.find_objects(lab), range(1, nlab + 1)):
        if sl is None:
            continue
        sub = lab[sl] == idx
        pts = np.argwhere(sub)
        if len(pts) < 3:
            continue
        pts = pts + np.array([sl[0].start, sl[1].start])
        order = _walk(pts)
        pts = pts[order]
        n = len(pts)
        for k in range(n):
            lo, hi = max(0, k - arc_px), min(n, k + arc_px + 1)
            seg = pts[lo:hi].astype(float)
            if len(seg) < 3:
                continue
            seg = seg - seg.mean(0)
            u, s, vt = np.linalg.svd(seg, full_matrices=False)
            v = vt[0]
            ang[pts[k, 0], pts[k, 1]] = np.arctan2(v[0], v[1]) % np.pi
    return ang


def _walk(pts):
    """Order the pixels of a thin branch from one end to the other (greedy nearest walk)."""
    n = len(pts)
    d = np.abs(pts[:, None, :] - pts[None, :, :]).max(-1)
    deg = ((d == 1).sum(1))
    start = int(np.argmin(deg)) if (deg == 1).any() else 0
    order, seen = [start], {start}
    cur = start
    for _ in range(n - 1):
        cand = np.where((d[cur] == 1))[0]
        cand = [c for c in cand if c not in seen]
        if not cand:
            rest = [i for i in range(n) if i not in seen]
            if not rest:
                break
            cur = min(rest, key=lambda i: d[cur, i])
        else:
            cur = min(cand, key=lambda c: d[cur, c])
        seen.add(cur)
        order.append(cur)
    return np.array(order)


def circ_mean_axis(angles, weights=None):
    """Circular mean of an axial (mod pi) angle set, via the doubled-angle representation."""
    if len(angles) == 0:
        return np.nan
    w = np.ones(len(angles)) if weights is None else weights
    c = np.sum(w * np.cos(2 * angles))
    s = np.sum(w * np.sin(2 * angles))
    return (0.5 * np.arctan2(s, c)) % np.pi


def mode_axis(angles, kappa_deg=10.0, nbin=180):
    """Peak of the smoothed axial angle distribution.

    WHY THE MODE AND NOT THE MEAN: with two families present the circular MEAN sits
    BETWEEN them, so both families fall inside any angle tolerance wide enough to be
    useful and the second family is counted as a neighbour of the first -- which halves
    the apparent spacing exactly when a second family appears, i.e. along the axis under
    test. The mode sits ON a family, so the other one is excluded whenever the separation
    exceeds the tolerance. Verified in controls.py (two families 30-90 deg apart).
    """
    if len(angles) == 0:
        return np.nan
    grid = np.linspace(0, np.pi, nbin, endpoint=False)
    d = np.abs(((angles[None, :] - grid[:, None] + np.pi / 2) % np.pi) - np.pi / 2)
    w = np.exp(-0.5 * (d / np.deg2rad(kappa_deg)) ** 2)
    return float(grid[np.argmax(w.sum(1))])


# ----------------------------------------------------------------- the spacing measure
def scan_gaps(sub_mask, theta, um_per_px, line_step_um=1.0, sample_step_px=0.25):
    """Distances between consecutive crossings along scan lines perpendicular to theta.

    sub_mask : boolean, the (already family-filtered) skeleton inside the window
    theta    : dominant direction in radians, angle of the fibres
    Returns gaps in um. Only INTERIOR gaps are returned -- the leading and trailing
    intervals of every scan line are censored by the window edge and are dropped, which
    would otherwise bias the distribution downward.
    """
    if sub_mask.sum() < 2:
        return np.array([])
    # Close the corner contacts of a diagonally-stepping skeleton so a scan line cannot
    # slip between two touching pixels. A PLUS-shaped element is used, not a full 3x3:
    # it closes the diagonal gap just as well but widens each hit by 1 px instead of 2,
    # which lowers the smallest resolvable spacing from ~2.2 um to ~1.6 um (controls.py).
    m = ndi.binary_dilation(sub_mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    h, w = m.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # theta measured from the +x axis in (row, col) space: direction along fibre
    d = np.array([np.sin(theta), np.cos(theta)])       # along fibre  (row, col)
    u = np.array([np.cos(theta), -np.sin(theta)])      # across fibre (row, col)

    half = 0.5 * np.hypot(h, w)
    step_px = line_step_um / um_per_px
    offsets = np.arange(-half, half + 1e-9, step_px)   # positions of scan lines along d
    t = np.arange(-half, half + 1e-9, sample_step_px)  # samples along u

    gaps = []
    for off in offsets:
        base = np.array([cy, cx]) + off * d
        pts = base[None, :] + t[:, None] * u[None, :]
        rr, cc = pts[:, 0], pts[:, 1]
        inside = (rr >= 0) & (rr <= h - 1) & (cc >= 0) & (cc <= w - 1)
        if inside.sum() < 4:
            continue
        vals = np.zeros(len(t), bool)
        vals[inside] = ndi.map_coordinates(m.astype(np.uint8),
                                           [rr[inside], cc[inside]],
                                           order=0, mode='constant') > 0
        # contiguous True runs -> one hit each, located at the run centre
        idx = np.where(vals)[0]
        if len(idx) < 3:
            continue
        splits = np.where(np.diff(idx) > 1)[0]
        runs = np.split(idx, splits + 1)
        centres = np.array([t[r].mean() for r in runs])
        if len(centres) < 3:
            continue
        # drop the first and last interval: censored by the window edge
        dif = np.diff(centres)
        gaps.append(dif * um_per_px)
    if not gaps:
        return np.array([])
    return np.concatenate(gaps)


def image_spacing(sk, um_per_px=TARGET_UM_PER_PX, window_um=15.0, stride_um=7.5,
                  angle_tol_deg=20.0, min_px=25, min_gaps=5):
    """Per-window spacing over one normalised image.

    Returns (rows, all_gaps_family, all_gaps_pore) where rows is a list of per-window dicts.
    """
    ang = tangent_angles(sk)
    n = sk.shape[0]
    wpx = int(round(window_um / um_per_px))
    spx = int(round(stride_um / um_per_px))
    tol = np.deg2rad(angle_tol_deg)

    rows, fam_all, pore_all = [], [], []
    for i0 in range(0, n - wpx + 1, spx):
        for j0 in range(0, n - wpx + 1, spx):
            sub = sk[i0:i0 + wpx, j0:j0 + wpx]
            suba = ang[i0:i0 + wpx, j0:j0 + wpx]
            good = np.isfinite(suba)
            if good.sum() < min_px:
                continue
            theta = mode_axis(suba[good])
            # axial difference to the window director
            dth = np.abs(((suba - theta + np.pi / 2) % np.pi) - np.pi / 2)
            fam = good & (dth <= tol)
            if fam.sum() < min_px:
                continue
            g_fam = scan_gaps(fam, theta, um_per_px)
            g_pore = scan_gaps(sub, theta, um_per_px)
            if len(g_fam) < min_gaps:
                continue
            rows.append(dict(i0=i0, j0=j0, theta_deg=np.rad2deg(theta),
                             n_skel=int(good.sum()), n_family=int(fam.sum()),
                             family_frac=float(fam.sum() / good.sum()),
                             n_gaps=len(g_fam),
                             spacing_um=float(np.median(g_fam)),
                             pore_um=float(np.median(g_pore)) if len(g_pore) else np.nan))
            fam_all.append(g_fam)
            pore_all.append(g_pore)
    fam_all = np.concatenate(fam_all) if fam_all else np.array([])
    pore_all = np.concatenate(pore_all) if pore_all else np.array([])
    return rows, fam_all, pore_all


def line_density(sk, um_per_px=TARGET_UM_PER_PX):
    """Traced skeleton length per unit area, um / um^2."""
    area = sk.size * um_per_px ** 2
    return sk.sum() * um_per_px / area


def isotropic_spacing(lam):
    """Mean crossing spacing along a test line through an ISOTROPIC Poisson line process
    of length density lam. Intersections per unit test length = 2*lam/pi, so spacing =
    pi/(2*lam). This is the null: what the spacing would be if the traced length were
    thrown down with no structure at all. Measured spacing well below it means the fibres
    are ordered; equal to it means the measure is reporting density and nothing more."""
    return np.pi / (2.0 * lam)


def load_normalised(path):
    return normalise(extract_trace(path), read_scale(path))
