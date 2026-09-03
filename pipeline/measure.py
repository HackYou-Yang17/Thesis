"""Shared measurement layer — the ONE pipeline both the traces and the model go through.

Everything here is lifted unchanged from the v1/v2 handoff scripts so that the v5 numbers
stay comparable in method (not in value) with the earlier rounds:

  normalise()   = traced_dominance.load_trace minus the red-channel extraction
                  (skeletonise -> 0.40 um/px -> re-skeletonise -> 1.2 um width -> 61.4 um
                   centre crop). Required because traced pixel size runs 0.361-0.5675 um/px
                  and CORRELATES with timepoint.
  density()     = skeleton line density on the normalised mask (centreline length per area)
  order()       = +-45 deg whole-crop band statistic with the surrogate-matched null
                  (band_dominance.band_map/band_summary, half_band=45)
  foam()        = enclosed background regions per 1000 px of skeleton (sweep2.foam_index)
  gap()         = distance-to-nearest-fibre percentiles (fieldstats.gap_percentiles),
                  border_margin 4 um, p90/p95 only  <-- NEW in v5
  count()       = skeleton segment count (bias_check.branch_angles, reimplemented)

NOTE ON count(): the handover records that bias_check.branch_angles was missing from the
handoff and had to be reimplemented, so the absolute value may carry an offset against the
v3/v4 numbers. It is REPORTED, never optimised (see handover S4.1).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, dilation, disk
from skimage.transform import resize

import fieldstats as cs
from band_dominance import band_map, band_summary

SCALE_UM = 0.40      # common resampling grid
FIELD_UM = 61.4      # smallest field in the traced set; all crops trimmed to it
WIDTH_PX = 3         # fixed trace width after normalisation (1.2 um)
HALF_BAND = 45.0     # the order statistic's band half-width
BAND = (0.05, 0.8)   # cycles/um
BORDER_MARGIN_UM = 4.0
RED_EXCESS = 60
MIN_SEG_PX = 8       # branch_angles: runs shorter than this are dropped


def normalise(mask, scale_in):
    """Binary fibre mask at any pixel size -> normalised mask at SCALE_UM, FIELD_UM crop."""
    skel = skeletonize(np.asarray(mask, bool))
    n_out = [int(round(n * scale_in / SCALE_UM)) for n in skel.shape]
    rs = resize(skel.astype(float), n_out, order=1, anti_aliasing=False) > 0.15
    rs = skeletonize(rs)                       # resampling thickens; re-thin
    out = dilation(rs, disk(WIDTH_PX // 2))
    n = int(round(FIELD_UM / SCALE_UM))         # centre-crop, never rescale
    r0 = max(0, (out.shape[0] - n) // 2)
    c0 = max(0, (out.shape[1] - n) // 2)
    out = out[r0:r0 + n, c0:c0 + n]
    if out.shape != (n, n):                     # pad the 1-2 px rounding shortfall
        pad = np.zeros((n, n), bool)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out


def load_trace(path):
    """ImageJ red annotation layer -> normalised binary line drawing at SCALE_UM."""
    import tifffile
    a = tifffile.imread(path)[..., :3].astype(int)
    with tifffile.TiffFile(path) as tf:
        xr = tf.pages[0].tags["XResolution"].value
        yr = tf.pages[0].tags["YResolution"].value
        s_in = (yr[1] / yr[0], xr[1] / xr[0])
    raw = (a[..., 0] - np.maximum(a[..., 1], a[..., 2])) > RED_EXCESS
    return normalise(raw, s_in[0]), s_in[0]


# ── the statistics ───────────────────────────────────────────────────────────

def density(m):
    """Skeleton line density: centreline pixels per pixel of field."""
    return float(skeletonize(m).sum() / m.size)


def order(m, seed=0):
    """+-45 deg whole-crop band statistic, null-corrected against radial surrogates."""
    r = band_summary(band_map(np.asarray(m, float), window=None, scale=SCALE_UM,
                              band=BAND, half_band=HALF_BAND, seed=seed))
    return float(r["nematic_frac"])


MIN_HOLE_PX = 8          # 8 px = 1.13 um across. See foam2().


def _hole_areas(m):
    """Areas (px) of every enclosed (non-border-touching) background region."""
    sk = skeletonize(np.asarray(m, bool))
    lab, n = ndi.label(~sk, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return np.zeros(0, int)
    border = (set(lab[0].tolist()) | set(lab[-1].tolist())
              | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist()))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = [i for i in range(1, n + 1) if i not in border]
    return sizes[keep] if keep else np.zeros(0, int)


def foam(m):
    """Enclosed background regions per 1000 px. A fibre network has few; a slab is holes."""
    return 1000.0 * _hole_areas(m).size / m.size


def foam2(m, min_px=MIN_HOLE_PX):
    """foam, and foam counting only regions of at least `min_px`.

    ADDED 23 Aug 2026, and it is a correction rather than an extra statistic. foam() is a
    COUNT of enclosed regions and does not weight them by size, so a 1-px loop closed by a
    skeletonisation artefact counts exactly as much as a mesh cell 3 um across. Measured on
    the 18 traced hearts, enclosed regions have a median area of 48-113 px at 40-52 hpf
    (2.8-4.3 um across, i.e. MESH_SPACING) and only 5-17 % are <= 3 px. Measured on the
    model, 48-65 % are <= 3 px and the median is 0.4-1.2 um across. So the raw index counts
    mesh in the traces and counts rendering noise in the model, and the two are not the
    same quantity even though they carry the same name.

    min_px = 8 (1.13 um across) is set well BELOW the traced mode, not at it: the cut has to
    remove the artefact population without trimming the real one. foam_ge4/16/32 were all
    computed and the conclusion is identical for every cut from 4 px up.
    """
    a = _hole_areas(m)
    return (1000.0 * a.size / m.size, 1000.0 * int((a >= min_px).sum()) / m.size)


def gap(m, percentiles=(90, 95)):
    """Distance-to-nearest-fibre upper percentiles, um. Border margin is mandatory."""
    return cs.gap_percentiles(m, scale=SCALE_UM, percentiles=percentiles,
                              border_margin=BORDER_MARGIN_UM)


_N8 = np.ones((3, 3), int)


def branch_angles(mask, min_len_px=MIN_SEG_PX):
    """Skeletonise -> drop junction pixels (degree >= 3, 8-connectivity) -> label runs ->
    keep runs >= min_len_px -> principal-axis angle of each. len(angles) is the count."""
    sk = skeletonize(np.asarray(mask, bool))
    deg = ndi.convolve(sk.astype(int), _N8, mode="constant") - 1
    runs = sk & (deg < 3)
    lab, n = ndi.label(runs, structure=_N8)
    if n == 0:
        return np.array([])
    angles = []
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        if sub.sum() < min_len_px:
            continue
        yy, xx = np.nonzero(sub)
        y = yy - yy.mean(); x = xx - xx.mean()
        # principal axis of the point cloud
        cov = np.array([[float(np.dot(x, x)), float(np.dot(x, y))],
                        [float(np.dot(x, y)), float(np.dot(y, y))]])
        w, v = np.linalg.eigh(cov)
        vx, vy = v[:, int(np.argmax(w))]
        angles.append(np.degrees(np.arctan2(vy, vx)) % 180.0)
    return np.asarray(angles, float)


def count(m):
    return int(len(branch_angles(m)))


def structure(mask, min_len_px=MIN_SEG_PX):
    """Segment count, mean segment length, and the two STRUCTURE statistics, in one pass.

    frag = share of skeleton length sitting in runs shorter than min_len_px (3.2 um)
    junc = share of skeleton length in junction pixels (8-connectivity degree >= 3)

    These are what the visual gate sees and the scalar metrics do not: a field can carry the
    right amount of fibre, at the right void size, and still be a spray of stubs rather than
    long crossing lines. Neither is a measure of AMOUNT -- both are shares of the skeleton --
    so they are close to orthogonal to density, gap and coverage by construction.
    """
    sk = skeletonize(np.asarray(mask, bool))
    tot = int(sk.sum())
    if tot == 0:
        return dict(count=0, seg_len_um=np.nan, frag=np.nan, junc=np.nan)
    deg = ndi.convolve(sk.astype(int), _N8, mode="constant") - 1
    lab, n = ndi.label(sk & (deg < 3), structure=_N8)
    if n == 0:
        return dict(count=0, seg_len_um=np.nan, frag=np.nan, junc=1.0)
    sizes = np.bincount(lab.ravel())[1:]
    keep = sizes >= min_len_px
    return dict(count=int(keep.sum()),
                seg_len_um=float(sizes[keep].mean() * SCALE_UM) if keep.any() else np.nan,
                frag=float(sizes[~keep].sum() / tot),
                junc=float((tot - sizes.sum()) / tot))


def measure_all(m, seed=0):
    g = gap(m)
    st = structure(m)
    fm, fm8 = foam2(m)
    fb = fibres(m)                      # ADDITIVE: fib_* only; no existing key changes
    return dict(density=density(m), order=order(m, seed=seed), foam=fm, foam_ge8=fm8,
                gap_p90=g[90], gap_p95=g[95], coverage=float(m.mean()), **st, **fb)


# ── fibre stitching: whole fibres THROUGH junctions ──────────────────────────
# ADDITIVE. Nothing above this line is touched; seg_len_um and every other key are unchanged.
#
# WHY: seg_len_um measures runs BETWEEN junctions, so it is mechanically entangled with how
# many crossings there are -- more crossings cut the same fibre into more, shorter pieces.
# fib_len_um traces a fibre THROUGH its crossings, so "how long are the fibres" is separated
# from "how many times do they cross".
#
# TWO DEVIATIONS FROM THE OBVIOUS IMPLEMENTATION, both deliberate and both measured below:
#  * LENGTH IS EUCLIDEAN PATH LENGTH (1.0 orthogonal, sqrt(2) diagonal), not pixel count.
#    NOTE THAT structure()'s seg_len_um IS A PIXEL COUNT, so it under-measures a diagonal run
#    by up to 41 %. The two length statistics are therefore NOT on the same scale and must not
#    be differenced. seg_len_um is left exactly as it is -- changing it would move a number the
#    whole v9-v13 record is quoted against.
#  * WALK ORDER IS LONGEST-EDGE-FIRST. The stitch is greedy, so which edge starts a fibre
#    changes the result; starting from the longest unused run is a stated rule rather than
#    label order, which is an artefact of ndi.label's raster scan.

# max_turn_deg IS A NEW FREE PARAMETER in a pipeline whose selling point is having none, so it
# is SWEPT and declared rather than chosen. Sweeping 15-60 deg (FIBRE_LENGTH.md, step 2): the
# traced mean rises steeply to ~30 deg (+19 % from 15 to 30) and then flattens -- per-step change
# 2.4 / 2.3 / 1.3 / 1.4 / 0.7 % across 35 / 40 / 45 / 50 / 60. 30 deg sits on the SHOULDER; the
# plateau starts at ~35-40. 40 is taken. Two families at 90 deg are never merged at any value
# tried up to 60 (their recovered length moves only +2.0 % -> +2.9 %), so the choice does not
# risk the failure the parameter exists to prevent.
_TURN_DEG = 40.0


def _order_run(pts):
    """Order one run's pixels end to end. Returns the ordered (row, col) array."""
    if len(pts) == 1:
        return pts
    idx = {tuple(p): i for i, p in enumerate(map(tuple, pts))}
    nbr = [[] for _ in pts]
    for i, (r, c) in enumerate(pts):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    j = idx.get((r + dr, c + dc))
                    if j is not None:
                        nbr[i].append(j)
    ends = [i for i, nb in enumerate(nbr) if len(nb) == 1]
    start = ends[0] if ends else 0            # no end => closed loop, start anywhere
    order, seen, cur, prev = [start], {start}, start, None
    while True:
        nxt = [j for j in nbr[cur] if j != prev and j not in seen]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        seen.add(cur)
        order.append(cur)
    return pts[order]


def _path_len(p):
    """Euclidean path length in pixels: 1.0 per orthogonal step, sqrt(2) per diagonal."""
    if len(p) < 2:
        return 0.0
    d = np.abs(np.diff(p.astype(float), axis=0))
    return float(np.sum(np.where(d.sum(1) == 2, np.sqrt(2.0), 1.0)))


def _end_dir(p, k=8):
    """Outward unit direction at each end, least squares over the terminal min(k, len) pixels.

    Returns (d_start, d_end), each pointing OUT of the run at that end.
    """
    def fit(seg, tip):
        seg = seg.astype(float)
        if len(seg) < 2:
            return np.array([0.0, 0.0])
        u, s, vt = np.linalg.svd(seg - seg.mean(0))
        d = vt[0]
        away = tip - seg.mean(0)
        return d if float(d @ away) >= 0 else -d
    kk = min(k, len(p))
    return fit(p[:kk][::-1], p[0].astype(float)), fit(p[-kk:], p[-1].astype(float))


def fibres(mask, max_turn_deg=_TURN_DEG, min_len_px=MIN_SEG_PX):
    """Stitch skeleton runs through junctions into whole fibres.

    Returns dict(fib_count, fib_len_um, fib_censored_frac).
    """
    sk = skeletonize(np.asarray(mask, bool))
    if sk.sum() == 0:
        return dict(fib_count=0, fib_len_um=np.nan, fib_censored_frac=np.nan)
    deg = ndi.convolve(sk.astype(int), _N8, mode="constant") - 1
    jmask = sk & (deg >= 3)
    # step 3: EXACTLY structure()'s segmentation, so the two statistics stay comparable
    lab, n = ndi.label(sk & (deg < 3), structure=_N8)
    if n == 0:
        return dict(fib_count=0, fib_len_um=np.nan, fib_censored_frac=np.nan)
    jlab, njun = ndi.label(jmask, structure=_N8)

    H, W = sk.shape
    objs = ndi.find_objects(lab)
    runs = {}
    for i in range(1, n + 1):
        sl = objs[i - 1]
        loc = np.argwhere(lab[sl] == i)
        pts = loc + np.array([sl[0].start, sl[1].start])
        p = _order_run(pts)
        d0, d1 = _end_dir(p)
        runs[i] = dict(pts=p, length=_path_len(p), d=(d0, d1),
                       tips=(p[0], p[-1]),
                       border=bool((p[:, 0] == 0).any() or (p[:, 1] == 0).any()
                                   or (p[:, 0] == H - 1).any() or (p[:, 1] == W - 1).any()))

    # which junction blob (if any) each run END touches
    att = {}
    for i, r in runs.items():
        for e, tip in enumerate(r["tips"]):
            js = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = tip[0] + dr, tip[1] + dc
                    if 0 <= rr < H and 0 <= cc < W and jlab[rr, cc]:
                        js.add(int(jlab[rr, cc]))
            att[(i, e)] = js
    incident = {j: [] for j in range(1, njun + 1)}
    for (i, e), js in att.items():
        for j in js:
            incident[j].append((i, e))
    jsize = np.bincount(jlab.ravel(), minlength=njun + 1)

    used, fib = set(), []
    for i in sorted(runs, key=lambda k: -runs[k]["length"]):     # longest first, stated rule
        if i in used:
            continue
        used.add(i)
        members, length, jpix = [i], runs[i]["length"], 0.0
        for e0 in (1, 0):                       # extend from each end of the seed run
            cur, ce = i, e0
            while True:
                cand = [(k, ee) for j in att[(cur, ce)] for (k, ee) in incident[j]
                        if k not in used]
                if not cand:
                    break
                d_in = runs[cur]["d"][ce]        # travel direction INTO the junction
                best, bturn = None, 1e9
                for (k, ee) in cand:
                    d_out = -runs[k]["d"][ee]    # travel direction leaving along k
                    cs_ = float(np.clip(d_in @ d_out, -1, 1))
                    t = abs(np.degrees(np.arccos(cs_)))
                    if t < bturn:
                        best, bturn = (k, ee), t
                if best is None or bturn > max_turn_deg:
                    break
                j_used = next(iter(att[(cur, ce)] & att[best]), None)
                if j_used is not None:
                    jpix += float(jsize[j_used])
                k, ee = best
                used.add(k)
                members.append(k)
                length += runs[k]["length"]
                cur, ce = k, 1 - ee
        fib.append(dict(length=length + jpix,
                        border=any(runs[k]["border"] for k in members)))

    L = np.array([f["length"] for f in fib])
    B = np.array([f["border"] for f in fib])
    keep = L >= min_len_px
    tot = float(L.sum())
    return dict(fib_count=int(keep.sum()),
                fib_len_um=float(L[keep].mean() * SCALE_UM) if keep.any() else np.nan,
                fib_censored_frac=float(L[B].sum() / tot) if tot > 0 else np.nan)
