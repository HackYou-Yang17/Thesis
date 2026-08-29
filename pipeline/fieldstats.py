"""
Analysis module for everything except FFT/dominance.

Covers analyses A2–A5 of the analysis plan:

  A2  parameter sensitivity / identifiability   -> anova_table, tukey, plot_tornado,
                                                   plot_interaction_heatmap
  A3  boundary-free fibre density vs time       -> local_area_fraction, gap_percentiles,
                                                   plot_density_vs_time
  A4  monomer phase diagram + critical ratio    -> fit_hill, phase_grid, ratio_collapse_test,
                                                   plot_phase_diagram, plot_hill
  A5  aspect ratio (height sweep)               -> uses the A2/A3 primitives

Design notes
------------
* Everything that touches images is DIMENSION-AGNOSTIC (2D or 3D), so the ImageJ
  segmentations can be analysed as z-stacks without a second code path. Physical
  scale is passed as `scale` = units per pixel, scalar or per-axis tuple.
* No model is imported. Functions take arrays and dataframes.
* Width normalisation is offered explicitly (`normalise_width`) because raw area
  fraction is not comparable between sim and real unless fibre width matches.
* Colour convention: orange = nematic, green = quartic. The categorical pair is
  CVD-validated (worst-case protan dE 9.3, normal-vision dE 22.7). The orange sits
  below 3:1 against the surface, so every figure using it carries a legend or a
  direct label -- do not remove them.

Install
-------
    pip install numpy scipy scikit-image pandas statsmodels matplotlib tifffile

NOTE: the package is `scikit-image`, not `skimage`. `skimage` is the IMPORT name;
`pip install skimage` pulls a dummy placeholder package that fails on purpose with
"Please install the `scikit-image` package (instead of `skimage`)".

If pip reports an externally-managed environment, either use a venv
(`python -m venv .venv && source .venv/bin/activate`) or pass
`--break-system-packages`.

Self-check after installing:
    python fieldstats.py          # expect "20/20 checks passed"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────

NEMATIC = "#E0961F"   # orange  — established convention
QUARTIC = "#1C9C6B"   # green   — established convention
REAL    = "#4F7CD0"   # blue    — real/observed data
SIM     = "#9B6BD0"   # purple  — simulated data

INK        = "#1a1d21"
INK_2      = "#4a5158"
INK_MUTED  = "#8b9298"
SURFACE    = "#fcfcfb"
GRID       = "#e3e5e8"

# sequential ramp, single hue, light -> dark (magnitude encoding)
SEQ = LinearSegmentedColormap.from_list(
    "field_seq", ["#f2f6fd", "#c3d4f0", "#8fb0e2", "#5b87d1", "#3560ab", "#1d3a6e"]
)

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def use_style():
    """Apply the module's rcParams. Called automatically by every plot_* function."""
    matplotlib.rcParams.update(_RC)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def load_imagej(path, reduce_z=None, allow_unit_scale=False):
    """Load an ImageJ TIFF and read its PIXEL SIZE from the file rather than
    trusting a hardcoded value.

    A wrong `scale` silently shifts the frequency band -- and therefore every
    dominance number -- with no error raised anywhere. Since your exports vary in
    pixel resolution, read it per file.

    UNCALIBRATED FILES ARE THE TRAP. A TIFF with no calibration still carries
    XResolution = (1, 1), so a naive reader returns 1.0 units/px and everything
    downstream is quietly wrong. This function treats a scale of exactly 1.0 with
    no physical unit as uncalibrated and raises. Pass allow_unit_scale=True only if
    your data really is 1 unit per pixel (e.g. simulation output in lattice units).

    reduce_z : None | 'mid' | 'max' | (z0, z1)
        The sim is 2D, so a z-stack must be reduced before dominance analysis.
        'mid' takes the middle section, 'max' a full max-projection, a tuple a
        fixed-thickness projection. Apply the SAME choice to every timepoint.

    Returns (array, scale) in MICRONS per pixel: (y, x), or (z, y, x) if kept 3D.
    """
    import tifffile
    _TO_UM = {"um": 1.0, "µm": 1.0, "micron": 1.0, "microns": 1.0,
              "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6, "inch": 25400.0}

    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        tags = tf.pages[0].tags
        ij = tf.imagej_metadata or {}

        def _res(name):
            t = tags.get(name)
            if t is None:
                return None
            v = t.value
            num, den = v if isinstance(v, tuple) else (v, 1)
            return (den / num) if num else None

        sx, sy, sz = _res("XResolution"), _res("YResolution"), ij.get("spacing")
        unit = str(ij.get("unit", "") or "").strip().lower()

    if sx is None or sy is None:
        raise ValueError(f"{path}: no resolution tags at all — set the scale in "
                         f"ImageJ (Image > Properties) before exporting.")

    uncalibrated = (unit in ("", "pixel", "pixels")) or (
        abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9 and unit not in _TO_UM)
    if uncalibrated and not allow_unit_scale:
        raise ValueError(
            f"{path}: appears UNCALIBRATED (scale={sx:g}, unit={unit or 'none'!r}). "
            f"An uncalibrated TIFF still reports XResolution=(1,1), so reading it "
            f"would silently give 1.0 um/px and shift the frequency band without "
            f"any error. Set Image > Properties in ImageJ and re-export, pass "
            f"`scale` explicitly, or use allow_unit_scale=True if 1 unit/px is "
            f"genuinely correct (e.g. simulation lattice units).")

    k = _TO_UM.get(unit, 1.0)
    sx, sy = sx * k, sy * k
    sz = sz * k if sz is not None else None

    if arr.ndim == 3 and reduce_z is not None:
        if reduce_z == "mid":
            arr = arr[arr.shape[0] // 2]
        elif reduce_z == "max":
            arr = arr.max(axis=0)
        else:
            z0, z1 = reduce_z
            arr = arr[int(z0):int(z1)].max(axis=0)

    if arr.ndim == 3:
        if sz is None:
            raise ValueError(f"{path}: 3D stack but no z-spacing in the ImageJ metadata")
        return arr, (float(sz), float(sy), float(sx))
    return arr, (float(sy), float(sx))


def binarise(img, threshold=None, method="otsu", min_object_px=0, fill_holes=False):
    """Binarise a greyscale fibre image.

    Parameters
    ----------
    img : ndarray            greyscale, any dimensionality
    threshold : float        explicit threshold; overrides `method`
    method : str             'otsu' | 'li' | 'yen' | 'mean' — used when threshold is None
    min_object_px : int      remove connected components smaller than this
    fill_holes : bool        fill enclosed holes (rarely wanted for fibres)

    Returns
    -------
    mask : bool ndarray

    NOTE: the threshold is a free knob that directly sets area fraction. Never
    report a single-threshold density curve — use `threshold_band` instead.
    """
    from skimage import filters, morphology

    img = np.asarray(img, dtype=float)
    if threshold is None:
        fn = {"otsu": filters.threshold_otsu,
              "li": filters.threshold_li,
              "yen": filters.threshold_yen,
              "mean": filters.threshold_mean}[method]
        threshold = float(fn(img))
    mask = img > threshold

    if min_object_px > 0:
        try:
            mask = morphology.remove_small_objects(mask, min_size=int(min_object_px))
        except TypeError:                      # skimage >= 2.0 renamed the kwarg
            mask = morphology.remove_small_objects(mask, int(min_object_px))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    return mask


def threshold_band(img, method="otsu", spread=0.25, n=5, **kw):
    """Binarise at a range of thresholds around the automatic one.

    Returns a list of (threshold, mask). Use this to produce the *band* that
    every real-image density figure must show, rather than a single curve.
    `spread` is fractional: 0.25 -> 0.75x to 1.25x the automatic threshold.
    """
    from skimage import filters
    img = np.asarray(img, dtype=float)
    fn = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
          "yen": filters.threshold_yen, "mean": filters.threshold_mean}[method]
    t0 = float(fn(img))
    out = []
    for f in np.linspace(1.0 - spread, 1.0 + spread, n):
        out.append((t0 * f, binarise(img, threshold=t0 * f, **kw)))
    return out


def mean_fibre_width(mask, scale=1.0):
    """Mean fibre width in physical units, from the skeleton + distance transform.

    Width at a skeleton point is (2*EDT - 1) pixels, NOT 2*EDT: for a band of odd
    width w the centre pixel's distance to the nearest background pixel is
    (w+1)/2, so the naive 2*EDT overestimates every width by exactly one pixel.
    Verified in self_test against bands of known width.

    Use this to CHECK that sim and real fibre widths match before comparing any
    area fraction — an area-fraction difference driven by a width mismatch is a
    rendering artefact, not biology.
    """
    from skimage.morphology import skeletonize
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.nan
    skel = skeletonize(mask)
    samp = _sampling(mask.ndim, scale)
    dist_px = ndi.distance_transform_edt(mask)            # in pixels
    vals = dist_px[skel]
    if not vals.size:
        return np.nan
    width_px = 2.0 * float(np.mean(vals)) - 1.0
    return width_px * float(np.mean(samp))


def normalise_width(mask, target_width_px):
    """Skeletonise, then re-dilate to a fixed width. Removes fibre width as a
    confound entirely, so the resulting area fraction reflects fibre LENGTH
    density and is directly comparable between sim and real.

    This is the recommended preprocessing for every A3 comparison. Measuring width
    with `mean_fibre_width` and matching by hand is the fallback if you would
    rather not skeletonise.
    """
    from skimage.morphology import skeletonize
    mask = np.asarray(mask, bool)
    skel = skeletonize(mask)
    r = max(0.0, (float(target_width_px) - 1.0) / 2.0)
    if r <= 0:
        return skel
    dist = ndi.distance_transform_edt(~skel)
    return dist <= r


def tissue_mask(img, scale, smooth_um=4.0, method="li", close_um=6.0,
                keep_largest=True):
    """Where the tissue is. REQUIRED for real images; pair with min_coverage=1.0.

    Two distinct things go wrong without it, and only the second is serious:

      * pure background windows carry no oriented power, gate out, and merely
        inflate frac_gated so that check reports on the crop rather than the tissue;
      * windows straddling the TISSUE EDGE are the hazard. The myocardial boundary
        is a long bright arc — one strong sharp orientation — so such a window reads
        as a near-perfect single family, biasing p UPWARD, preferentially wherever
        the outline happens to sit, which differs at every timepoint.

    Measured on 61.5 um crops: tissue covers 74% (32 hpf) and 58% (48 hpf), so a
    quarter to nearly half of every crop is outside the myocardium.

    Deliberately coarse — heavy smoothing at cell scale, global threshold, closing,
    hole fill. It answers "is there tissue here", not "is there a fibre here", and
    must NOT track fibre density or it becomes the circular denominator that A3 was
    reframed to avoid.
    """
    from skimage import filters, morphology
    img = np.asarray(img, float)
    samp = _sampling(img.ndim, scale)
    sm = ndi.gaussian_filter(img, sigma=[smooth_um / s for s in samp])
    fn = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
          "yen": filters.threshold_yen, "mean": filters.threshold_mean}[method]
    m = sm > float(fn(sm))
    r = max(1, int(round(close_um / float(np.mean(samp)))))
    m = morphology.closing(m, morphology.disk(r))
    m = ndi.binary_fill_holes(m)
    if keep_largest:
        lab, n = ndi.label(m)
        if n > 1:
            m = lab == (int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1)
    return m


def _sampling(ndim, scale):
    """Normalise a scalar-or-tuple physical scale into a per-axis sampling tuple."""
    if np.isscalar(scale):
        return tuple([float(scale)] * ndim)
    s = tuple(float(v) for v in scale)
    if len(s) != ndim:
        raise ValueError(f"scale has {len(s)} entries for a {ndim}-D image")
    return s


# ═════════════════════════════════════════════════════════════════════════════
# 2.  A3 — BOUNDARY-FREE DENSITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def local_area_fraction(mask, window, scale=1.0, stride=None, tissue_mask=None,
                        min_coverage=1.0):
    """Fibre area (or volume) fraction in fixed-size windows.

    Needs NO cell boundaries — this is the replacement for "mesh completeness",
    which is unmeasurable on the real images because the fibre and boundary
    channels are not co-registered.

    Parameters
    ----------
    mask : bool ndarray      binarised fibre image (2D or 3D)
    window : float           window edge length in PHYSICAL units (e.g. 28.8 um)
    scale : float | tuple    physical units per pixel
    stride : float | None    step between windows in physical units; defaults to
                             window/2 (50% overlap)
    tissue_mask : bool ndarray | None
                             region considered valid; windows must be at least
                             `min_coverage` covered by it
    min_coverage : float     fraction of the window that must lie inside tissue

    Returns
    -------
    fracs : 1D float array   one value per accepted window
    """
    mask = np.asarray(mask, bool)
    samp = _sampling(mask.ndim, scale)
    wpx = [max(1, int(round(window / s))) for s in samp]
    spx = ([max(1, int(round((stride if stride is not None else window / 2.0) / s)))
            for s in samp])

    if any(w > n for w, n in zip(wpx, mask.shape)):
        raise ValueError(
            f"window {window} phys units = {wpx} px does not fit in image {mask.shape}")

    starts = [range(0, n - w + 1, st) for n, w, st in zip(mask.shape, wpx, spx)]
    out = []
    for origin in _nd_product(starts):
        sl = tuple(slice(o, o + w) for o, w in zip(origin, wpx))
        if tissue_mask is not None:
            cov = float(np.asarray(tissue_mask, bool)[sl].mean())
            if cov < min_coverage:
                continue
        out.append(float(mask[sl].mean()))
    return np.asarray(out, dtype=float)


def gap_percentiles(mask, scale=1.0, percentiles=(50, 90, 95), tissue_mask=None,
                    border_margin=None):
    """Distance-to-nearest-fibre distribution, in physical units.

    The Euclidean distance transform of the EMPTY space. The upper percentiles
    capture anomalously large voids, which is the actual "the mesh does not fill
    the cell" failure mode.

    Two things that will bite if ignored:

    1. The SMALL-gap end is induced by exclusion_len, which sets minimum fibre
       separation, so it is not evidence. Report the UPPER tail only (90th/95th),
       which the exclusion rule does not set.

    2. BORDER ARTEFACT — the distance transform has no fibres to find beyond the
       image or tissue edge, so edge pixels get spuriously large distances and can
       dominate the upper tail entirely. On a test volume with bounding planes the
       p95 is 9.5 px; remove the far bounding plane and the same structure reads
       17 px, an 80% inflation from the edge alone. ALWAYS pass `border_margin`
       (in physical units, >= the expected gap scale) for real images, or the p95
       measures how you cropped rather than how the mesh is built.

    Returns
    -------
    dict {percentile: distance in physical units}
    """
    mask = np.asarray(mask, bool)
    samp = _sampling(mask.ndim, scale)
    if not mask.any():
        return {p: np.nan for p in percentiles}
    dist = ndi.distance_transform_edt(~mask, sampling=samp)

    sel = ~mask
    if tissue_mask is not None:
        sel = sel & np.asarray(tissue_mask, bool)

    if border_margin is not None:
        valid = np.asarray(tissue_mask, bool) if tissue_mask is not None \
            else np.ones(mask.shape, bool)
        # Pad with False so the IMAGE edge counts as outside too. Without the pad,
        # an all-True valid region has no background for the transform to find and
        # the margin silently excludes nothing.
        valid_p = np.pad(valid, 1, mode="constant", constant_values=False)
        inner = ndi.distance_transform_edt(valid_p, sampling=samp)
        inner = inner[tuple(slice(1, -1) for _ in range(mask.ndim))]
        sel = sel & (inner >= float(border_margin))

    vals = dist[sel]
    if vals.size == 0:
        return {p: np.nan for p in percentiles}
    return {p: float(np.percentile(vals, p)) for p in percentiles}


def density_summary(mask, window, scale=1.0, tissue_mask=None, percentiles=(50, 90, 95)):
    """Both A3 measures in one call. Returns a flat dict, one row of a dataframe."""
    fr = local_area_fraction(mask, window, scale=scale, tissue_mask=tissue_mask)
    gp = gap_percentiles(mask, scale=scale, percentiles=percentiles, tissue_mask=tissue_mask)
    row = {
        "n_windows": int(fr.size),
        "area_frac_mean": float(np.mean(fr)) if fr.size else np.nan,
        "area_frac_sd": float(np.std(fr, ddof=1)) if fr.size > 1 else np.nan,
        "area_frac_p25": float(np.percentile(fr, 25)) if fr.size else np.nan,
        "area_frac_p75": float(np.percentile(fr, 75)) if fr.size else np.nan,
        "fibre_width": mean_fibre_width(mask, scale=scale),
    }
    for p, v in gp.items():
        row[f"gap_p{p}"] = v
    return row


def _nd_product(ranges):
    """Cartesian product of N ranges, yielding tuples. Avoids itertools import noise."""
    if not ranges:
        yield ()
        return
    head, rest = ranges[0], ranges[1:]
    for h in head:
        for tail in _nd_product(rest):
            yield (h,) + tail


# ═════════════════════════════════════════════════════════════════════════════
# 3.  A4 — HILL FIT AND CRITICAL RATIO
# ═════════════════════════════════════════════════════════════════════════════

def hill(x, y0, ymax, K, n):
    """Hill function. K is the half-maximal (critical) value, n the cooperativity.

    n ~ 1  -> a dial (graded response)
    n >> 1 -> a switch
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        xn = np.power(np.clip(x, 1e-12, None), n)
        Kn = np.power(max(K, 1e-12), n)
        return y0 + (ymax - y0) * xn / (Kn + xn)


@dataclass
class HillFit:
    y0: float
    ymax: float
    K: float             # critical ratio
    n: float             # Hill coefficient — the switch-vs-dial number
    K_ci: tuple = (np.nan, np.nan)
    n_ci: tuple = (np.nan, np.nan)
    r2: float = np.nan
    n_points: int = 0
    n_in_transition: int = 0
    warnings: list = field(default_factory=list)

    def __repr__(self):
        return (f"HillFit(K={self.K:.3g} [{self.K_ci[0]:.3g},{self.K_ci[1]:.3g}], "
                f"n={self.n:.3g} [{self.n_ci[0]:.3g},{self.n_ci[1]:.3g}], "
                f"R2={self.r2:.3f}, pts_in_transition={self.n_in_transition})")


def fit_hill(x, y, n_boot=1000, seed=0):
    """Fit a Hill function and bootstrap CIs on K (critical value) and n (sharpness).

    This turns "switch, not dial" into a number with an interval. It also reports
    how many sampled points fall inside the transition — a threshold can look
    infinitely sharp purely because the sweep was too coarse, and with fewer than
    ~4 points between 10% and 90% of the response you cannot distinguish a Hill
    coefficient of 4 from one of 40.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        raise ValueError("need at least 4 finite points to fit a Hill function")

    order = np.argsort(x)
    x, y = x[order], y[order]

    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo if hi > lo else 1.0
    p0 = [lo, hi, float(np.median(x)), 2.0]
    bounds = ([lo - span, lo, x.min() * 1e-3, 0.1],
              [hi, hi + span, x.max() * 1e3, 200.0])

    warns = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(hill, x, y, p0=p0, bounds=bounds, maxfev=200_000)

    yhat = hill(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    y10 = popt[0] + 0.1 * (popt[1] - popt[0])
    y90 = popt[0] + 0.9 * (popt[1] - popt[0])
    n_trans = int(np.sum((yhat >= y10) & (yhat <= y90)))
    if n_trans < 4:
        warns.append(
            f"only {n_trans} sampled points lie inside the 10-90% transition; "
            "the sweep is too coarse to bound the Hill coefficient — resample "
            "finely around K before interpreting sharpness")
    if popt[3] > 150:
        warns.append("Hill coefficient at the fit bound — treat as 'unresolvably sharp', "
                     "not as a measured value")

    rng = np.random.default_rng(seed)
    Ks, ns = [], []
    resid = y - yhat
    for _ in range(int(n_boot)):
        yb = yhat + rng.choice(resid, size=resid.size, replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pb, _ = curve_fit(hill, x, yb, p0=popt, bounds=bounds, maxfev=50_000)
            Ks.append(pb[2]); ns.append(pb[3])
        except Exception:
            continue
    K_ci = (float(np.percentile(Ks, 2.5)), float(np.percentile(Ks, 97.5))) if len(Ks) > 20 else (np.nan, np.nan)
    n_ci = (float(np.percentile(ns, 2.5)), float(np.percentile(ns, 97.5))) if len(ns) > 20 else (np.nan, np.nan)

    return HillFit(y0=popt[0], ymax=popt[1], K=popt[2], n=popt[3],
                   K_ci=K_ci, n_ci=n_ci, r2=r2, n_points=int(x.size),
                   n_in_transition=n_trans, warnings=warns)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  A4 — PHASE DIAGRAM AND THE CONSTANT-RATIO TEST
# ═════════════════════════════════════════════════════════════════════════════

def phase_grid(df, x="monomers_per_seg", y="monomers_per_point", z="completeness",
               agg="mean"):
    """Pivot a long results dataframe into a 2D grid for the phase diagram.

    Returns (xs, ys, Z) with Z[j, i] corresponding to ys[j], xs[i].
    """
    p = df.pivot_table(index=y, columns=x, values=z, aggfunc=agg)
    return p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float)


def boundary_points(xs, ys, Z, level=0.5, relative=True):
    """Locate the phase boundary: for each column of the grid, the y value at which
    Z first crosses `level`, found by linear interpolation between grid rows.

    `relative=True` interprets `level` as a fraction of the grid's own max, which is
    the right choice when the 'on' state does not reach 1.0.

    Returns (x_at_boundary, y_at_boundary) as 1D arrays.
    """
    Z = np.asarray(Z, float)
    thr = level * np.nanmax(Z) if relative else level
    bx, by = [], []
    for i, xv in enumerate(xs):
        col = Z[:, i]
        good = np.isfinite(col)
        if good.sum() < 2:
            continue
        yy, cc = np.asarray(ys)[good], col[good]
        above = cc >= thr
        if not above.any() or above.all():
            continue           # no crossing inside the swept range
        k = int(np.argmax(above))          # first index at/above threshold
        if k == 0:
            continue
        y0, y1 = yy[k - 1], yy[k]
        c0, c1 = cc[k - 1], cc[k]
        t = 0.0 if c1 == c0 else (thr - c0) / (c1 - c0)
        bx.append(float(xv)); by.append(float(y0 + t * (y1 - y0)))
    return np.asarray(bx), np.asarray(by)


@dataclass
class RatioTest:
    slope: float               # critical ratio, if the through-origin model holds
    slope_ci: tuple
    intercept: float           # free-intercept model
    intercept_ci: tuple
    intercept_frac: float      # intercept as a fraction of the swept y range
    tol: float
    r2_through_origin: float
    r2_free: float
    verdict: str
    conclusive: bool

    def __repr__(self):
        return (f"RatioTest(critical_ratio={self.slope:.3g} "
                f"[{self.slope_ci[0]:.3g},{self.slope_ci[1]:.3g}], "
                f"intercept={self.intercept:.3g} "
                f"({100*self.intercept_frac:.2f}% of range), "
                f"verdict='{self.verdict}')")


def ratio_collapse_test(bx, by, tol=0.05):
    """Does the phase boundary pass through the origin?

    If the RATIO monomers_per_point / monomers_per_seg is the control parameter,
    the boundary in the (per_seg, per_point) plane is a straight line through the
    origin, and the two parameters collapse to one dimensionless group. If the
    boundary is meaningfully offset, one parameter is doing the work instead.

    EQUIVALENCE, NOT SIGNIFICANCE. An earlier version of this function tested the
    intercept against exactly zero and rejected the ratio on data generated from a
    pure ratio: grid interpolation leaves a systematic bias of ~0.15% of the swept
    range, which is statistically significant once the scatter is small, and
    practically meaningless. The test therefore asks whether the intercept is
    SMALL relative to the swept range, not whether it is distinguishable from zero.

    `tol` is that practical threshold as a fraction of the y range (default 5%).

    Three outcomes:
      * 'ratio'        — CI lies entirely inside +-tol; the ratio governs
      * 'not ratio'    — CI lies entirely outside +-tol; one parameter dominates
      * 'inconclusive' — CI straddles the tolerance; sweep more finely or wider
    """
    import statsmodels.api as sm
    bx = np.asarray(bx, float); by = np.asarray(by, float)
    if bx.size < 3:
        raise ValueError("need at least 3 boundary points for the collapse test")

    m0 = sm.OLS(by, bx[:, None]).fit()                       # through origin
    m1 = sm.OLS(by, sm.add_constant(bx)).fit()               # free intercept

    ci1 = m1.conf_int()
    inter, inter_ci = float(m1.params[0]), (float(ci1[0][0]), float(ci1[0][1]))
    ci0 = m0.conf_int()
    slope, slope_ci = float(m0.params[0]), (float(ci0[0][0]), float(ci0[0][1]))

    yr = float(np.max(by) - np.min(by)) or 1.0
    band = tol * yr
    frac = inter / yr

    inside = abs(inter_ci[0]) <= band and abs(inter_ci[1]) <= band
    outside = min(abs(inter_ci[0]), abs(inter_ci[1])) > band and \
        (inter_ci[0] * inter_ci[1] > 0)

    if inside:
        verdict = (f"intercept within +-{100*tol:.0f}% of the swept range -> boundary is "
                   f"consistent with a line through the origin; the RATIO is the "
                   f"control parameter (critical ratio {slope:.3g})")
        conclusive = True
    elif outside:
        verdict = (f"intercept is {100*frac:.1f}% of the swept range, beyond the "
                   f"+-{100*tol:.0f}% tolerance -> the boundary does NOT pass through the "
                   f"origin; the ratio framing is wrong and one parameter dominates")
        conclusive = True
    else:
        verdict = (f"intercept CI straddles the +-{100*tol:.0f}% tolerance -> INCONCLUSIVE; "
                   f"extend the sweep range or add boundary points before deciding")
        conclusive = False

    return RatioTest(slope=slope, slope_ci=slope_ci, intercept=inter,
                     intercept_ci=inter_ci, intercept_frac=frac, tol=tol,
                     r2_through_origin=float(m0.rsquared),
                     r2_free=float(m1.rsquared_adj),
                     verdict=verdict, conclusive=conclusive)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  A2 — ANOVA, TUKEY, EFFECT SIZES
# ═════════════════════════════════════════════════════════════════════════════

def anova_table(df, response, factors, interactions=True, typ=2):
    """Factorial ANOVA with partial eta-squared effect sizes.

    KEEP THE INTERACTION TERMS. One-at-a-time sweeps and main-effects-only models
    miss parameter trade-offs, and trade-offs are exactly what make a calibration
    non-unique. `interactions=True` adds all 2-way terms.

    Factors are treated as CATEGORICAL (C(...)), which is what makes Tukey valid on
    the swept levels. For a genuinely continuous predictor use a regression instead.

    Returns a dataframe with sum_sq, df, F, PR(>F), partial_eta_sq.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    d = df.copy()
    terms = [f"C(Q('{f}'))" for f in factors]
    if interactions and len(factors) > 1:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                terms.append(f"C(Q('{factors[i]}'))*C(Q('{factors[j]}'))")
    formula = f"Q('{response}') ~ " + " + ".join(terms)

    model = ols(formula, data=d).fit()
    tab = sm.stats.anova_lm(model, typ=typ)

    ss_resid = float(tab.loc["Residual", "sum_sq"])
    tab["partial_eta_sq"] = [
        (float(ss) / (float(ss) + ss_resid)) if idx != "Residual" else np.nan
        for idx, ss in zip(tab.index, tab["sum_sq"])
    ]
    tab.index = [_pretty_term(t) for t in tab.index]
    return tab


def _pretty_term(t):
    """Strip statsmodels' C(Q('x')) wrapping back to readable factor names."""
    import re
    t = re.sub(r"C\(Q\('([^']+)'\)\)", r"\1", str(t))
    return t.replace(":", " x ")


def tukey(df, response, factor, alpha=0.05):
    """Tukey HSD across the levels of one factor. Returns a tidy dataframe.

    Valid because the sweep levels are discrete. Run it only on factors the ANOVA
    flags as significant — running it on everything inflates the comparison count
    without adding information.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    d = df[[response, factor]].dropna()
    res = pairwise_tukeyhsd(d[response].to_numpy(float),
                            d[factor].astype(str).to_numpy(), alpha=alpha)
    out = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return out


def effect_ranges(df, response, factors):
    """Response range across each factor's levels, holding nothing fixed (marginal).

    This is what a tornado plot shows: how far the outcome moves when the parameter
    is swept over the range you actually sampled. Complements partial eta-squared,
    which is variance-explained rather than magnitude.

    A near-zero range means the response is FLAT in that parameter -- the parameter
    is unidentifiable from this outcome, and any value quoted for it is arbitrary.
    """
    rows = []
    grand = float(df[response].mean())
    for f in factors:
        g = df.groupby(f, observed=True)[response].mean()
        if g.size < 2:
            continue
        rows.append({
            "factor": f,
            "low": float(g.min()),
            "high": float(g.max()),
            "range": float(g.max() - g.min()),
            "low_level": g.idxmin(),
            "high_level": g.idxmax(),
            "grand_mean": grand,
        })
    return pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_tornado(eff, path, response_label="response", title=None,
                 eta=None, flat_threshold=None):
    """F2a — tornado plot of marginal effect ranges, ordered by magnitude.

    `eta` optionally maps factor -> partial eta squared, annotated on each bar.
    `flat_threshold` draws the line below which a parameter is called
    unidentifiable; pass the replicate noise level for a principled cut.
    """
    use_style()
    eff = eff.sort_values("range")
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(eff) + 1.9))

    grand = float(eff["grand_mean"].iloc[0])
    ypos = np.arange(len(eff))
    for k, (_, r) in enumerate(eff.iterrows()):
        ax.plot([r["low"], r["high"]], [k, k], color=REAL, lw=5.5,
                solid_capstyle="round")
        if eta is not None and r["factor"] in eta and np.isfinite(eta[r["factor"]]):
            ax.text(r["high"], k + 0.30, f"η²={eta[r['factor']]:.2f}",
                    va="center", ha="right", fontsize=8, color=INK_MUTED)

    ax.axvline(grand, color=INK_MUTED, lw=1.0, ls="--", zorder=0)
    ax.set_ylim(-0.7, len(eff) - 0.3)
    ax.text(grand, -0.55, " grand mean", fontsize=8,
            color=INK_MUTED, ha="left", va="center")

    if flat_threshold is not None:
        for k, (_, r) in enumerate(eff.iterrows()):
            if r["range"] < flat_threshold:
                ax.text(max(r["high"], grand), k, "  unidentifiable",
                        va="center", fontsize=8, color=INK_MUTED, style="italic")

    ax.set_yticks(ypos)
    ax.set_yticklabels(eff["factor"])
    ax.set_xlabel(response_label)
    ax.set_title(title or f"Marginal effect on {response_label}", fontweight="bold", loc="left")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_interaction_heatmap(tab, factors, path, title=None):
    """F2b — parameter x parameter interaction strength (partial eta squared).

    Non-zero off-diagonal cells are trade-offs: two parameters that compensate for
    each other, which is what makes a fit non-unique. This is the panel that
    justifies keeping interaction terms in the model.
    """
    use_style()
    n = len(factors)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                key = factors[i]
            else:
                key = f"{factors[i]} x {factors[j]}"
                alt = f"{factors[j]} x {factors[i]}"
                key = key if key in tab.index else alt
            if key in tab.index:
                M[i, j] = tab.loc[key, "partial_eta_sq"]

    fig, ax = plt.subplots(figsize=(1.05 * n + 3.0, 1.05 * n + 2.4))
    im = ax.imshow(M, cmap=SEQ, vmin=0.0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(factors, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(factors)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                shade = M[i, j] / (np.nanmax(M) or 1.0)
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if shade > 0.55 else INK)
    ax.set_title(title or "Interaction strength (partial η²)", fontweight="bold",
                 loc="left", pad=26)
    ax.text(0.0, 1.015, "diagonal = main effect · off-diagonal = 2-way interaction",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("partial η²", color=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_diagram(xs, ys, Z, path, bx=None, by=None, ratio=None, n_sub=None,
                       xlabel="monomers_per_seg", ylabel="monomers_per_point",
                       zlabel="mesh completeness", title=None, fontsize=12):
    """F4a — the monomer phase diagram, with the boundary and the constant-ratio line.

    If the ratio is the control parameter, the boundary is a straight line THROUGH
    THE ORIGIN. A boundary that misses the origin means one parameter dominates.

    `fontsize` sets a FLOOR applied to every text element — labels, ticks, legend,
    title, colourbar — rather than the module default of 10 with a 9 pt legend.
    Note the unit is POINTS, not pixels: at savefig.dpi = 200 one point is 2.78 px,
    so 12 pt rasterises to ~33 px. Nothing here renders below 12 pt.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.1))   # grown to match the larger type
    im = ax.pcolormesh(xs, ys, Z, cmap=SEQ, shading="auto", vmin=0.0,
                       vmax=float(np.nanmax(Z)) if np.isfinite(Z).any() else 1.0)

    if bx is not None and by is not None and len(bx):
        ax.plot(bx, by, "o", color=NEMATIC, ms=6, mec=SURFACE, mew=1.2,
                label="measured boundary", zorder=3)
    if ratio is not None:
        xr = np.linspace(0, float(np.max(xs)), 100)
        ax.plot(xr, ratio * xr, color=QUARTIC, lw=2.0, ls="--",
                label=f"constant ratio = {ratio:.2f}", zorder=2)
    xr = np.linspace(0, float(np.max(xs)), 100)
    ax.plot(xr, xr, color=INK_MUTED, lw=1.0, ls=":", label="ratio = 1 (per segment)",
            zorder=1)
    # WHICH reference line is the right one depends on what the availability gate
    # actually checks. An event draws at most one QUANTUM = monomers_per_seg/n_sub,
    # so if the gate tests the quantum rather than the whole segment, unit supply:
    # demand sits at monomers_per_point/monomers_per_seg = 1/n_sub, not 1. At
    # n_sub=4 that is 0.25 — and a critical ratio reported as "above 1" is then 4x
    # above the quantum-normalised expectation, not just above it. Resolve with
    # gate 0.1 before interpreting the offset.
    if n_sub:
        ax.plot(xr, xr / float(n_sub), color=INK_MUTED, lw=1.0, ls="-.",
                label=f"ratio = 1/n_sub = {1/float(n_sub):.2g} (per quantum)", zorder=1)

    ax.set_xlim(float(np.min(xs)), float(np.max(xs)))
    ax.set_ylim(float(np.min(ys)), float(np.max(ys)))
    ax.set_xlabel(xlabel, fontsize=fontsize); ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_title(title or "Monomer phase diagram", fontweight="bold", loc="left",
                 fontsize=fontsize + 2)
    # WHITE legend text here and ONLY here. This legend is drawn over the pcolormesh,
    # and the upper-left corner is the high-completeness end of the sequential ramp
    # (SEQ tops out at #1d3a6e), so the default near-black label text sits on dark
    # navy and is barely legible. Every other plot_* legend sits on SURFACE, where
    # white would be invisible — so this is deliberately not a global rcParam.
    leg = ax.legend(loc="upper left", fontsize=fontsize)
    for t in leg.get_texts():
        t.set_color("#ffffff")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(zlabel, color=INK_2, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_hill(x, y, fit, path, xlabel="monomers_per_point / monomers_per_seg",
              ylabel="mesh completeness", title=None, label=None, ax=None):
    """F4b — the Hill fit, with K and n annotated and coarse-sweep warnings shown."""
    use_style()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
    xs = np.linspace(float(np.min(x)) * 0.9, float(np.max(x)) * 1.05, 400)
    ax.plot(xs, hill(xs, fit.y0, fit.ymax, fit.K, fit.n), color=REAL, lw=2.0,
            label=label or f"Hill fit  n={fit.n:.2f}, K={fit.K:.3g}")
    ax.plot(x, y, "o", color=INK_2, ms=6, mec=SURFACE, mew=1.0, ls="none",
            label="simulated runs")
    ax.axvline(fit.K, color=NEMATIC, lw=1.4, ls="--")
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls=":")

    # annotate against axis fractions so labels never collide with the data,
    # the legend, or each other regardless of the fitted values
    ax.annotate(f"K={fit.K:.3g}", xy=(fit.K, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -4), textcoords="offset points",
                color=NEMATIC, fontsize=9, va="top", ha="left")
    ax.annotate("ratio=1", xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -18), textcoords="offset points",
                color=INK_MUTED, fontsize=8, va="top", ha="left")

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or "Critical ratio and sharpness", fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    _despine(ax)

    # Warnings go BELOW the axes, not inside them — a sigmoid leaves no reliably
    # empty interior region, so any in-axes placement collides for some fit.
    if fit.warnings:
        import textwrap
        msg = "\n".join(textwrap.fill("! " + w, 92) for w in fit.warnings)
        ax.figure.text(0.01, -0.02, msg, fontsize=7.4, color=INK_MUTED,
                       va="top", ha="left")

    if own:
        ax.figure.savefig(path)
        plt.close(ax.figure)
    return path


def plot_hill_vs_nsub(nsubs, fits, path, title=None):
    """F4b inset — Hill coefficient vs n_sub, the conjunction-artefact discriminator.

    Falling n with rising n_sub  -> the switch was update granularity.
    Flat n                       -> the threshold is dynamical, not an artefact.
    """
    use_style()
    n = np.array([f.n for f in fits], float)
    lo = np.array([f.n_ci[0] for f in fits], float)
    hi = np.array([f.n_ci[1] for f in fits], float)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.fill_between(nsubs, lo, hi, color=REAL, alpha=0.16, lw=0)
    ax.plot(nsubs, n, "o-", color=REAL, mec=SURFACE, mew=1.0)
    ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=":")
    ax.text(float(np.max(nsubs)), 1.0, " n=1 (dial)", fontsize=8,
            color=INK_MUTED, va="bottom", ha="right")
    ax.set_xlabel("n_sub  (1 = atomic, higher = graded)")
    ax.set_ylabel("Hill coefficient")
    ax.set_title(title or "Is the switch update granularity?", fontweight="bold", loc="left")
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_density_vs_time(real_df, sim_df, path, value="area_frac_mean",
                         lo=None, hi=None, time="hpf", ylabel=None, title=None):
    """F3 — density (or gap size) vs developmental time, sim vs real, both as BANDS.

    The real band spans binarisation thresholds; the sim band spans replicate runs.
    Never plot either as a single line: the real threshold is a free knob and the
    sim is stochastic, and a bare line implies a precision neither has.

    `lo`/`hi` name the band columns; defaults to value+'_lo' / value+'_hi'.
    """
    use_style()
    lo = lo or value + "_lo"
    hi = hi or value + "_hi"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for df, col, name in ((real_df, REAL, "real"), (sim_df, SIM, "simulated")):
        if df is None or not len(df):
            continue
        d = df.sort_values(time)
        if lo in d and hi in d:
            ax.fill_between(d[time], d[lo], d[hi], color=col, alpha=0.18, lw=0)
        ax.plot(d[time], d[value], "o-", color=col, mec=SURFACE, mew=1.0, label=name)

    ax.set_xlabel("developmental time (hpf)")
    ax.set_ylabel(ylabel or value.replace("_", " "))
    ax.set_title(title or "Fibre density over time", fontweight="bold", loc="left")
    ax.legend(loc="best", fontsize=9)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_critical_vs_knob(knob_vals, Ks, K_cis, path, knob_label,
                          expected=None, title=None):
    """F4b main panels — critical ratio vs each discriminating knob.

    Three knobs separate the three explanations for a threshold above ratio 1:

        knob                 conjunction   kinetic Cc   depletion
        n_sub increasing     K -> 1        flat         flat
        depoly:poly ratio    flat          tracks       flat
        cell size / branch   flat          flat          shifts

    `expected` optionally overlays the prediction of one hypothesis for reference.
    """
    use_style()
    Ks = np.asarray(Ks, float)
    lo = np.array([c[0] for c in K_cis], float)
    hi = np.array([c[1] for c in K_cis], float)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.fill_between(knob_vals, lo, hi, color=REAL, alpha=0.16, lw=0)
    ax.plot(knob_vals, Ks, "o-", color=REAL, mec=SURFACE, mew=1.0, label="measured K")
    if expected is not None:
        ax.plot(knob_vals, expected, ls="--", color=QUARTIC, lw=1.8, label="predicted")
    ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=":")
    ax.text(float(np.max(knob_vals)), 1.0, " ratio=1", fontsize=8,
            color=INK_MUTED, va="bottom", ha="right")
    ax.set_xlabel(knob_label)
    ax.set_ylabel("critical ratio K")
    ax.set_title(title or f"Critical ratio vs {knob_label}", fontweight="bold", loc="left")
    ax.legend(loc="best", fontsize=9)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 7.  SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

def _synth_lines(shape=(400, 400), pitch=20, width=3, axis=0, offset=0):
    """Parallel lines of known pitch and width — ground truth for the density metrics."""
    m = np.zeros(shape, bool)
    if axis == 0:
        for r in range(offset, shape[0], pitch):
            m[r:r + width, :] = True
    else:
        for c in range(offset, shape[1], pitch):
            m[:, c:c + width] = True
    return m


def self_test(verbose=True):
    """Validate every metric against a case where the answer is known analytically."""
    ok = []

    def check(name, got, want, tol):
        good = abs(got - want) <= tol
        ok.append(good)
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.4g}, want ~{want:.4g}")

    print("area fraction — parallel lines, pitch 20 px, width 3 px")
    m = _synth_lines(pitch=20, width=3)
    fr = local_area_fraction(m, window=100, scale=1.0)
    check("mean area fraction", float(fr.mean()), 3 / 20, 0.02)

    print("area fraction — physical scale honoured (0.5 units/px, 50-unit window)")
    fr2 = local_area_fraction(m, window=50, scale=0.5)
    check("mean area fraction, scaled", float(fr2.mean()), 3 / 20, 0.02)

    print("fibre width")
    check("mean width", mean_fibre_width(m, scale=1.0), 3.0, 0.6)

    print("width normalisation collapses two widths to one")
    a = normalise_width(_synth_lines(pitch=20, width=3), 3)
    b = normalise_width(_synth_lines(pitch=20, width=9), 3)
    fa = float(local_area_fraction(a, window=100).mean())
    fb = float(local_area_fraction(b, window=100).mean())
    check("width-normalised fractions agree", fa - fb, 0.0, 0.02)

    print("gap percentiles — max gap between lines of pitch 20, width 3 is ~8.5 px half-gap")
    gp = gap_percentiles(m, scale=1.0, percentiles=(95,))
    check("p95 gap", gp[95], 8.5, 1.5)

    print("gap percentiles — 3D works (planes at 0,20,40,60 bound every gap)")
    m3 = np.zeros((61, 40, 40), bool)
    m3[::20, :, :] = True
    gp3 = gap_percentiles(m3, scale=1.0, percentiles=(95,))
    check("3D p95 gap", gp3[95], 9.5, 1.0)

    print("border artefact — an unbounded edge inflates the tail, border_margin fixes it")
    m4 = np.zeros((60, 40, 40), bool)
    m4[::20, :, :] = True          # planes at 0,20,40 — region 41..59 is unbounded
    naive = gap_percentiles(m4, scale=1.0, percentiles=(95,))[95]
    fixed = gap_percentiles(m4, scale=1.0, percentiles=(95,), border_margin=12.0)[95]
    inflated = naive > 1.5 * fixed
    ok.append(bool(inflated))
    print(f"  [{'PASS' if inflated else 'FAIL'}] naive p95={naive:.3g} inflated vs "
          f"border-corrected p95={fixed:.3g}")

    print("Hill fit — recover known n and K")
    xs = np.linspace(0.2, 4.0, 40)
    ys = hill(xs, 0.0, 1.0, 1.7, 8.0) + np.random.default_rng(1).normal(0, 0.01, xs.size)
    f = fit_hill(xs, ys, n_boot=200)
    check("Hill K", f.K, 1.7, 0.08)
    check("Hill n", f.n, 8.0, 1.5)

    print("Hill fit — a dial is not read as a switch")
    ys2 = hill(xs, 0.0, 1.0, 1.7, 1.0) + np.random.default_rng(2).normal(0, 0.01, xs.size)
    f2 = fit_hill(xs, ys2, n_boot=200)
    check("Hill n for a dial", f2.n, 1.0, 0.4)

    print("coarse-sweep warning fires")
    xc = np.array([0.2, 0.5, 1.6, 1.8, 3.0, 4.0])
    yc = hill(xc, 0.0, 1.0, 1.7, 40.0)
    fc = fit_hill(xc, yc, n_boot=100)
    ok.append(bool(fc.warnings))
    print(f"  [{'PASS' if fc.warnings else 'FAIL'}] warnings raised: {len(fc.warnings)}")

    print("ratio collapse — boundary through the origin is detected")
    seg = np.linspace(1, 10, 12)
    pt = 1.8 * seg + np.random.default_rng(3).normal(0, 0.05, seg.size)
    rt = ratio_collapse_test(seg, pt)
    ok.append("RATIO is the\n                   control" in rt.verdict or
              "RATIO is the control" in rt.verdict)
    check("recovered ratio", rt.slope, 1.8, 0.1)
    print(f"  [{'PASS' if 'RATIO is the' in rt.verdict else 'FAIL'}] verdict: {rt.verdict[:58]}...")

    print("ratio collapse — an offset boundary is rejected")
    pt2 = 1.8 * seg + 4.0 + np.random.default_rng(4).normal(0, 0.05, seg.size)
    rt2 = ratio_collapse_test(seg, pt2)
    good = "does NOT pass" in rt2.verdict
    ok.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] offset boundary rejected")

    print("ratio collapse — a REAL grid-derived boundary from a pure-ratio model is "
          "accepted (regression test: significance-testing the intercept failed here)")
    segs = np.linspace(1, 10, 19)
    pts = np.linspace(1, 20, 25)
    Zg = np.array([[1.0 / (1.0 + (1.8 * s / q) ** 12) for s in segs] for q in pts])
    gbx, gby = boundary_points(segs, pts, Zg, level=0.5)
    rtg = ratio_collapse_test(gbx, gby)
    good = "RATIO is the" in rtg.verdict
    ok.append(good)
    check("grid-derived critical ratio", rtg.slope, 1.8, 0.05)
    print(f"  [{'PASS' if good else 'FAIL'}] pure-ratio grid accepted "
          f"(intercept {100*rtg.intercept_frac:.2f}% of range)")

    print("ANOVA — an inert factor gets a near-zero effect size")
    rng = np.random.default_rng(5)
    N = 240
    d = pd.DataFrame({
        "alpha": rng.choice([1, 2, 3], N),
        "inert": rng.choice([1, 2, 3], N),
    })
    d["y"] = 0.5 * d["alpha"] + rng.normal(0, 0.1, N)
    tab = anova_table(d, "y", ["alpha", "inert"])
    check("eta2 for the real factor", float(tab.loc["alpha", "partial_eta_sq"]), 0.95, 0.06)
    check("eta2 for the inert factor", float(tab.loc["inert", "partial_eta_sq"]), 0.0, 0.06)

    print("effect ranges rank correctly")
    eff = effect_ranges(d, "y", ["alpha", "inert"])
    good = eff.iloc[0]["factor"] == "alpha"
    ok.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] tornado ordering")

    print("\nFFT dominance controls")
    dom_ok = dominance_controls(verbose=verbose)
    ok.append(dom_ok)

    n_pass, n_tot = sum(ok), len(ok)
    print(f"\n{n_pass}/{n_tot} checks passed")
    return n_pass == n_tot



# ═════════════════════════════════════════════════════════════════════════════
# 8.  FFT DOMINANCE  —  nematic vs quartic, by mixture decomposition
# ═════════════════════════════════════════════════════════════════════════════
#
# WHY NOT m2/(m2+m4)
# ------------------
# A single sharp family has m2 = m4 = 1 and therefore scores 0.50 on that ratio --
# exactly the crossover value -- despite being perfectly nematic. Worse, broadening
# one family makes the ratio CLIMB above 0.5, because a_k = I_k/I_0 decays faster
# for k=4 than k=2. So the ratio tracks fibre waviness, not family structure. It is
# the same objection that killed the raw S4 parameter.
#
# THE MIXTURE DECOMPOSITION
# -------------------------
# Model the orientation distribution as: a fraction p of the oriented power in ONE
# family, and (1-p) in TWO families 90 deg apart. Write a single family's harmonic
# content as a2(kappa), a4(kappa) for a von Mises of concentration kappa in the
# doubled angle.
#
# Two orthogonal families CANCEL in the 2nd harmonic (e^{i2(mu+90deg)} = -e^{i2mu})
# and ADD in the 4th (e^{i4(mu+90deg)} = +e^{i4mu}). Therefore
#
#       m2 = p * a2(kappa)
#       m4 =     a4(kappa)          <-- independent of p
#
# So: measure m4, invert a4 to get kappa, evaluate a2(kappa), and
#
#       p = m2 / a2(kappa)          nematic fraction
#       1 - p                       quartic fraction
#
# Angular spread cancels exactly. A single family reads p = 1 at ANY spread; two
# orthogonal families read p = 0 at any spread. That is the property the ratio
# lacked.
#
# THE THINGS THAT MANUFACTURE A FALSE CROSSOVER
# ---------------------------------------------
# 1. A SQUARE window. Rectangular edges put a hard cross into the power spectrum,
#    which is 4-fold symmetric and reads as quartic. Even a separable Hann (outer
#    product of 1D windows) is square-ish. This module uses a RADIAL window.
# 2. The PIXEL LATTICE itself is 4-fold symmetric and biases m4 upward more than
#    m2. Uncorrected, that pushes every window toward "quartic".
# 3. A floor estimated GLOBALLY. Both moments are positively biased at finite
#    sample size, and the bias depends on the window's own SNR and radial content.
#    If image quality varies across timepoints -- and yours does -- a global floor
#    turns an imaging trend into a biological one. The floor here is estimated PER
#    WINDOW by angular shuffling within radial annuli, which preserves the radial
#    spectrum and the lattice geometry and destroys only the orientation structure.
# 4. Applying the +-25 deg vertical suppression at SOME timepoints only. Correcting
#    one end of a trend and then reporting the trend is circular. `suppress_deg` is
#    off by default and `dominance_vs_time` runs it both ways.


def _radial_window(shape, alpha=0.5):
    """Radially symmetric Tukey window.

    A separable Hann (np.hanning outer product) is square-symmetric and leaks a
    4-fold cross into the spectrum, which reads as spurious quartic power. This
    window depends only on radius, so it has no preferred direction and cannot
    create angular structure.
    """
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = np.sqrt(sum(g ** 2 for g in grids))
    w = np.ones_like(r)
    taper = r > (1.0 - alpha)
    w[taper] = 0.5 * (1 + np.cos(np.pi * (r[taper] - (1 - alpha)) / alpha))
    w[r >= 1.0] = 0.0
    return w


def _windowed_power(win):
    """Radially-windowed power spectrum with correct DC removal.

    Removing the mean and THEN windowing leaves a residual DC component, because
    w*(x - mean(x)) does not sum to zero. Subtracting the mean again would add a
    constant across the whole array and break the taper at the edges, reintroducing
    the very discontinuity the window exists to remove. The correct form keeps the
    taper: w*(x - c) with c = sum(w*x)/sum(w), which sums to zero by construction.
    """
    win = np.asarray(win, float)
    w = _radial_window(win.shape)
    sw = w.sum()
    c = float((w * win).sum() / sw) if sw > 0 else float(win.mean())
    x = w * (win - c)
    return np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2


def _fft_geometry(shape, scale, band):
    """Radius, real-space fibre angle, and the HALF-PLANE band mask.

    HALF-PLANE. The power spectrum of a real image is centrosymmetric,
    P(-f) = P(f), and a Hermitian pair shares the same orientation mod 180, so both
    members land in the same angular bin and contribute identically. Keeping only
    one member of each pair leaves every normalised moment unchanged, halves the
    work, and -- critically -- makes the surrogate shuffle correct: shuffling the
    full plane treats the two members of a pair as independent, which gives the
    surrogate more effective samples than the real spectrum has and UNDER-estimates
    the floor. Under-subtracting leaves residual positive bias in both moments, and
    the residue is larger for m4, so it tilts every window toward quartic.
    """
    samp = _sampling(2, scale)
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=samp[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=samp[1]))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)

    # -FY converts image row order (downward) to standard orientation (upward);
    # +90 deg converts the spectral direction to the real-space fibre direction,
    # because the frequency vector of a set of parallel fibres is PERPENDICULAR
    # to them.
    TH = (np.degrees(np.arctan2(-FY, FX)) + 90.0) % 180.0

    half = (FY > 0) | ((FY == 0) & (FX > 0))      # one member of each Hermitian pair
    sel = (R >= band[0]) & (R <= band[1]) & half
    return R, TH, sel, (abs(fy[1] - fy[0]), abs(fx[1] - fx[0]))


def _bin_angles(TH_sel, nbins):
    edges = np.linspace(0, 180, nbins + 1)
    idx = np.clip(np.digitize(TH_sel, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    angles = 0.5 * (edges[:-1] + edges[1:])
    return idx, cnt, angles


def _suppress(angles, prof, suppress_deg):
    if not suppress_deg:
        return prof
    d = np.minimum(np.abs(angles - 90.0), 180.0 - np.abs(angles - 90.0))
    return np.where(d <= suppress_deg, np.nan, prof)


def angular_profile(win, scale, band=(0.05, 0.8), nbins=180, suppress_deg=None):
    """Angular power profile of one window, in REAL-SPACE fibre orientation.

    Parameters
    ----------
    win : 2D ndarray       one image window (greyscale or binary)
    scale : float|tuple    physical units per pixel (e.g. um/px)
    band : (lo, hi)        radial band in CYCLES PER PHYSICAL UNIT, not per pixel.
                           Fixing the band physically is what makes real (0.417
                           um/px) and sim (0.362 um/lu) measure the same feature
                           sizes.
    suppress_deg : float|None
                           if set, blank the profile within +-suppress_deg of
                           vertical. DECLARE IT IF YOU USE IT -- see module notes.

    Returns
    -------
    angles_deg : (nbins,)  bin centres, 0..180, real-space fibre orientation
    power      : (nbins,)  mean power per bin (COUNT-NORMALISED)
    """
    win = np.asarray(win, float)
    if win.ndim != 2:
        raise ValueError("angular_profile expects a 2D window")

    P = _windowed_power(win)
    R, TH, sel, _ = _fft_geometry(win.shape, scale, band)
    if sel.sum() < 32:
        raise ValueError(
            f"only {int(sel.sum())} Fourier samples in band {band} cycles/unit — "
            f"window too small or band too narrow for this pixel size")

    idx, cnt, angles = _bin_angles(TH[sel], nbins)
    tot = np.bincount(idx, weights=P[sel], minlength=nbins)
    # COUNT-NORMALISE. Summing raw power per bin weights by however many lattice
    # points land in each bin, which imprints the grid's preferred directions.
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return angles, _suppress(angles, prof, suppress_deg)


def circular_moments(angles_deg, power):
    """m2 and m4 of an angular power profile. Returns (m2, m4, dir2_deg, dir4_deg).

    m_k = |sum P(phi) e^{i k phi}| / sum P(phi), with phi the fibre orientation.
    """
    a = np.asarray(angles_deg, float)
    p = np.asarray(power, float)
    ok = np.isfinite(p) & (p >= 0)
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan
    a, p = a[ok], p[ok]
    tot = p.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan, np.nan
    phi = np.radians(a)
    z2 = np.sum(p * np.exp(2j * phi)) / tot
    z4 = np.sum(p * np.exp(4j * phi)) / tot
    return (float(abs(z2)), float(abs(z4)),
            float(np.degrees(np.angle(z2)) / 2.0 % 180.0),
            float(np.degrees(np.angle(z4)) / 4.0 % 90.0))


def shuffle_floor(win, scale, band=(0.05, 0.8), nbins=180, n_surrogates=24,
                  seed=0, suppress_deg=None):
    """Per-window null floor for m2 and m4, by angular shuffling within annuli.

    Permutes power among the pixels of each radial annulus (within the half-plane,
    so Hermitian pairs are not double-counted). This keeps the radial spectrum AND
    the lattice sampling geometry exactly, and destroys only angular structure --
    so the resulting moments are what THIS window would read with no orientational
    order at all.

    Better than a white-noise floor, which has a flat radial spectrum the real image
    does not have, and better than a global floor, which cannot track the
    per-timepoint SNR changes across an image set.

    Uses exactly the same windowing, DC removal and geometry as angular_profile --
    an earlier version computed the floor on a slightly different spectrum, which
    made the subtraction inconsistent with the signal.

    Returns (floor_m2, floor_m4).
    """
    win = np.asarray(win, float)
    rng = np.random.default_rng(seed)

    P = _windowed_power(win)
    R, TH, sel, (dfy, dfx) = _fft_geometry(win.shape, scale, band)
    if sel.sum() < 32:
        return np.nan, np.nan

    rsel, psel = R[sel], P[sel]
    dr = float(min(dfy, dfx))                      # annuli at native resolution
    rbin = np.round(rsel / dr).astype(int)
    idx, cnt, angles = _bin_angles(TH[sel], nbins)

    groups = [np.flatnonzero(rbin == rb) for rb in np.unique(rbin)]
    groups = [g for g in groups if g.size > 1]

    m2s, m4s = [], []
    for _ in range(int(n_surrogates)):
        shuffled = psel.copy()
        for g in groups:
            shuffled[g] = shuffled[rng.permutation(g)]
        tot = np.bincount(idx, weights=shuffled, minlength=nbins)
        prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        a2m, a4m, _, _ = circular_moments(angles, _suppress(angles, prof, suppress_deg))
        if np.isfinite(a2m):
            m2s.append(a2m); m4s.append(a4m)

    if not m2s:
        return np.nan, np.nan
    return float(np.mean(m2s)), float(np.mean(m4s))


def _a2_of_kappa(k):
    from scipy.special import ive
    return float(ive(1, k) / ive(0, k))


def _a4_of_kappa(k):
    from scipy.special import ive
    return float(ive(2, k) / ive(0, k))


def kappa_from_a4(a4):
    """Invert a4 = I_2(kappa)/I_0(kappa). Monotone on (0,1), so brentq is safe."""
    from scipy.optimize import brentq
    a4 = float(a4)
    if not np.isfinite(a4) or a4 <= 0:
        return 0.0
    if a4 >= 0.999999:
        return 1e6
    return float(brentq(lambda k: _a4_of_kappa(k) - a4, 1e-9, 1e7, xtol=1e-10))


def nematic_fraction(m2, m4):
    """Mixture decomposition -> nematic fraction p. Quartic fraction is 1 - p.

    Returns (p, kappa, spread_deg). p is clipped to [0, 1]; a raw value far outside
    that range means the mixture model does not describe the window (three families,
    or a floor over-subtraction) and the window should be discarded, so the
    UNCLIPPED value is worth inspecting when debugging.
    """
    if not (np.isfinite(m2) and np.isfinite(m4)) or m4 <= 0:
        return np.nan, np.nan, np.nan
    k = kappa_from_a4(min(m4, 0.999999))
    a2 = _a2_of_kappa(k)
    if a2 <= 1e-9:
        return np.nan, k, np.nan
    p = m2 / a2
    # circular sd of the doubled angle, reported in orientation degrees
    spread = np.degrees(np.sqrt(-2.0 * np.log(max(_a2_of_kappa(k), 1e-12)))) / 2.0
    return float(np.clip(p, 0.0, 1.0)), float(k), float(spread)


def window_dominance(win, scale, band=(0.05, 0.8), nbins=180, floor_scale=1.0,
                     n_surrogates=24, seed=0, suppress_deg=None, m4_gate=0.05):
    """Full dominance measurement for a single window. Returns a dict (one row)."""
    angles, prof = angular_profile(win, scale, band=band, nbins=nbins,
                                   suppress_deg=suppress_deg)
    m2, m4, d2, d4 = circular_moments(angles, prof)
    f2, f4 = shuffle_floor(win, scale, band=band, nbins=nbins,
                           n_surrogates=n_surrogates, seed=seed,
                           suppress_deg=suppress_deg)

    # QUADRATURE, NOT LINEAR, floor subtraction.
    #
    # m_k is the magnitude of a resultant vector. The unoriented part of the power
    # does not push that vector in a fixed direction — it adds an independent
    # random vector of expected magnitude floor_k. Magnitudes of independent
    # vectors combine in quadrature, so the debiased estimate is
    #
    #     m_k_corrected = sqrt(max(0, m_k_raw^2 - floor_k^2))
    #
    # Subtracting linearly (the original method, at 1.1x) over-corrects badly when
    # the signal is strong. On perfect parallel lines it turned m2=0.999, m4=0.994
    # into m2=0.818, m4=0.848 — INVERTING their order, because floor_m2 (0.180)
    # exceeds floor_m4 (0.146). That drove p from ~0.99 down to 0.85. The bias runs
    # toward QUARTIC at every timepoint, which is the direction that manufactures a
    # nematic->quartic crossover. Quadrature subtraction removes it.
    fs = float(floor_scale)
    m2c = float(np.sqrt(max(0.0, m2 ** 2 - (fs * f2) ** 2))) if np.isfinite(m2) and np.isfinite(f2) else np.nan
    m4c = float(np.sqrt(max(0.0, m4 ** 2 - (fs * f4) ** 2))) if np.isfinite(m4) and np.isfinite(f4) else np.nan

    gated = (not np.isfinite(m4c)) or (m4c < m4_gate)
    p, k, spread = (np.nan, np.nan, np.nan) if gated else nematic_fraction(m2c, m4c)

    # SNR diagnostics. p is SENSITIVE TO IMAGE QUALITY -- additive noise dilutes
    # the oriented power and pushes p down (toward quartic), and binarising does
    # NOT recover it. On a single-family texture, p fell 0.91 -> 0.43 as noise rose,
    # and binarising first gave 0.33, slightly worse. Since your image clarity
    # varies across timepoints, record these and check they do not track p --
    # see confound_report().
    P = _windowed_power(win)
    R, _, sel, _ = _fft_geometry(win.shape, scale, band)
    tot_p = float(P[R > 0].sum())
    band_frac = float(P[sel].sum() / tot_p) if tot_p > 0 else np.nan
    snr_ratio = float(m4 / f4) if (np.isfinite(f4) and f4 > 0) else np.nan

    return {"m2_raw": m2, "m4_raw": m4, "floor_m2": f2, "floor_m4": f4,
            "m2": m2c, "m4": m4c, "nematic_frac": p, "quartic_frac": (1 - p) if p == p else np.nan,
            "kappa": k, "spread_deg": spread, "dir2_deg": d2, "dir4_deg": d4,
            "band_frac": band_frac, "snr_ratio": snr_ratio,
            "gated": bool(gated)}


def dominance_map(img, window, scale, stride=None, tissue_mask=None,
                  min_coverage=1.0, **kw):
    """Per-window dominance across a whole image. Returns a tidy DataFrame.

    LOCAL WINDOWS. The original rule ("not optional") was established on 166-192 um
    crops, where one FFT averages many cells with different axes and the moments
    cancel before they are taken. At a 61.5 um crop that argument is weaker --
    whole-crop and windowed agreed within 0.10 on six of eight real sub-crops -- so
    the case now rests on three measured failures rather than an assertion:

      1. one crop in eight gated out ENTIRELY (whole-crop m4 = 0.000 against a 0.05
         gate; whole-crop m4 runs only 0.09-0.24). One FFT is one throw;
      2. placement sensitivity 4.3x worse (SD 0.157 vs 0.037 across four sub-crops);
      3. a synthetic of 25 um domains, each internally ONE family (true p = 0.93),
         reads 0.494 whole-crop against 0.919 windowed -- landing exactly on the
         crossover value. Few-domain crops are where the error is largest, and
         31 x 13 um cells in a 61.5 um crop are exactly that.

    Window size: 15 um on the current data, set by the >=15-INDEPENDENT-window floor
    (2L/5 = 24.6 um at L = 61.5), not fitted to an m2 plateau -- no real crop has
    shown one. 28.8 um was inherited and is no longer used.

    Known confound to state with any result: a window straddling two cells with
    different axes reads as two families even if each cell is purely nematic. Some
    of the early "quartic" signal may be inter-cell axis variation.

    STRONGER AT 32 HPF, measured: the F-actin signal there is dominated by cell
    CORTEX, not myofibrils -- high-p windows each contain one bright cell boundary,
    low-p windows each sit on a junction where boundaries meet -- so p reads the
    boundary honeycomb. Pass a tissue_mask with min_coverage=1.0 in every case:
    windows on the tissue EDGE read the myocardial outline as one sharp family and
    bias p upward, at a location that differs per timepoint.
    """
    img = np.asarray(img, float)
    if img.ndim != 2:
        raise ValueError("dominance_map expects a 2D image; reduce a z-stack first")
    samp = _sampling(2, scale)
    wpx = [max(8, int(round(window / s))) for s in samp]
    spx = [max(1, int(round((stride if stride is not None else window / 2.0) / s)))
           for s in samp]
    if any(w > n for w, n in zip(wpx, img.shape)):
        raise ValueError(f"window {window} = {wpx} px does not fit image {img.shape}")

    rows = []
    for i, y in enumerate(range(0, img.shape[0] - wpx[0] + 1, spx[0])):
        for j, x in enumerate(range(0, img.shape[1] - wpx[1] + 1, spx[1])):
            sl = (slice(y, y + wpx[0]), slice(x, x + wpx[1]))
            if tissue_mask is not None:
                if float(np.asarray(tissue_mask, bool)[sl].mean()) < min_coverage:
                    continue
            try:
                r = window_dominance(img[sl], scale, seed=(i * 7919 + j), **kw)
            except ValueError:
                continue
            r.update({"iy": i, "ix": j, "y": y, "x": x})
            rows.append(r)
    return pd.DataFrame(rows)


def _circ_sd_deg(dirs_deg):
    """Circular standard deviation of axial (mod-180) directions, in degrees."""
    d = np.asarray(dirs_deg, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    Rbar = abs(np.mean(np.exp(2j * np.radians(d))))
    Rbar = min(max(Rbar, 1e-12), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(Rbar))) / 2.0)


def window_size_scan(img, scale, sizes, band=(0.05, 0.8), n_surrogates=8, **kw):
    """Choose the dominance window from the DATA, not from a guess at cell size.

    You do not need cell boundaries for this, which matters because no image has
    both boundaries and fibres. The requirement is not "one cell" -- that was only
    ever a proxy. It is that the window be smaller than the ORIENTATION DOMAIN: a
    window spanning two regions with different axes averages them, and the moments
    cancel before they are measured. That domain scale is visible in the fibre
    channel by itself.

    Sweep window size and watch three columns:

      m2_med      collapses once windows exceed the domain scale (domains average out)
      frac_gated  high at small sizes (too few Fourier samples, noise-dominated)
      n_windows   falls as size grows; below ~15 the distribution is not usable

    Pick a size in the PLATEAU between those two failure modes. If there is no
    plateau, the crop is too small or too heterogeneous for this measure and that
    is worth knowing before exporting the full set.

    THE CROP CAN PIN THE WINDOW, in which case this scan is documentation rather
    than a choice. At a 61.5 um field, requiring >= 15 INDEPENDENT (non-overlapping)
    windows caps the window at 2L/5 = 24.6 um, and 15 um is the largest size that
    clears it comfortably (49 overlapping / 16 independent). No real crop in this
    dataset has shown a plateau. Say the window was set by the count floor rather
    than implying it was fitted to a knee.

    Note also that 50%-overlap window COUNTS are not sample sizes: 25 overlapping
    windows at 20 um are ~9 independent ones. The overlap buys denser position
    sampling, not extra n.

    Whatever you choose, apply the same PHYSICAL size to sim and real.

    Returns a DataFrame, one row per window size.
    """
    rows = []
    for w in sizes:
        try:
            dm = dominance_map(img, window=float(w), scale=scale, band=band,
                               n_surrogates=n_surrogates, **kw)
        except ValueError:
            rows.append({"window": float(w), "n_windows": 0, "frac_gated": np.nan,
                         "m2_med": np.nan, "m4_med": np.nan, "p_med": np.nan,
                         "dir2_dispersion": np.nan})
            continue
        if len(dm) == 0:
            # No window was accepted at this size — with a tissue_mask and
            # min_coverage=1.0 this happens as soon as the window outgrows the
            # tissue. Record the size as unusable rather than raising, so the
            # scan still shows where the usable range ends.
            rows.append({"window": float(w), "n_windows": 0, "frac_gated": np.nan,
                         "m2_med": np.nan, "m4_med": np.nan, "p_med": np.nan,
                         "dir2_dispersion": np.nan})
            continue
        good = dm[~dm["gated"]]
        rows.append({
            "window": float(w),
            "n_windows": int(len(dm)),
            "frac_gated": float(1 - len(good) / len(dm)) if len(dm) else np.nan,
            "m2_med": float(good["m2"].median()) if len(good) else np.nan,
            "m4_med": float(good["m4"].median()) if len(good) else np.nan,
            "p_med": float(good["nematic_frac"].median()) if len(good) else np.nan,
            "dir2_dispersion": _circ_sd_deg(good["dir2_deg"].to_numpy(float))
            if len(good) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_window_scan(scan, path, chosen=None, title=None):
    """Companion figure for window_size_scan. Three stacked panels sharing an x axis."""
    use_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.4), sharex=True)
    for ax, col, lab, col_c in (
            (axes[0], "m2_med", "median m2 (oriented signal)", REAL),
            (axes[1], "frac_gated", "fraction of windows gated", NEMATIC),
            (axes[2], "n_windows", "windows per crop", INK_2)):
        ax.plot(scan["window"], scan[col], "o-", color=col_c, mec=SURFACE, mew=1.0)
        ax.set_ylabel(lab, fontsize=9)
        if chosen:
            ax.axvline(chosen, color=INK_2, lw=1.2, ls="--")
        _despine(ax)
    axes[1].axhline(0.35, color=INK_MUTED, lw=1.0, ls=":")
    axes[1].annotate("0.35 — above this the band or window is wrong",
                     xy=(0.99, 0.35), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)
    axes[2].axhline(15, color=INK_MUTED, lw=1.0, ls=":")
    axes[2].annotate("15 — below this the distribution is not usable",
                     xy=(0.99, 15), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)
    axes[2].set_xlabel("window size (physical units)")
    axes[0].set_title(title or "Choosing the dominance window from the data",
                      fontweight="bold", loc="left")
    fig.savefig(path)
    plt.close(fig)
    return path


def dominance_summary(dm, label=None):
    """Collapse a dominance_map to one row. Median + IQR across ungated windows.

    Median, not mean: the per-window distribution is bounded on [0,1] and skewed,
    and a handful of near-degenerate windows drag a mean around.
    """
    if len(dm) == 0 or "gated" not in dm.columns:
        # Empty map: with a tissue_mask and min_coverage=1.0 this happens whenever no
        # window fits entirely inside the tissue. That is a fact about the crop, not
        # an error, so return a NaN row rather than raising and losing the batch.
        return {"label": label, "n_windows": 0, "n_used": 0, "frac_gated": np.nan,
                **{k: np.nan for k in
                   ("nematic_frac", "nematic_frac_lo", "nematic_frac_hi",
                    "quartic_frac", "spread_deg", "dir2_deg", "dir2_dispersion",
                    "band_frac", "snr_ratio", "m4")}}
    good = dm[~dm["gated"]].dropna(subset=["nematic_frac"])
    row = {"label": label,
           "n_windows": int(len(dm)),
           "n_used": int(len(good)),
           "frac_gated": float(1 - len(good) / len(dm)) if len(dm) else np.nan}
    if len(good):
        v = good["nematic_frac"].to_numpy(float)
        # circular medians of the directors, so a ROTATING family can be told apart
        # from a SECOND family appearing — see the note below.
        d2 = np.degrees(np.angle(np.mean(np.exp(2j * np.radians(good["dir2_deg"].to_numpy(float)))))) / 2 % 180
        row.update({"nematic_frac": float(np.median(v)),
                    "nematic_frac_lo": float(np.percentile(v, 25)),
                    "nematic_frac_hi": float(np.percentile(v, 75)),
                    "quartic_frac": float(1 - np.median(v)),
                    "spread_deg": float(good["spread_deg"].median()),
                    "dir2_deg": float(d2),
                    "band_frac": float(good["band_frac"].median()),
                    "snr_ratio": float(good["snr_ratio"].median()),
                    # CIRCULAR dispersion — a linear std would call 179 deg and
                    # 1 deg 178 apart when they are 2 apart
                    "dir2_dispersion": float(_circ_sd_deg(good["dir2_deg"].to_numpy(float))),
                    "m4": float(good["m4"].median())})
    else:
        row.update({k: np.nan for k in
                    ("nematic_frac", "nematic_frac_lo", "nematic_frac_hi",
                     "quartic_frac", "spread_deg", "dir2_deg", "dir2_dispersion",
                     "band_frac", "snr_ratio", "m4")})
    return row


# ─────────────────────────────────────────────────────────────────────────────
# READ THIS BEFORE INTERPRETING A DOMINANCE CURVE
#
# Quartic dominance means TWO COEXISTING FAMILIES. It does not mean "vertical".
# A single family at 90 deg is exactly as nematic as a single family at 0 deg.
#
# So a tissue whose single family merely ROTATES from horizontal to vertical
# produces nematic -> quartic -> nematic: a peak, not a plateau, with the quartic
# maximum at the 50/50 midpoint of the rotation. Verified on a synthetic rotation
# series, which read p = 0.82, 0.62, 0.17, 0.36, 0.59, 0.68 — a dip and recovery,
# not a transition.
#
# A genuine nematic->quartic transition (one family, then two) instead gives a
# MONOTONIC fall in p that stays low.
#
# The two are distinguished by `dir2_deg` in the summary: if the director rotates
# across timepoints while p dips and recovers, you are watching a rotation. If the
# director holds steady while p falls and stays down, a second family is appearing.
# Always plot dir2_deg alongside p before claiming a transition.
# ─────────────────────────────────────────────────────────────────────────────


def confound_report(df, p_col="nematic_frac", verbose=True,
                    range_tol=0.20, r_tol=0.5):
    """Does the dominance trend track image quality rather than biology?

    NOT OPTIONAL BEFORE CLAIMING A CROSSOVER. Measured on synthetic textures with
    KNOWN, CONSTANT orientation structure:

      * noise   -- p fell 0.91 -> 0.43 as additive noise rose. This is real signal
                   dilution, and binarising first does NOT fix it (it gave 0.33).
      * density -- p fell 0.97 -> 0.93 across a 6x density range at realistic area
                   fractions (0.06-0.35); the drift only reaches -0.15 near
                   saturation. Within one image, per-window density did not drive p
                   at all (r = +0.04).

    Both confounds push p DOWNWARD, i.e. toward quartic -- the direction of the
    hypothesis. So the failure mode is a false positive, not a false negative.

    A CORRELATION ALONE PROVES NOTHING. Any two monotone series over six timepoints
    correlate. Validated against a real transition and an SNR-driven fake, both
    gave r > 0.97 with band_frac. Two things actually separate them:

    1. HOW FAR the nuisance variable moved. Real: band_frac 0.425 -> 0.404 (5%),
       far too small to explain a 0.8 swing in p. Fake: 0.423 -> 0.154 (64%),
       easily enough.
    2. THE SIGN of r(p, snr_ratio). Adding a genuine second family RAISES 4-fold
       structure while p falls, so a real transition gives a NEGATIVE correlation
       (measured -0.79). An acquisition artefact degrades everything together, so
       it gives a POSITIVE one (measured +0.98). Positive is the danger sign.
    """
    res = {}
    for c in ("band_frac", "snr_ratio", "frac_gated", "area_frac_mean", "n_used"):
        if c in df and df[c].notna().sum() >= 3:
            sub = df[[p_col, c]].dropna()
            if len(sub) < 3 or sub[c].std() == 0:
                continue
            v = sub[c].to_numpy(float)
            lo, hi = float(np.min(v)), float(np.max(v))
            res[c] = {"r": float(np.corrcoef(sub[p_col], v)[0, 1]),
                      "lo": lo, "hi": hi,
                      "frac_range": float((hi - lo) / abs(hi)) if hi else np.nan}

    verdicts = []
    bf = res.get("band_frac")
    if bf and bf["frac_range"] > range_tol:
        verdicts.append(
            f"band_frac moved {100*bf['frac_range']:.0f}% across timepoints "
            f"(> {100*range_tol:.0f}% tolerance) — the acquisition changed enough to "
            f"account for part of the trend on its own")
    fg = res.get("frac_gated")
    if fg and (fg["hi"] - fg["lo"]) > 0.25:
        verdicts.append(
            f"gating rate swings {100*fg['lo']:.0f}%–{100*fg['hi']:.0f}% across "
            f"timepoints — DIFFERENTIAL GATING. Windows are dropped when m4 falls "
            f"below the gate, which happens to disordered tissue. If early "
            f"timepoints gate more heavily, the surviving early windows are the "
            f"most ORDERED subsample, biasing early p upward and manufacturing a "
            f"crossover. Report p on a common gating rate, or report the gated "
            f"fraction beside every point")

    sr = res.get("snr_ratio")
    if sr and sr["r"] > r_tol:
        verdicts.append(
            f"r(p, snr_ratio) = {sr['r']:+.2f} is POSITIVE — dominance and 4-fold "
            f"structure are degrading together, the signature of an acquisition "
            f"artefact rather than a second family appearing")
    if sr and sr["r"] < -0.2:
        verdicts.append(
            f"r(p, snr_ratio) = {sr['r']:+.2f} is negative — 4-fold structure RISES "
            f"as p falls, which is what a genuine second family looks like")

    if verbose:
        print("confound check")
        for c, d in res.items():
            print(f"   {c:16s} r={d['r']:+.3f}   range {d['lo']:.3f}–{d['hi']:.3f} "
                  f"({100*d['frac_range']:.0f}%)")
        if not res:
            print("   nothing to check — record band_frac / snr_ratio per timepoint")
        for v in verdicts:
            print(f"   * {v}")
        if not verdicts:
            print("   * no nuisance variable moved enough to explain the trend")
    return {"stats": res, "verdicts": verdicts}


def dominance_vs_time(images, scale, window, suppress_deg=None,
                      suppress_upto_hpf=36.0, **kw):
    """Dominance at each timepoint.

    `window` has NO DEFAULT. It used to default to 28.8 um, which was inherited
    from a reconstructed process and never fitted; a default that silently carries
    a stale number into every call is worse than an error. Choose it per dataset
    (see window_size_scan, and the note there about the window-count floor).

    `suppress_deg` now defaults to None — NO ANGULAR SUPPRESSION. It was tested on
    matched 61.5 um crops and dropped: it RAISED p at both ends (0.584 -> 0.664 at
    32 hpf, 0.497 -> 0.777 at 48 hpf) because removing one of two families makes
    the remainder look single-family, and it drove frac_gated to 0.44-0.50. Applied
    to the early end only the 32-48 gap doubled; applied symmetrically it CHANGED
    SIGN. The direction of the result was decided by which end was corrected.
    Set suppress_deg explicitly if you want the variants; the reason must then be
    biological and stated, and a spatial mask is the better instrument anyway
    because angle cannot separate an ECM rail from a myofibril running the same way.

    With suppression on, still computed BOTH WITH AND WITHOUT, because applying a
    correction to only the early timepoints and then reporting a trend across
    timepoints is circular.

    `images` : list of (hpf, 2D array) or dict {hpf: array}

    Returns a DataFrame with one row per timepoint per variant
    ('none' / 'early_only' / 'all'). Report the 'none' variant as primary and the
    others as sensitivity, or justify in the text why a corrected variant is the
    right primary -- but never show only the corrected one.
    """
    items = sorted(images.items()) if isinstance(images, dict) else sorted(images)
    variants = ("none",) if not suppress_deg else ("none", "early_only", "all")
    rows = []
    for hpf, img in items:
        for variant in variants:
            sd = (None if variant == "none"
                  else suppress_deg if variant == "all"
                  else (suppress_deg if hpf <= suppress_upto_hpf else None))
            dm = dominance_map(img, window, scale, suppress_deg=sd, **kw)
            r = dominance_summary(dm, label=f"{hpf}hpf")
            r.update({"hpf": float(hpf), "variant": variant})
            rows.append(r)
    return pd.DataFrame(rows)


def plot_dominance_vs_time(df, path, per_window=None, title=None,
                           anchors=None, variant="none"):
    """F1 — nematic and quartic fraction vs developmental time.

    Orange = nematic, green = quartic (established convention; the pair is
    CVD-validated). `anchors` marks the timepoints used for calibration so
    held-out points are visibly distinguished from fitted ones.

    `per_window` optionally supplies {hpf: array of per-window p} to draw the
    distribution behind the medians — the spread across windows is the real
    uncertainty and a bare median hides it.
    """
    use_style()
    d = df[df["variant"] == variant].sort_values("hpf") if "variant" in df else df.sort_values("hpf")
    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    if per_window:
        for hpf, vals in sorted(per_window.items()):
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            if v.size:
                parts = ax.violinplot([v], positions=[float(hpf)], widths=1.8,
                                      showextrema=False, showmedians=False)
                for b in parts["bodies"]:
                    b.set_facecolor(INK_MUTED); b.set_alpha(0.16); b.set_edgecolor("none")

    ax.fill_between(d["hpf"], d["nematic_frac_lo"], d["nematic_frac_hi"],
                    color=NEMATIC, alpha=0.16, lw=0)
    ax.plot(d["hpf"], d["nematic_frac"], "o-", color=NEMATIC, mec=SURFACE, mew=1.0,
            label="nematic  p")
    ax.fill_between(d["hpf"], 1 - d["nematic_frac_hi"], 1 - d["nematic_frac_lo"],
                    color=QUARTIC, alpha=0.16, lw=0)
    ax.plot(d["hpf"], 1 - d["nematic_frac"], "o-", color=QUARTIC, mec=SURFACE, mew=1.0,
            label="quartic  1−p")

    ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls=":")
    xc = crossover_time(d)
    if np.isfinite(xc):
        ax.axvline(xc, color=INK_2, lw=1.2, ls="--")
        ax.annotate(f"crossover {xc:.1f} hpf", xy=(xc, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", fontsize=9, color=INK_2, va="top")

    if anchors:
        for a in anchors:
            ax.plot([a], [-0.04], marker="^", color=INK_2, ms=6, clip_on=False)
        ax.text(0.0, -0.16, "▲ calibration anchors · unmarked timepoints are held out",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("developmental time (hpf)")
    ax.set_ylabel("fraction of oriented power")
    ax.set_title(title or "Nematic → quartic dominance", fontweight="bold", loc="left")
    ax.legend(loc="center right", fontsize=9)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def crossover_time(d, col="nematic_frac", time="hpf", level=0.5):
    """Linearly interpolated time at which nematic drops through `level`.

    This is the headline scalar for A1 and the one number the model must predict.
    Returns NaN if the series never crosses — say so rather than reporting the
    nearest timepoint.
    """
    d = d.sort_values(time)
    t = d[time].to_numpy(float); v = d[col].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    below = v < level
    if not below.any() or below.all():
        return np.nan
    k = int(np.argmax(below))
    if k == 0:
        return np.nan
    t0, t1, v0, v1 = t[k - 1], t[k], v[k - 1], v[k]
    if v1 == v0:
        return float(t0)
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0))


def synth_fibres(shape=(200, 200), angles_deg=(0.0,), spread_deg=5.0, n=260,
                 length=90, width=3, seed=0):
    """Synthetic oriented fibre texture with KNOWN family structure.

    `angles_deg` lists the family means; fibres are assigned to families in equal
    proportion, with orientation jitter of `spread_deg` (circular sd).
    Used by dominance_controls to validate the decomposition against ground truth.
    """
    from skimage.draw import line as skline
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, float)
    fams = np.asarray(angles_deg, float)
    for i in range(n):
        mu = fams[i % len(fams)]
        th = np.radians(mu + rng.normal(0, spread_deg))
        cy = rng.uniform(0, shape[0]); cx = rng.uniform(0, shape[1])
        dy, dx = -np.sin(th) * length / 2, np.cos(th) * length / 2
        r0 = int(np.clip(cy - dy, 0, shape[0] - 1)); c0 = int(np.clip(cx - dx, 0, shape[1] - 1))
        r1 = int(np.clip(cy + dy, 0, shape[0] - 1)); c1 = int(np.clip(cx + dx, 0, shape[1] - 1))
        rr, cc = skline(r0, c0, r1, c1)
        img[rr, cc] = 1.0
    if width > 1:
        img = (ndi.distance_transform_edt(img == 0) <= (width - 1) / 2.0).astype(float)
    return img


def dominance_controls(verbose=True, window=None, scale=1.0, band=(0.02, 0.30)):
    """Validate the mixture decomposition against cases with known ground truth.

    The property that matters: p must be invariant to ANGULAR SPREAD. A single
    family reads p~1 whether it is sharp or broad; two orthogonal families read
    p~0 at any spread. The old m2/(m2+m4) ratio failed exactly here.
    """
    cases = [
        ("one sharp family (0 deg)",        (0.0,),          4.0,  1.0),
        ("one family, 20 deg spread",       (0.0,),         20.0,  1.0),
        ("one family, 35 deg spread",       (0.0,),         35.0,  1.0),
        ("one sharp family, off-axis 37deg",(37.0,),         4.0,  1.0),
        ("two families 0/90",               (0.0, 90.0),     5.0,  0.0),
        ("two families 60/150",             (60.0, 150.0),   5.0,  0.0),
        ("two families 0/90, 20deg spread", (0.0, 90.0),    20.0,  0.0),
    ]
    ok = []
    for ci, (name, fams, spread, want) in enumerate(cases):
        # deterministic seed — hash() of a str is randomised per process, which made
        # this suite flake between runs
        img = synth_fibres((220, 220), fams, spread, seed=1000 + ci)
        r = window_dominance(img, scale, band=band, n_surrogates=12)
        got = r["nematic_frac"]
        good = np.isfinite(got) and abs(got - want) <= 0.20
        ok.append(good)
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name:36s} p={got:.3f} "
                  f"(want ~{want:.1f})  m2={r['m2']:.3f} m4={r['m4']:.3f}")

    # isotropic must be gated out, not reported as some fraction
    rng = np.random.default_rng(0)
    iso = rng.random((220, 220))
    r = window_dominance(iso, scale, band=band, n_surrogates=12)
    good = r["gated"]
    ok.append(good)
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] {'structureless is gated out':36s} "
              f"gated={r['gated']} m4={r['m4']:.4f}")

    # the square-window artefact: a single family must NOT read as quartic
    img = synth_fibres((220, 220), (0.0,), 5.0, seed=7)
    r_radial = window_dominance(img, scale, band=band, n_surrogates=12)
    good = r_radial["nematic_frac"] > 0.8
    ok.append(good)
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] {'single family not read as quartic':36s} "
              f"p={r_radial['nematic_frac']:.3f}")

    # the old ratio, for contrast — shown to fail on the same inputs
    if verbose:
        print("\n  for contrast, the OLD m2/(m2+m4) ratio on the same windows:")
        for ci, (name, fams, spread, want) in enumerate(cases[:4]):
            img = synth_fibres((220, 220), fams, spread, seed=1000 + ci)
            rr = window_dominance(img, scale, band=band, n_surrogates=8)
            old = rr["m2"] / (rr["m2"] + rr["m4"]) if (rr["m2"] + rr["m4"]) > 0 else np.nan
            print(f"      {name:36s} old={old:.3f}  vs  mixture={rr['nematic_frac']:.3f}")

    n_pass = sum(ok)
    print(f"\n  {n_pass}/{len(ok)} dominance controls passed")
    return n_pass == len(ok)


if __name__ == "__main__":
    self_test()
