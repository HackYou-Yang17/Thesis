"""FFT dominance on the HAND-TRACED 32-52 hpf crops (18 images, 6 timepoints, 3 hearts each).

Substrate is different from the raw fluorescence runs, and that changes the
preprocessing:

  * the trace is a pure annotation layer -- ImageJ red (237,28,36), no
    anti-aliasing -- so extraction is exact, not a threshold choice;
  * the grain/blur confound that dominated the raw-image runs is GONE by
    construction. The confound it is replaced by is the tracer's eye;
  * `tissue_mask` is meaningless here (it thresholds local brightness). Not used.

WIDTH AND PIXEL-SCALE NORMALISATION IS REQUIRED, not cosmetic. Pixel size runs
0.361-0.5675 um/px across the set and is CORRELATED WITH TIMEPOINT (coarsest at
48 hpf), so a 1-px pen stroke is 0.36 um wide early and 0.57 um late. Left alone
that is a systematic change in the measured object across the very axis under
test. Every trace is therefore skeletonised, resampled to a common 0.40 um/px
grid, re-skeletonised and re-dilated to one fixed physical width.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.transform import resize

import fieldstats as cs
from datapaths import ROOT as DATA_ROOT

# ── fixed parameters, all declared up front ─────────────────────────────────
WINDOW_UM = 15.0        # set by the >=15-independent-window floor at 61.4 um
FIELD_UM = 61.4         # smallest field in the set; all crops trimmed to it
SCALE_UM = 0.40         # common resampling grid
WIDTH_PX = 3            # fixed trace width after normalisation (1.2 um)
BAND = (0.05, 0.8)      # cycles/um -- unchanged from the raw-image pipeline
MIN_OCC = 0.02          # per-window trace occupancy floor (see note below)
SUPPRESS = None         # no angular suppression at any timepoint
RED_EXCESS = 60


def load_trace(path):
    """Red annotation layer -> normalised binary line drawing at SCALE_UM."""
    a = tifffile.imread(path)[..., :3].astype(int)
    with tifffile.TiffFile(path) as tf:
        xr = tf.pages[0].tags["XResolution"].value
        yr = tf.pages[0].tags["YResolution"].value
        s_in = (yr[1] / yr[0], xr[1] / xr[0])
    raw = (a[..., 0] - np.maximum(a[..., 1], a[..., 2])) > RED_EXCESS

    skel = skeletonize(raw)
    n_out = [int(round(n * s / SCALE_UM)) for n, s in zip(skel.shape, s_in)]
    rs = resize(skel.astype(float), n_out, order=1, anti_aliasing=False) > 0.15
    rs = skeletonize(rs)                       # resampling thickens; re-thin
    out = binary_dilation(rs, disk(WIDTH_PX // 2))

    n = int(round(FIELD_UM / SCALE_UM))        # centre-crop, never rescale
    r0 = max(0, (out.shape[0] - n) // 2)
    c0 = max(0, (out.shape[1] - n) // 2)
    out = out[r0:r0 + n, c0:c0 + n]
    if out.shape != (n, n):                    # pad the 1-2 px rounding shortfall
        pad = np.zeros((n, n), bool)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out, raw.mean(), s_in[0]


def occupancy_map(mask, wpx, spx):
    """Trace occupancy per window, on the same grid dominance_map walks."""
    occ = {}
    for i, y in enumerate(range(0, mask.shape[0] - wpx + 1, spx)):
        for j, x in enumerate(range(0, mask.shape[1] - wpx + 1, spx)):
            occ[(i, j)] = float(mask[y:y + wpx, x:x + wpx].mean())
    return occ


# ── independent check: angles read straight off the skeleton ────────────────
def skeleton_dominance(mask, wpx, spx, occ_floor=MIN_OCC):
    """Same mixture decomposition, but the angular profile comes from the
    LOCAL TANGENT of the traced lines rather than from a Fourier transform.

    No windowing, no band, no noise floor -- so if this agrees with the FFT the
    agreement is not an artefact shared by the two routes.
    """
    skel = skeletonize(mask)
    sm = ndi.gaussian_filter(skel.astype(float), 1.5)
    gy, gx = np.gradient(sm)
    rows = []
    for y in range(0, mask.shape[0] - wpx + 1, spx):
        for x in range(0, mask.shape[1] - wpx + 1, spx):
            sl = (slice(y, y + wpx), slice(x, x + wpx))
            if mask[sl].mean() < occ_floor:
                continue
            sel = skel[sl]
            if sel.sum() < 20:
                continue
            # gradient is normal to the line; +90 deg gives the tangent
            th = (np.degrees(np.arctan2(gy[sl][sel], gx[sl][sel])) + 90.0) % 180.0
            w = np.hypot(gx[sl][sel], gy[sl][sel])
            if w.sum() <= 0:
                continue
            m2 = abs(np.sum(w * np.exp(2j * np.radians(th))) / w.sum())
            m4 = abs(np.sum(w * np.exp(4j * np.radians(th))) / w.sum())
            p, _, _ = cs.nematic_fraction(m2, m4)
            if np.isfinite(p):
                rows.append(p)
    return float(np.median(rows)) if rows else np.nan


def main():
    files = sorted(glob.glob(DATA_ROOT + "/*/*Copy.tif"),
                   key=lambda f: (int(re.search(r"(\d+)hpf", f).group(1)), f))
    wpx = int(round(WINDOW_UM / SCALE_UM))
    spx = max(1, wpx // 2)
    rows = []
    for f in files:
        hpf = float(re.search(r"(\d+)hpf", f).group(1))
        name = os.path.basename(f).replace(" - Copy.tif", "")
        mask, cov_raw, s_in = load_trace(f)
        img = mask.astype(float)

        dm = cs.dominance_map(img, window=WINDOW_UM, scale=SCALE_UM,
                              band=BAND, n_surrogates=16, suppress_deg=SUPPRESS)
        occ = occupancy_map(mask, wpx, spx)
        dm["occ"] = [occ[(i, j)] for i, j in zip(dm.iy, dm.ix)]
        n_all = len(dm)
        dm_used = dm[dm.occ >= MIN_OCC].copy()
        r = cs.dominance_summary(dm_used, label=name)
        r.update({"hpf": hpf, "file": name,
                  "px_um_in": s_in, "trace_cov": mask.mean(),
                  "n_win_total": n_all, "n_win_sparse": n_all - len(dm_used),
                  "p_skel": skeleton_dominance(mask, wpx, spx)})
        rows.append(r)
        print(f"  {name:9s} {hpf:3.0f} hpf  in {s_in:.4f} um/px  cov {mask.mean():.3f}  "
              f"windows {len(dm_used)}/{n_all}  gated {r['frac_gated']:.2f}  "
              f"p_fft {r['nematic_frac']:.3f}  p_skel {r['p_skel']:.3f}  "
              f"dir2 {r['dir2_deg']:.0f}deg", flush=True)
        dm_used.assign(file=name, hpf=hpf).to_csv(
            f"windows_{name}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv("traced_dominance_per_image.csv", index=False)

    tp = (df.groupby("hpf").agg(
        n_hearts=("file", "size"),
        p_mean=("nematic_frac", "mean"), p_sd=("nematic_frac", "std"),
        p_min=("nematic_frac", "min"), p_max=("nematic_frac", "max"),
        p_skel=("p_skel", "mean"),
        gated=("frac_gated", "mean"), dir2=("dir2_deg", "median"),
        m4=("m4", "mean"), snr=("snr_ratio", "mean"),
        band_frac=("band_frac", "mean")).reset_index())
    from scipy import stats as st
    tp["p_sem"] = tp.p_sd / np.sqrt(tp.n_hearts)
    tp["ci95"] = tp.p_sem * st.t.ppf(0.975, tp.n_hearts - 1)
    tp.to_csv("traced_dominance_per_timepoint.csv", index=False)

    print("\n=== per timepoint (mean over hearts) ===")
    print(tp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    d = df.dropna(subset=["nematic_frac"])
    r, pv = st.pearsonr(d.hpf, d.nematic_frac)
    rs, pvs = st.spearmanr(d.hpf, d.nematic_frac)
    early = d[d.hpf <= 36].nematic_frac
    late = d[d.hpf >= 44].nematic_frac
    tt = st.ttest_ind(early, late, equal_var=False)
    mw = st.mannwhitneyu(early, late)
    print(f"\ntrend across hearts (n={len(d)}): pearson r={r:.3f} p={pv:.4f} | "
          f"spearman rho={rs:.3f} p={pvs:.4f}")
    print(f"early (32-36, n={len(early)}) {early.mean():.3f} vs "
          f"late (44-52, n={len(late)}) {late.mean():.3f}: "
          f"Welch p={tt.pvalue:.4f}, Mann-Whitney p={mw.pvalue:.4f}")
    print(f"crossover: {cs.crossover_time(tp.rename(columns={'p_mean':'nematic_frac'}))}")
    ra, pa = st.pearsonr(d.nematic_frac, d.snr_ratio)
    print(f"confound sign r(p, snr_ratio) = {ra:+.3f} (p={pa:.3f})  "
          f"[negative = genuine second family, positive = danger sign]")
    rc, pc = st.pearsonr(d.nematic_frac, d.trace_cov)
    print(f"r(p, trace coverage) = {rc:+.3f} (p={pc:.3f})")
    ok = d.dropna(subset=["p_skel"])
    rk, pk = st.pearsonr(ok.nematic_frac, ok.p_skel)
    print(f"FFT vs skeleton-angle route: r = {rk:+.3f} (p={pk:.4f}), "
          f"mean |dp| = {np.abs(ok.nematic_frac - ok.p_skel).mean():.3f}")
    return df, tp


if __name__ == "__main__":
    main()
