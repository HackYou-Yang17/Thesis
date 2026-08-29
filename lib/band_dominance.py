"""Bias-free dominance: a hard ±22.5° band around each window's OWN dominant direction.

Replaces the von Mises mixture decomposition. What each change buys:

  * NO 90° ASSUMPTION. The mixture model solved for one family plus a second exactly
    90° away, so a real 40° crossing was scored 0.91 nematic. Here anything more than
    22.5° off the dominant direction is quartic, whatever angle it actually sits at.
    22.5° is not a tuned choice -- it is where cos(4θ) changes sign, i.e. the exact
    boundary between 2-fold and 4-fold in the harmonic the old measure used.
  * NO ISOTROPIC PEDESTAL. The surrogate profile (power shuffled within radial
    annuli, so the radial spectrum and lattice geometry survive and only orientation
    is destroyed) is subtracted per bin, so a structureless window has no oriented
    power to allocate and gates out instead of reading 0.69 nematic.
  * FRAME-INDEPENDENT. The band follows the window's own peak, so a cell running
    diagonally scores the same as one running horizontally.

Reading the scale: 1.0 = all oriented power in one family. 0.5 = two families of
equal strength at ANY separation beyond 22.5°. Below 0.5 = the off-axis power
outweighs the dominant family. Isotropic reference before floor subtraction is 0.25
(45° of 180°), which is why the subtraction matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import carma_stats as cs
from carma_stats import _bin_angles, _fft_geometry, _windowed_power, _sampling

HALF_BAND = 22.5


def oriented_profile(win, scale, band=(0.05, 0.8), nbins=180, n_surrogates=16,
                     seed=0):
    """Angular profile with the per-bin isotropic pedestal removed.

    Returns (angles, oriented_power, snr) where snr is total oriented power over
    total surrogate power -- the diagnostic that replaces the m4 gate.
    """
    win = np.asarray(win, float)
    P = _windowed_power(win)
    R, TH, sel, (dfy, dfx) = _fft_geometry(win.shape, scale, band)
    if sel.sum() < 32:
        raise ValueError(f"only {int(sel.sum())} Fourier samples in band {band}")

    idx, cnt, angles = _bin_angles(TH[sel], nbins)
    psel = P[sel]
    tot = np.bincount(idx, weights=psel, minlength=nbins)
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    rng = np.random.default_rng(seed)
    dr = float(min(dfy, dfx))
    rbin = np.round(R[sel] / dr).astype(int)
    groups = [np.flatnonzero(rbin == rb) for rb in np.unique(rbin)]
    groups = [g for g in groups if g.size > 1]
    acc = np.zeros(nbins)
    for _ in range(int(n_surrogates)):
        sh = psel.copy()
        for g in groups:
            sh[g] = sh[rng.permutation(g)]
        t = np.bincount(idx, weights=sh, minlength=nbins)
        acc += np.where(cnt > 0, t / np.maximum(cnt, 1), 0.0)
    base = acc / max(1, int(n_surrogates))

    ori = np.clip(np.nan_to_num(prof) - base, 0.0, None)
    snr = float(ori.sum() / base.sum()) if base.sum() > 0 else np.nan
    return angles, ori, snr


def _band_share(angles, prof, half_band, smooth_deg, nbins):
    """Share of profile power within +-half_band of the profile's OWN peak."""
    k = max(1, int(round(smooth_deg / (180.0 / nbins))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.convolve(np.r_[prof[-k:], prof, prof[:k]], ker, mode="same")[k:k + nbins]
    phi0 = angles[int(np.argmax(sm))]
    d = np.abs(angles - phi0) % 180.0
    d = np.minimum(d, 180.0 - d)
    tot = float(prof.sum())
    return (float(prof[d <= half_band].sum()) / tot if tot > 0 else np.nan), float(phi0)


def band_dominance(win, scale, band=(0.05, 0.8), nbins=180, n_surrogates=16,
                   seed=0, half_band=HALF_BAND, min_snr=0.05, smooth_deg=5.0):
    """One window. 'nematic_frac' is the null-corrected +-half_band share.

    THE NULL MUST BE MATCHED, not assumed. Under isotropy a +-22.5 deg band holds
    45/180 = 25% of the power in expectation -- but the band is placed on the
    profile's own peak, so on a structureless window it lands on the largest
    fluctuation and holds far more than 25%. An earlier version subtracted a flat
    floor, clipped at zero and then took the argmax, which locked that winner's
    curse in: it read 0.77 on a purely isotropic field and only 0.66 on two
    orthogonal families -- worse discrimination than the measure it replaced.

    The fix is to run the SAME statistic on the surrogates, peak-finding included,
    so the curse cancels:

        nematic = (f_observed - f_null) / (1 - f_null)

    0 = no more concentrated than chance, 1 = all oriented power in one family.
    Two equal families land near the middle, wherever they sit relative to each
    other -- which is the whole point of dropping the 90 deg assumption.
    """
    win = np.asarray(win, float)
    P = _windowed_power(win)
    R, TH, sel, (dfy, dfx) = _fft_geometry(win.shape, scale, band)
    if sel.sum() < 32:
        raise ValueError(f"only {int(sel.sum())} Fourier samples in band {band}")
    idx, cnt, angles = _bin_angles(TH[sel], nbins)
    psel = P[sel]
    tot = np.bincount(idx, weights=psel, minlength=nbins)
    prof = np.nan_to_num(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan))

    f_obs, phi0 = _band_share(angles, prof, half_band, smooth_deg, nbins)

    rng = np.random.default_rng(seed)
    dr = float(min(dfy, dfx))
    rbin = np.round(R[sel] / dr).astype(int)
    groups = [np.flatnonzero(rbin == rb) for rb in np.unique(rbin)]
    groups = [g for g in groups if g.size > 1]
    fs, base = [], np.zeros(nbins)
    for _ in range(int(n_surrogates)):
        sh = psel.copy()
        for g in groups:
            sh[g] = sh[rng.permutation(g)]
        t = np.bincount(idx, weights=sh, minlength=nbins)
        sp = np.nan_to_num(np.where(cnt > 0, t / np.maximum(cnt, 1), 0.0))
        base += sp
        f, _ = _band_share(angles, sp, half_band, smooth_deg, nbins)
        if np.isfinite(f):
            fs.append(f)
    if not fs or not np.isfinite(f_obs):
        return {"nematic_frac": np.nan, "quartic_frac": np.nan, "dir_deg": np.nan,
                "f_obs": f_obs, "f_null": np.nan, "snr": np.nan, "gated": True}
    f_null = float(np.mean(fs))
    base /= max(1, int(n_surrogates))
    ori = np.clip(prof - base, 0.0, None)
    snr = float(ori.sum() / base.sum()) if base.sum() > 0 else np.nan

    p = (f_obs - f_null) / (1.0 - f_null) if f_null < 1 else np.nan
    gated = (not np.isfinite(snr)) or (snr < min_snr) or (not np.isfinite(p))
    p = np.nan if gated else float(np.clip(p, 0.0, 1.0))
    return {"nematic_frac": p, "quartic_frac": (1 - p) if p == p else np.nan,
            "dir_deg": phi0, "f_obs": f_obs, "f_null": f_null, "snr": snr,
            "gated": bool(gated)}


def band_map(img, window, scale, stride=None, **kw):
    """Tile an image. window=None measures the whole crop as one window."""
    img = np.asarray(img, float)
    if window is None:
        r = band_dominance(img, scale, **kw)
        r.update({"iy": 0, "ix": 0})
        return pd.DataFrame([r])
    samp = _sampling(2, scale)
    wpx = [max(8, int(round(window / s))) for s in samp]
    spx = [max(1, int(round((stride if stride is not None else window / 2.0) / s)))
           for s in samp]
    if any(w > n for w, n in zip(wpx, img.shape)):
        raise ValueError(f"window {window} = {wpx} px does not fit {img.shape}")
    rows = []
    for i, y in enumerate(range(0, img.shape[0] - wpx[0] + 1, spx[0])):
        for j, x in enumerate(range(0, img.shape[1] - wpx[1] + 1, spx[1])):
            try:
                r = band_dominance(img[y:y + wpx[0], x:x + wpx[1]], scale,
                                   seed=(i * 7919 + j), **kw)
            except ValueError:
                continue
            r.update({"iy": i, "ix": j, "y": y, "x": x})
            rows.append(r)
    return pd.DataFrame(rows)


def band_summary(dm, label=None):
    if len(dm) == 0:
        return {"label": label, "n_windows": 0, "frac_gated": np.nan,
                "nematic_frac": np.nan, "quartic_frac": np.nan,
                "dir_deg": np.nan, "snr": np.nan}
    good = dm[~dm["gated"]].dropna(subset=["nematic_frac"])
    out = {"label": label, "n_windows": int(len(dm)),
           "frac_gated": float(1 - len(good) / len(dm))}
    if len(good):
        v = good["nematic_frac"].to_numpy(float)
        dirs = good["dir_deg"].to_numpy(float)
        out.update({"nematic_frac": float(np.median(v)),
                    "nematic_frac_lo": float(np.percentile(v, 25)),
                    "nematic_frac_hi": float(np.percentile(v, 75)),
                    "quartic_frac": float(1 - np.median(v)),
                    "dir_deg": float(np.degrees(np.angle(
                        np.mean(np.exp(2j * np.radians(dirs))))) / 2 % 180),
                    "dir_dispersion": cs._circ_sd_deg(dirs),
                    "snr": float(good["snr"].median())})
    else:
        out.update({k: np.nan for k in ("nematic_frac", "nematic_frac_lo",
                                        "nematic_frac_hi", "quartic_frac",
                                        "dir_deg", "dir_dispersion", "snr")})
    return out
