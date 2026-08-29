"""
raw_width.py -- fibre bundle width from the UNTRACED images alone. No trace anywhere.

DIFFERENCE FROM width.py
------------------------
width.py measured the raw greyscale but sampled it at positions the TRACER chose. That
still carries the tracer's decision about which structures are fibres. Here the centrelines
are found by a detector, so nothing human enters the measurement.

THE THRESHOLD DOES NOT SET THE WIDTH
------------------------------------
A percentile threshold on the tubeness response decides WHICH ridges get sampled and WHERE
their centres are. It does not decide how wide they come out, because the width is read from
the raw intensity profile by FWHM, not from the thresholded mask. That is the whole reason
for measuring this way, and it is checked rather than asserted: the percentile is swept
50-90 and the width reported at each.

For contrast, two mask-based widths are computed alongside, both of which ARE functions of
the threshold:
    2*EDT - 1   on the mask, at the skeleton
    area / length   total mask area divided by total skeleton length
They are reported so the threshold dependence is visible, not because either is preferred.

NATIVE SCALE
------------
No resampling. Interpolation onto the common 0.40 um/px grid widens a narrow ridge, which
is harmless for a spacing (both edges move together) and not harmless for a width.

RESOLUTION
----------
A measured FWHM is the true width convolved with the imaging and sampling response. Both are
reported: w_meas, and w_deconv = sqrt(w_meas^2 - psf^2 - pixel^2), with psf = 0.23 um
(63x / NA 1.4, the project's stated figure) and pixel = the image's own pixel size. The
deconvolved value is a lower bound and the measured value an upper bound.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects

import spacing as sp
import width as W
from datapaths import ROOT as DATA_ROOT

ROOT = DATA_ROOT
HPF = [32, 36, 40, 44, 48, 52]
FIBRE_SCALES_UM = (0.4, 0.9)      # the project's established fibre scale band
PSF_UM = 0.23                     # diffraction, 63x / NA 1.4
MAX_SAMPLES = 4000                # cap on profiles per image, for runtime only
LEVELS = [0.50, 0.25]


def raw_files():
    out = []
    for h in HPF:
        for f in sorted(glob.glob(os.path.join(ROOT, '%dhpf' % h, '*.tif'))):
            b = os.path.basename(f)
            if 'Copy' in b:
                continue
            out.append((h, f))
    return out


def detect_native(img, um_per_px, percentile, min_obj_um=2.0):
    """Sato tubeness at the fibre scale, percentile threshold, skeleton. Native scale."""
    lo, hi = FIBRE_SCALES_UM[0] / um_per_px, FIBRE_SCALES_UM[1] / um_per_px
    t = sato(img, sigmas=np.linspace(lo, hi, 3), black_ridges=False)
    m = t > np.percentile(t, percentile)
    m = ndi.binary_closing(m, np.ones((3, 3)))
    m = remove_small_objects(m, int(round(min_obj_um / um_per_px)) ** 2)
    return m, skeletonize(m)


def profile_widths(grey, sk, um_per_px, levels=LEVELS, search_um=3.0, step_px=0.1,
                   recentre_um=0.6, max_samples=MAX_SAMPLES, seed=0):
    """FWHM (and other levels) of the raw profile perpendicular to each detected centreline."""
    ang = sp.tangent_angles(sk)
    pts = np.argwhere(np.isfinite(ang))
    if len(pts) > max_samples:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_samples, replace=False)]
    h, w = grey.shape
    n_search = search_um / um_per_px
    ts = np.arange(-n_search, n_search + 1e-9, step_px)
    n_rec = recentre_um / um_per_px
    res = {L: [] for L in levels}
    for (r0, c0) in pts:
        a0 = ang[r0, c0]
        u = np.array([np.cos(a0), -np.sin(a0)])
        rr, cc = r0 + ts * u[0], c0 + ts * u[1]
        if rr.min() < 0 or rr.max() > h - 1 or cc.min() < 0 or cc.max() > w - 1:
            continue
        p = ndi.map_coordinates(grey, [rr, cc], order=1, mode='nearest')
        i0 = len(ts) // 2
        lo = max(0, int(i0 - n_rec / step_px)); hi = min(len(p), int(i0 + n_rec / step_px) + 1)
        ic = lo + int(np.argmax(p[lo:hi]))
        peak = p[ic]
        left, right = p[:ic + 1], p[ic:]
        if len(left) < 3 or len(right) < 3:
            continue
        base = max(left.min(), right.min())
        if peak - base < 1e-9:
            continue
        for L in levels:
            lev = base + L * (peak - base)
            kl = np.where(left <= lev)[0]
            kr = np.where(right <= lev)[0]
            if not len(kl) or not len(kr):
                continue
            il = kl[-1]
            xl = il + ((lev - p[il]) / (p[il + 1] - p[il])
                       if il + 1 <= ic and p[il + 1] != p[il] else 0.0)
            ir = ic + kr[0]
            xr = ((ir - 1) + (p[ir - 1] - lev) / (p[ir - 1] - p[ir])
                  if ir - 1 >= ic and p[ir - 1] != p[ir] else float(ir))
            res[L].append((xr - xl) * step_px * um_per_px)
    return {L: np.array(v) for L, v in res.items()}


def mask_widths(m, sk, um_per_px):
    d = ndi.distance_transform_edt(m)
    edt = (2.0 * d[sk] - 1.0) * um_per_px
    area = m.sum() * um_per_px ** 2
    length = sk.sum() * um_per_px
    return float(np.median(edt)), (area / length if length else np.nan)


def deconv(w_meas, um_per_px, psf=PSF_UM):
    v = w_meas ** 2 - psf ** 2 - um_per_px ** 2
    return float(np.sqrt(v)) if v > 0 else np.nan


def run(percentile=70.0, files=None):
    rows = []
    for h, f in (files or raw_files()):
        um = sp.read_scale(f)
        grey = W.raw_grey(f)
        m, sk = detect_native(grey, um, percentile)
        r = profile_widths(grey, sk, um)
        edt, arealen = mask_widths(m, sk, um)
        w50 = float(np.median(r[0.50])) if len(r[0.50]) else np.nan
        w25 = float(np.median(r[0.25])) if len(r[0.25]) else np.nan
        rows.append(dict(hpf=h, image=os.path.basename(f)[:-4], um_per_px=um,
                         percentile=percentile, n=len(r[0.50]),
                         fwhm_um=w50, fwhm_px=w50 / um,
                         fwhm_deconv_um=deconv(w50, um),
                         w25_um=w25, edt_um=edt, area_over_len_um=arealen))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 240)
    fmt = lambda v: '%.2f' % v

    print('=== ALL 18 UNTRACED HEARTS, percentile 70 ===')
    df = run(70.0)
    df.to_csv('raw_width.csv', index=False)
    print(df.to_string(index=False, float_format=fmt))

    print('\nPER TIMEPOINT (mean of 3 hearts)')
    print('%5s %12s %12s %12s %12s %12s' % ('hpf', 'FWHM', 'deconv', '25% level',
                                            '2*EDT-1', 'area/length'))
    for h in HPF:
        s = df[df.hpf == h]
        print('%5d %12s %12s %12s %12s %12s' % (
            h, '%.2f um' % s.fwhm_um.mean(), '%.2f um' % s.fwhm_deconv_um.mean(),
            '%.2f um' % s.w25_um.mean(), '%.2f um' % s.edt_um.mean(),
            '%.2f um' % s.area_over_len_um.mean()))
    print('\nPOOLED over all 18: FWHM %.2f +- %.2f um (SD), deconvolved %.2f um'
          % (df.fwhm_um.mean(), df.fwhm_um.std(ddof=1), df.fwhm_deconv_um.mean()))
    from scipy import stats
    r, p = stats.pearsonr(df.hpf, df.fwhm_um)
    print('trend with stage: r = %+.3f, p = %.3f' % (r, p))

    print('\n=== THRESHOLD SWEEP (32 and 52 hpf) ===')
    sub = [(h, f) for h, f in raw_files() if h in (32, 52)]
    out = []
    for P in [50.0, 60.0, 70.0, 80.0, 90.0]:
        d = run(P, sub)
        for h in (32, 52):
            s = d[d.hpf == h]
            out.append(dict(percentile=P, hpf=h, fwhm_um=s.fwhm_um.mean(),
                            w25_um=s.w25_um.mean(), edt_um=s.edt_um.mean(),
                            area_over_len_um=s.area_over_len_um.mean(),
                            n_skel_px=s.n.mean()))
    sw = pd.DataFrame(out)
    sw.to_csv('raw_width_sweep.csv', index=False)
    print(sw.to_string(index=False, float_format=fmt))
    print('\nFWHM range over the sweep: %.2f - %.2f um  (spread %.2f)'
          % (sw.fwhm_um.min(), sw.fwhm_um.max(), sw.fwhm_um.max() - sw.fwhm_um.min()))
    print('2*EDT-1 range over the same sweep: %.2f - %.2f um  (spread %.2f)'
          % (sw.edt_um.min(), sw.edt_um.max(), sw.edt_um.max() - sw.edt_um.min()))
    print('area/length range over the same sweep: %.2f - %.2f um  (spread %.2f)'
          % (sw.area_over_len_um.min(), sw.area_over_len_um.max(),
             sw.area_over_len_um.max() - sw.area_over_len_um.min()))
