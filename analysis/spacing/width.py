"""
width.py -- how thick is a fibre bundle, in um, measured where the tracer put the line.

WHY THIS EXISTS
---------------
The nearest-neighbour measure records the distance from one traced CENTRELINE to the next.
That is a PITCH (centre-to-centre), not a GAP (edge-to-edge). The two differ by exactly one
fibre bundle width:

        pitch = gap + width

So the 3.2 um figure needs no correction wherever the model parameter is itself a pitch, and
needs width subtracted wherever the parameter is a free gap. Which one applies is a property
of the parameter, not of the measurement.

HOW THE WIDTH IS MEASURED
-------------------------
Not by thresholding a mask -- a mask width is a function of the threshold, and the threshold
is the analyst's. Instead, at each traced centreline pixel:

  1. take the RAW greyscale intensity profile perpendicular to the local tangent
  2. recentre on the local intensity maximum (the pen is only approximately on the ridge)
  3. baseline = the local minimum on each side within the search half-width
  4. width = full width at half maximum above that baseline

FWHM has no free parameter beyond the search half-width, and the search half-width only
decides whether a profile is measurable, not how wide it comes out.

RESOLUTION FLOOR, stated because it bounds the answer
-----------------------------------------------------
Nothing narrower than the sampling limit can be resolved: 2 x pixel = 0.72 um at 32 hpf
(0.361 um/px) and 0.96 um at 52 hpf (0.4815 um/px). A measured width at or near that floor
is an upper bound on the true width and must be reported as such.

Run at NATIVE pixel scale. Resampling to the common 0.40 um/px grid interpolates, which
widens narrow ridges -- fine for a spacing (positions move together) but not for a width.
"""
import os
import glob
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

import spacing as sp

SEARCH_UM = 3.0        # half-width of the perpendicular profile
STEP_PX = 0.1
RECENTRE_UM = 0.6      # how far the pen may sit off the ridge crest


def raw_grey(path):
    a = tifffile.imread(path).astype(float)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def trace_mask(path):
    """Red (237,28,36) or yellow-green (181,230,29) ImageJ annotation."""
    a = tifffile.imread(path)
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    red = (r - np.maximum(g, b)) > 60
    grn = (g - b) > 60
    return red if red.sum() > grn.sum() else grn


def fwhm_widths(grey, sk, um_per_px, search_um=SEARCH_UM, step_px=STEP_PX,
                recentre_um=RECENTRE_UM):
    ang = sp.tangent_angles(sk)
    h, w = grey.shape
    n_search = search_um / um_per_px
    ts = np.arange(-n_search, n_search + 1e-9, step_px)
    n_rec = recentre_um / um_per_px
    out, fail = [], 0
    for (r0, c0) in np.argwhere(np.isfinite(ang)):
        a0 = ang[r0, c0]
        u = np.array([np.cos(a0), -np.sin(a0)])          # perpendicular, (row, col)
        rr = r0 + ts * u[0]
        cc = c0 + ts * u[1]
        if rr.min() < 0 or rr.max() > h - 1 or cc.min() < 0 or cc.max() > w - 1:
            continue
        p = ndi.map_coordinates(grey, [rr, cc], order=1, mode='nearest')
        i0 = len(ts) // 2
        # recentre on the ridge crest
        lo = max(0, int(i0 - n_rec / step_px)); hi = min(len(p), int(i0 + n_rec / step_px) + 1)
        ic = lo + int(np.argmax(p[lo:hi]))
        peak = p[ic]
        left, right = p[:ic + 1], p[ic:]
        if len(left) < 3 or len(right) < 3:
            fail += 1
            continue
        base = max(left.min(), right.min())              # the shallower shoulder
        if peak - base < 1e-9:
            fail += 1
            continue
        half = base + 0.5 * (peak - base)
        kl = np.where(left <= half)[0]
        kr = np.where(right <= half)[0]
        if not len(kl) or not len(kr):
            fail += 1
            continue
        # linear interpolation of the half-max crossing
        il = kl[-1]
        if il + 1 <= ic:
            f = (half - p[il]) / (p[il + 1] - p[il]) if p[il + 1] != p[il] else 0.0
            xl = il + f
        else:
            xl = float(il)
        ir = ic + kr[0]
        if ir - 1 >= ic and p[ir - 1] != p[ir]:
            f = (p[ir - 1] - half) / (p[ir - 1] - p[ir])
            xr = (ir - 1) + f
        else:
            xr = float(ir)
        out.append((xr - xl) * step_px * um_per_px)
    return np.array(out), fail


def edt_widths(grey, um_per_px, pct=70.0):
    """Threshold-based cross-check: 2*EDT - 1 on the skeleton of a tubeness mask."""
    from skimage.filters import sato
    lo, hi = 0.4 / um_per_px, 0.9 / um_per_px
    t = sato(grey, sigmas=np.linspace(lo, hi, 3), black_ridges=False)
    m = t > np.percentile(t, pct)
    m = ndi.binary_closing(m, np.ones((3, 3)))
    sk = skeletonize(m)
    d = ndi.distance_transform_edt(m)
    return (2.0 * d[sk] - 1.0) * um_per_px


if __name__ == '__main__':
    rows = []
    for hpf in (32, 52):
        for tf in sorted(glob.glob('/mnt/user-data/uploads/analysis/%dhpf/*- Copy (2).tif' % hpf)):
            raw = tf.replace(' - Copy (2)', '')
            name = os.path.basename(raw)[:-4]
            um = sp.read_scale(tf)
            grey = raw_grey(raw)
            tm = trace_mask(tf)
            if grey.shape != tm.shape:
                print('SHAPE MISMATCH', name, grey.shape, tm.shape)
                continue
            sk = skeletonize(tm)
            wF, fail = fwhm_widths(grey, sk, um)
            wE = edt_widths(grey, um)
            rows.append(dict(hpf=hpf, image=name, um_per_px=um,
                             n=len(wF), fail=fail,
                             fwhm_mean=wF.mean(), fwhm_med=np.median(wF),
                             fwhm_p25=np.percentile(wF, 25), fwhm_p75=np.percentile(wF, 75),
                             edt_med=np.median(wE), nyquist=2 * um))
            np.save('width_%s.npy' % name, wF)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv('fibre_width.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print()
    for hpf in (32, 52):
        s = df[df.hpf == hpf]
        print('%d hpf: FWHM per-heart medians %s -> mean %.2f um  (Nyquist floor %.2f um)'
              % (hpf, np.round(s.fwhm_med.values, 2), s.fwhm_med.mean(), s.nyquist.iloc[0]))
