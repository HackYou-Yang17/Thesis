"""
width_levels.py -- the same perpendicular profiles, read at several fractions of the peak.

"About 4 pixels" and "FWHM" are not the same statement. A ridge whose FULL WIDTH AT HALF
MAXIMUM is 2 px typically spans ~4 px before it disappears into background, because the
profile has skirts. Both numbers are true; only one of them is the bundle.

This reports the width at 75%, 50% (FWHM), 25% and 10% of the peak-above-baseline, in um and
in native pixels, so the choice of definition is visible rather than assumed.
"""
import glob
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import pandas as pd

import spacing as sp
import width as W
from datapaths import ROOT as DATA_ROOT

LEVELS = [0.75, 0.50, 0.25, 0.10]


def widths_at_levels(grey, sk, um_per_px, search_um=W.SEARCH_UM, step_px=W.STEP_PX,
                     recentre_um=W.RECENTRE_UM):
    ang = sp.tangent_angles(sk)
    h, w = grey.shape
    n_search = search_um / um_per_px
    ts = np.arange(-n_search, n_search + 1e-9, step_px)
    n_rec = recentre_um / um_per_px
    res = {L: [] for L in LEVELS}
    for (r0, c0) in np.argwhere(np.isfinite(ang)):
        a0 = ang[r0, c0]
        u = np.array([np.cos(a0), -np.sin(a0)])
        rr = r0 + ts * u[0]
        cc = c0 + ts * u[1]
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
        for L in LEVELS:
            lev = base + L * (peak - base)
            kl = np.where(left <= lev)[0]
            kr = np.where(right <= lev)[0]
            if not len(kl) or not len(kr):
                continue
            il = kl[-1]
            xl = il + ((lev - p[il]) / (p[il + 1] - p[il]) if il + 1 <= ic and p[il + 1] != p[il] else 0.0)
            ir = ic + kr[0]
            xr = (ir - 1) + ((p[ir - 1] - lev) / (p[ir - 1] - p[ir])) if ir - 1 >= ic and p[ir - 1] != p[ir] else float(ir)
            res[L].append((xr - xl) * step_px * um_per_px)
    return {L: np.array(v) for L, v in res.items()}


if __name__ == '__main__':
    rows = []
    for hpf in (32, 52):
        for tf in sorted(glob.glob(DATA_ROOT + '/%dhpf/*- Copy (2).tif' % hpf)):
            raw = tf.replace(' - Copy (2)', '')
            name = os.path.basename(raw)[:-4]
            um = sp.read_scale(tf)
            grey = W.raw_grey(raw)
            sk = skeletonize(W.trace_mask(tf))
            r = widths_at_levels(grey, sk, um)
            d = dict(hpf=hpf, image=name, um_per_px=um)
            for L in LEVELS:
                d['w%d_um' % int(L * 100)] = float(np.median(r[L]))
                d['w%d_px' % int(L * 100)] = float(np.median(r[L])) / um
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv('fibre_width_levels.csv', index=False)
    pd.set_option('display.width', 220)
    print(df.to_string(index=False, float_format=lambda v: '%.2f' % v))
    print('\nPER TIMEPOINT (mean of per-heart medians)')
    print('%5s %10s %10s %10s %10s' % ('hpf', '75%', 'FWHM', '25%', '10%'))
    for hpf in (32, 52):
        s = df[df.hpf == hpf]
        print('%5d %10s %10s %10s %10s' % (hpf,
              '%.2f um' % s.w75_um.mean(), '%.2f um' % s.w50_um.mean(),
              '%.2f um' % s.w25_um.mean(), '%.2f um' % s.w10_um.mean()))
        print('%5s %10s %10s %10s %10s' % ('', '%.1f px' % s.w75_px.mean(),
              '%.1f px' % s.w50_px.mean(), '%.1f px' % s.w25_px.mean(),
              '%.1f px' % s.w10_px.mean()))
