"""
simple_untraced.py -- the plainest version. Untraced images, perpendicular nearest
neighbour, nothing else.

Four steps:
  1. read the image, resample to a common 0.40 um/px grid, centre-crop to 61.4 um
  2. detect fibres: Sato tubeness at the fibre scale, keep the top 30% of the response,
     skeletonise
  3. for every skeleton pixel, step perpendicular to its own local tangent, both ways, and
     record the distance to the first other fibre running within 20 deg of the same direction
  4. report the median of those distances, one number per heart

No density matching, no outlier filter, no regional classification, no boundary masking.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import spacing as sp
import raw_detect as RD
import nn
import run_nn as R

PERCENTILE = 70.0     # set by agreement with the hand traces, not by the spacing answer
CAP_UM = 25.0         # search ceiling, so the march terminates
OFFSET_UM = 0.20      # half-pixel discretisation, measured on synthetics of known pitch


def spacing_of(path):
    img, _ = RD.load_raw(path)
    sk, _ = RD.detect(img, percentile=PERCENTILE)
    d, _nearest, censored = nn.nn_spacings(sk, max_um=CAP_UM)
    d = d + OFFSET_UM
    return np.median(d), d.mean(), censored, sp.line_density(sk)


if __name__ == '__main__':
    rows = []
    for hpf, traced in R.files():
        raw = traced.replace(' - Copy', '')
        med, mean, cens, dens = spacing_of(raw)
        rows.append(dict(hpf=hpf, image=os.path.basename(raw)[:-4],
                         median_um=med, mean_um=mean,
                         censored=cens, line_density=dens))
    df = pd.DataFrame(rows)
    df.to_csv('simple_untraced.csv', index=False)

    print('PER HEART')
    print(df.to_string(index=False, float_format=lambda v: '%.2f' % v))

    print('\nPER TIMEPOINT (mean of the three hearts, 95%% t-interval)')
    print(' hpf   median spacing      mean spacing')
    for h in R.HPF:
        a = df[df.hpf == h].median_um.values
        b = df[df.hpf == h].mean_um.values
        ca = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        cb = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
        print('  %d   %.2f +- %.2f um     %.2f +- %.2f um' % (h, a.mean(), ca, b.mean(), cb))

    for col in ['median_um', 'mean_um']:
        r, p = stats.pearsonr(df.hpf, df[col])
        e = df[df.hpf.isin([32, 36])][col]
        l = df[df.hpf.isin([44, 48, 52])][col]
        t = stats.ttest_ind(e, l, equal_var=False)
        print('\n%s: pooled %.2f um   trend r = %+.3f, p = %.4f' % (col, df[col].mean(), r, p))
        print('   early (32-36) %.2f   late (44-52) %.2f   difference %+.2f um, Welch p = %.4f'
              % (e.mean(), l.mean(), e.mean() - l.mean(), t.pvalue))
