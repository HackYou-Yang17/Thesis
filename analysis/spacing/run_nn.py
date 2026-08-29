"""
run_nn.py -- window-free perpendicular nearest-neighbour spacing on all 18 traced hearts.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
import spacing as sp
import nn

ROOT = '/mnt/user-data/uploads/analysis'
HPF = [32, 36, 40, 44, 48, 52]

# Half-pixel discretisation offset. The ray is sampled at rounded pixel centres, so every
# hit is found half a sample step early. Measured on synthetic families of KNOWN pitch
# (nn calibration, 1.5-10 um): a constant -0.20 um, independent of the pitch. Applied as a
# fixed additive correction, not fitted to the real data.
OFFSET_UM = 0.20

# Pre-declared regime split, taken from the dominance result (crossover 40.1 hpf
# [37.0, 42.9]) BEFORE any spacing number was seen. 40 hpf is transitional and is in
# neither group.
EARLY, LATE = [32, 36], [44, 48, 52]


def files():
    return [(h, f) for h in HPF
            for f in sorted(glob.glob(os.path.join(ROOT, '%dhpf' % h, '*- Copy.tif')))]


def per_image(angle_tol_deg=nn.ANGLE_TOL_DEG, max_um=nn.MAX_UM, target=0.40):
    rows, dists = [], {}
    for h, f in files():
        name = os.path.basename(f).replace(' - Copy.tif', '')
        sk = sp.normalise(sp.extract_trace(f), sp.read_scale(f), target=target)
        b, near, cens = nn.nn_spacings(sk, um_per_px=target,
                                       angle_tol_deg=angle_tol_deg, max_um=max_um)
        b = b + OFFSET_UM
        lam = sp.line_density(sk, target)
        rows.append(dict(hpf=h, image=name, input_um_per_px=sp.read_scale(f),
                         n_pairs=len(b), censored_frac=cens,
                         spacing_um=float(np.median(b)),
                         spacing_mean_um=float(np.mean(b)),
                         p25=float(np.percentile(b, 25)), p75=float(np.percentile(b, 75)),
                         nn_um=float(np.median(near)) + OFFSET_UM,
                         line_density=lam, iso_null_um=sp.isotropic_spacing(lam)))
        dists[name] = b
    return pd.DataFrame(rows), dists


def per_timepoint(df):
    out = []
    for h in HPF:
        s = df[df.hpf == h].spacing_um.values
        n = len(s); m = s.mean(); sd = s.std(ddof=1); se = sd / np.sqrt(n)
        ci = stats.t.ppf(0.975, n - 1) * se
        out.append(dict(hpf=h, n_hearts=n, mean_um=m, sd_um=sd, se_um=se, ci95_um=ci,
                        lo=m - ci, hi=m + ci,
                        censored=df[df.hpf == h].censored_frac.mean(),
                        iso_null_um=df[df.hpf == h].iso_null_um.mean()))
    return pd.DataFrame(out)


def two_values(df, early=EARLY, late=LATE):
    e = df[df.hpf.isin(early)].spacing_um.values
    l = df[df.hpf.isin(late)].spacing_um.values
    t = stats.ttest_ind(e, l, equal_var=False)
    u = stats.mannwhitneyu(e, l)
    pooled = np.sqrt(((len(e) - 1) * e.var(ddof=1) + (len(l) - 1) * l.var(ddof=1))
                     / (len(e) + len(l) - 2))
    return dict(nematic_um=e.mean(), nematic_sd=e.std(ddof=1), nematic_n=len(e),
                nematic_ci=stats.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / np.sqrt(len(e)),
                mesh_um=l.mean(), mesh_sd=l.std(ddof=1), mesh_n=len(l),
                mesh_ci=stats.t.ppf(0.975, len(l) - 1) * l.std(ddof=1) / np.sqrt(len(l)),
                difference_um=e.mean() - l.mean(), ratio=e.mean() / l.mean(),
                welch_t=t.statistic, welch_p=t.pvalue,
                mw_U=u.statistic, mw_p=u.pvalue, cohens_d=(e.mean() - l.mean()) / pooled)


def trend(df):
    r, p = stats.pearsonr(df.hpf, df.spacing_um)
    rs, ps = stats.spearmanr(df.hpf, df.spacing_um)
    return dict(pearson_r=r, pearson_p=p, spearman_r=rs, spearman_p=ps)


if __name__ == '__main__':
    pd.set_option('display.width', 170)
    fmt = lambda v: '%.3f' % v
    df, dists = per_image()
    df.to_csv('nn_per_image.csv', index=False)
    print('=== PER IMAGE ===')
    print(df[['hpf', 'image', 'n_pairs', 'censored_frac', 'spacing_um', 'p25', 'p75',
              'nn_um', 'line_density', 'iso_null_um']].to_string(index=False, float_format=fmt))
    tp = per_timepoint(df); tp.to_csv('nn_per_timepoint.csv', index=False)
    print('\n=== PER TIMEPOINT ===')
    print(tp.to_string(index=False, float_format=fmt))
    print('\n=== TREND ===', trend(df))
    print('\n=== TWO VALUES (32-36 vs 44-52) ===')
    for k, v in two_values(df).items():
        print('  %-16s %.4f' % (k, v))
    print('\n=== DENSITY ===')
    print('  r(density,hpf) = %.3f p=%.4f' % stats.pearsonr(df.hpf, df.line_density))
    print('  r(density,spacing) = %.3f p=%.4f' % stats.pearsonr(df.line_density, df.spacing_um))
    print('  measured / isotropic null = %.3f' % (df.spacing_um / df.iso_null_um).mean())
    np.save('nn_dists.npy', dists, allow_pickle=True)
