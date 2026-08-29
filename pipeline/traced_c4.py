"""C2 / C4 on the HAND-TRACED skeletons, and on the density-matched reference fields.

order() (the +-45 deg band statistic) reads low for BOTH a quartic field and an
isotropic one, so it cannot separate them. C4 can: isotropic 0, one family 1,
two at 90 deg 1 -- with C2 splitting those last two. Branch angles are already
available on traced skeletons via measure.branch_angles, so C4 is computable on
the traced set and not only on the model.

Angles are weighted by branch length in pixels: a 60 px bundle should not count
the same as a 9 px stub.
"""
import sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import measure as M
import order_params as OP
import reference_fields as RF
from datapaths import ROOT as DATA_ROOT

_N8 = np.ones((3, 3), int)


def branch_angles_weighted(mask, min_len_px=M.MIN_SEG_PX):
    """As measure.branch_angles, but also returns each branch's pixel count."""
    sk = skeletonize(np.asarray(mask, bool))
    deg = ndi.convolve(sk.astype(int), _N8, mode="constant") - 1
    lab, n = ndi.label(sk & (deg < 3), structure=_N8)
    ang, wt = [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        if sub.sum() < min_len_px:
            continue
        yy, xx = np.nonzero(sub)
        y, x = yy - yy.mean(), xx - xx.mean()
        cov = np.array([[float(x @ x), float(x @ y)], [float(x @ y), float(y @ y)]])
        w_, v = np.linalg.eigh(cov)
        vx, vy = v[:, int(np.argmax(w_))]
        ang.append(np.arctan2(vy, vx) % np.pi)
        wt.append(float(sub.sum()))
    return np.asarray(ang), np.asarray(wt)


def summarise(mask):
    a, w = branch_angles_weighted(mask)
    if len(a) < 4:
        return None
    r = OP.analyse(a, w)
    return r


print("TRACED SET -- C2/C4 on branch angles, length-weighted, per heart then pooled")
print(f"{'stage':>7s} {'hearts':>7s} {'C2':>8s} {'C4':>8s} {'C4 sig':>7s} {'interfam':>9s}")
for hpf in (32, 36, 40, 44, 48, 52):
    fs = sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif"))
    if not fs:
        continue
    c2, c4, sig, ifa = [], [], [], []
    for f in fs:
        m, _ = M.load_trace(f)
        r = summarise(m)
        if r is None:
            continue
        c2.append(r["C2"]); c4.append(r["C4"])
        sig.append(bool(r["C4_significant"])); ifa.append(r["interfamily_deg"])
    print(f"{hpf:>5d}hpf {len(c2):>7d} {np.mean(c2):>8.3f} {np.mean(c4):>8.3f}"
          f" {sum(sig)}/{len(sig):<5d} {np.nanmean(ifa):>9.1f}")

print()
print("REFERENCE FIELDS -- same statistic, density-matched")
print(f"{'field':<34s} {'C2':>8s} {'C4':>8s} {'C4 sig':>7s} {'interfam':>9s}")
d32, d52 = 0.047, 0.078
p1 = RF.match_pitch([0.0], d32)
p2 = RF.match_pitch([0.0, 90.0], d52)
p3 = RF.match_pitch([0.0, 60.0], d52)
n_iso = RF.match_isotropic(d52)
cases = [("single family", lambda s: RF.families(s, [0.0], p1)),
         ("two families 90 deg", lambda s: RF.families(s, [0.0, 90.0], p2)),
         ("two families 60 deg", lambda s: RF.families(s, [0.0, 60.0], p3)),
         ("isotropic", lambda s: RF.isotropic(s, n_iso))]
for nm, fn in cases:
    rs = [summarise(M.normalise(fn(s), M.SCALE_UM)) for s in range(5)]
    rs = [r for r in rs if r]
    print(f"{nm:<34s} {np.mean([r['C2'] for r in rs]):>8.3f}"
          f" {np.mean([r['C4'] for r in rs]):>8.3f}"
          f" {sum(bool(r['C4_significant']) for r in rs)}/{len(rs):<5d}"
          f" {np.nanmean([r['interfamily_deg'] for r in rs]):>9.1f}")
