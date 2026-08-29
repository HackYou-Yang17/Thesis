"""C2 / C4 from the WHOLE-FIELD angular power spectrum, not from skeleton branches.

Branch-angle C4 fails: skeletonisation cuts every fibre at each junction, so a
branch is short and nearly straight whatever the fibre does, and C4 ends up
measuring branch straightness rather than tetratic order (see curl_match.py --
the single-family and two-family references fall on the same C4-vs-curvature
curve). The angular power spectrum has no junctions in it, so it is immune.

C_k = |sum_theta P(theta) exp(i k theta)| / sum_theta P(theta), on the
null-corrected oriented profile from band_dominance.oriented_profile.
"""
import sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
import measure as M
from band_dominance import oriented_profile
import reference_fields as RF
from curved_refs import curved_field
from datapaths import ROOT as DATA_ROOT


def spectral_C(mask, seed=0):
    ang, ori, snr = oriented_profile(np.asarray(mask, float), scale=M.SCALE_UM,
                                     band=(0.05, 0.8), seed=seed)
    th = np.deg2rad(np.asarray(ang, float))
    p = np.asarray(ori, float)
    tot = p.sum()
    if tot <= 0:
        return np.nan, np.nan, snr
    c2 = abs((p * np.exp(2j * th)).sum() / tot)
    c4 = abs((p * np.exp(4j * th)).sum() / tot)
    return float(c2), float(c4), snr


def avg(build, n=5):
    v = [spectral_C(M.normalise(build(s), M.SCALE_UM), seed=s) for s in range(n)]
    return tuple(np.nanmean([x[i] for x in v]) for i in range(3))


d32, d52 = 0.047, 0.078
p1 = RF.match_pitch([0.0], d32)
p2 = RF.match_pitch([0.0, 90.0], d52)
n_iso = RF.match_isotropic(d52)

print("REFERENCE CORNERS -- spectral C2/C4, density-matched")
print(f"{'field':<40s}{'C2':>8s}{'C4':>8s}{'snr':>8s}")
cases = [("single family (nematic)",     lambda s: RF.families(s, [0.0], p1)),
         ("two families 90 deg (quartic)", lambda s: RF.families(s, [0.0, 90.0], p2)),
         ("two families 60 deg",         lambda s: RF.families(s, [0.0, 60.0], p2)),
         ("isotropic",                   lambda s: RF.isotropic(s, n_iso))]
for nm, fn in cases:
    c2, c4, snr = avg(fn)
    print(f"{nm:<40s}{c2:>8.3f}{c4:>8.3f}{snr:>8.2f}")

print("\nCURVATURE ROBUSTNESS -- ideal fields degraded by curl")
print(f"{'field':<40s}{'C2':>8s}{'C4':>8s}")
for curl in (0.0, 1.0, 2.0, 3.0):
    c2, c4, _ = avg(lambda s: curved_field(s, [0.0, 90.0], 9.73, curl, 1.0))
    print(f"{'two families 90 deg, curl %.1f' % curl:<40s}{c2:>8.3f}{c4:>8.3f}")
for curl in (0.0, 2.0, 3.0):
    c2, c4, _ = avg(lambda s: curved_field(s, [0.0], 8.42, curl, 1.0))
    print(f"{'single family, curl %.1f' % curl:<40s}{c2:>8.3f}{c4:>8.3f}")

print("\nTRACED SET")
print(f"{'stage':>7s}{'C2':>8s}{'C4':>8s}{'snr':>8s}")
for hpf in (32, 36, 40, 44, 48, 52):
    fs = sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif"))
    if not fs:
        continue
    v = [spectral_C(M.load_trace(f)[0], seed=i) for i, f in enumerate(fs)]
    print(f"{hpf:>5d}hpf{np.nanmean([x[0] for x in v]):>8.3f}"
          f"{np.nanmean([x[1] for x in v]):>8.3f}{np.nanmean([x[2] for x in v]):>8.2f}")
