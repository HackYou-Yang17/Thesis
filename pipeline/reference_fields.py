"""Reference fields of KNOWN structure, put through the identical measurement
pipeline (xcheck/v5/measure.py) at MATCHED LINE DENSITY.

Purpose: the five traced statistics are pipeline-specific and unitless, so a
value like foam = 1.81 carries no scale on its own. These references supply the
scale -- what a single family, two families at a known separation, and a
disordered field of the SAME line density each score.

Matching on density matters: without it, an ideal family drawn at the measured
pitch is 2-3x denser than the traced tissue and every comparison is really a
comparison of how much line is in the field.

NOTE on the isotropic search: density is NOT monotonic in line count. Past
~200 lines the field saturates, skeletonising a filled mask returns almost
nothing, and density collapses to ~0. The search is therefore confined to the
rising branch (n <= 120).
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from skimage.morphology import dilation, disk
from skimage.draw import line as draw_line
import measure as M

N = int(round(M.FIELD_UM / M.SCALE_UM))
TRACED = {"32hpf": dict(density=0.047, order=0.78, foam=0.21, gap_p95=8.14, count=29),
          "52hpf": dict(density=0.078, order=0.27, foam=1.81, gap_p95=5.82, count=62)}


def draw(angle_offsets):
    img = np.zeros((N, N), bool)
    for a, o in angle_offsets:
        c, s = np.cos(a), np.sin(a)
        px, py = o * (-s) + N / 2, o * c + N / 2
        L = 3 * N
        rr, cc = draw_line(int(py - L * s), int(px - L * c),
                           int(py + L * s), int(px + L * c))
        ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
        if ok.sum() < 10:
            continue
        img[rr[ok], cc[ok]] = True
    return dilation(img, disk(M.WIDTH_PX // 2))


def families(seed, separations_deg, pitch_um, jitter_deg=10.0):
    rng = np.random.default_rng(seed)
    j = np.deg2rad(jitter_deg)
    ao = []
    for sep in separations_deg:
        for o in np.arange(-N, 2 * N, pitch_um / M.SCALE_UM):
            ao.append((np.deg2rad(sep) + rng.uniform(-j, j), o))
    return draw(ao)


def isotropic(seed, n_lines):
    # offsets confined to the range that actually intersects the field
    rng = np.random.default_rng(seed)
    return draw([(rng.uniform(0, np.pi), rng.uniform(-0.75 * N, 0.75 * N))
                 for _ in range(n_lines)])


def agg(build, n_seeds=5):
    vs = [M.measure_all(M.normalise(build(s), M.SCALE_UM), seed=s) for s in range(n_seeds)]
    return {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}


def match_pitch(separations, target_density, lo=2.0, hi=40.0):
    """Larger pitch -> fewer lines -> lower density. Monotone, so bisect."""
    for _ in range(18):
        mid = (lo + hi) / 2
        if agg(lambda s: families(s, separations, mid), 3)["density"] > target_density:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def match_isotropic(target_density, n_max=120):
    """Scan the rising branch only; density collapses past saturation."""
    best, best_err = 1, 1e9
    for n in range(1, n_max + 1):
        d = agg(lambda s: isotropic(s, n), 3)["density"]
        err = abs(d - target_density)
        if err < best_err:
            best, best_err = n, err
        if d > target_density * 1.6:
            break
    return best


if __name__ == "__main__":
    d32, d52 = TRACED["32hpf"]["density"], TRACED["52hpf"]["density"]
    p1 = match_pitch([0.0], d32)
    p2 = match_pitch([0.0, 90.0], d52)
    p3 = match_pitch([0.0, 60.0], d52)
    n_iso = match_isotropic(d52)
    print(f"density-matched: single family pitch {p1:.2f} um, "
          f"two families 90 deg pitch {p2:.2f} um, 60 deg pitch {p3:.2f} um, "
          f"isotropic {n_iso} lines\n")

    rows = [
        ("single family, matched to 32 hpf",   lambda s: families(s, [0.0], p1)),
        ("two families 90 deg, matched to 52 hpf", lambda s: families(s, [0.0, 90.0], p2)),
        ("two families 60 deg, matched to 52 hpf", lambda s: families(s, [0.0, 60.0], p3)),
        ("isotropic, matched to 52 hpf",       lambda s: isotropic(s, n_iso)),
    ]
    hdr = ["density", "order", "foam", "gap_p95", "count"]
    print(f"{'reference field':<42s}" + "".join(f"{h:>11s}" for h in hdr))
    for nm, fn in rows:
        a = agg(fn)
        print(f"{nm:<42s}" + "".join(f"{a[h]:>11.3f}" for h in hdr))
    print()
    for k, v in TRACED.items():
        print(f"{'TRACED ' + k:<42s}" + "".join(f"{v[h]:>11.3f}" for h in hdr))
