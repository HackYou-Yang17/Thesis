"""How much of the traced C4 deficit is fibre REALISM rather than arrangement?

The straight/unbroken reference fields set C4 = 0.93 for two families at 90 deg,
against a traced 52 hpf value of 0.37. But traced fibres curve and are broken by
skeletonisation at every junction, and both depress the branch-angle order
parameters. This sweeps curvature and breakage on an otherwise IDEAL 90 deg
two-family field to bound that depression.

curl  = s.d. of the angular random walk per step, degrees (0 = straight)
keep  = fraction of each fibre retained (1 = unbroken)
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from skimage.morphology import dilation, disk
import measure as M
import order_params as OP
from traced_c4 import branch_angles_weighted

N = int(round(M.FIELD_UM / M.SCALE_UM))
STEP = 2.0  # px per polyline step


def curved_field(seed, separations_deg, pitch_um, curl_deg, keep, jitter_deg=10.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N), bool)
    j = np.deg2rad(jitter_deg)
    curl = np.deg2rad(curl_deg)
    for sep in separations_deg:
        for o in np.arange(-N, 2 * N, pitch_um / M.SCALE_UM):
            a = np.deg2rad(sep) + rng.uniform(-j, j)
            c, s = np.cos(a), np.sin(a)
            x, y = o * (-s) + N / 2 - 1.5 * N * c, o * c + N / 2 - 1.5 * N * s
            for _ in range(int(3 * N / STEP)):
                a += rng.normal(0, curl)          # angular random walk
                x += STEP * np.cos(a); y += STEP * np.sin(a)
                if rng.random() > keep:            # drop this step -> a gap
                    continue
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < N and 0 <= yi < N:
                    img[yi, xi] = True
    return dilation(img, disk(M.WIDTH_PX // 2))


def stats(build, n_seeds=5):
    out = []
    for s in range(n_seeds):
        m = M.normalise(build(s), M.SCALE_UM)
        a, w = branch_angles_weighted(m)
        if len(a) < 4:
            continue
        r = OP.analyse(a, w)
        out.append((M.density(m), r["C2"], r["C4"]))
    d, c2, c4 = np.mean(out, axis=0)
    return d, c2, c4


print("IDEAL 90 deg TWO-FAMILY FIELD, degraded toward realism")
print(f"{'curl/step':>10s}{'keep':>7s}{'density':>9s}{'C2':>8s}{'C4':>8s}")
for curl in (0.0, 2.0, 5.0, 10.0):
    for keep in (1.0, 0.7, 0.4):
        d, c2, c4 = stats(lambda s: curved_field(s, [0.0, 90.0], 9.73, curl, keep))
        print(f"{curl:>10.1f}{keep:>7.2f}{d:>9.4f}{c2:>8.3f}{c4:>8.3f}")
print()
print(f"{'TRACED 52 hpf':<17s}{0.078:>9.4f}{0.403:>8.3f}{0.368:>8.3f}")
print(f"{'TRACED 32 hpf':<17s}{0.047:>9.4f}{0.858:>8.3f}{0.578:>8.3f}")
