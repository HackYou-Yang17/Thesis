"""v13 — the retune on the FIXED nucleation rule. L_dgo only.

    L_dgo = 2*X_density + 1.5*X_gap + 1*X_order      (moderated SDs)

WHY THIS ROUND EXISTS. cell_particle._seed_angle no longer snaps the firing direction to
whichever lattice direction points at the cell centroid; it fires along the long axis into the
cell. That removes a dead band one cell-height wide on each flat membrane and takes the
nucleation-competent site count from 159 to 222 (SEED_ANGLE.md). The change is NOT neutral: at
v12's parameters, with nucleation events per step held fixed, L_dgo moves +0.600 (p = 0.031,
six-seed resolution 0.46). So v12's rates were partly compensating for the artefact and the tune
has to be redone on the corrected model.

WHERE THE SEARCH IS CENTRED. rate_nucleate is per-SITE, so 159 -> 222 raises nucleation pressure
39.6 % at unchanged rate. The natural re-centre is v12 with rate_nucleate * 159/222 = 0.0565,
which holds events per step at the value v12 was tuned to. Both that point and v12-as-is are
carried as explicit base cases so the round can be scored against either.

DESIGN, unchanged from v12 and forced by NOISE_STUDY.md. Smallest detectable L_dgo difference is
0.65 at 3 seeds, 0.46 at 6, 0.25 at 20. So: LARGE moves (+-40 %); SIX common screening seeds;
finalists re-scored at TWENTY; the single winner then run on TEN HELD-OUT seeds that took no part
in selection, because a 20-seed selection value is still optimistic (v12 measured -0.378 at its
selection seeds and -0.002 held out). A second round runs only if round one beats the base by
more than 0.46 — otherwise the round is declared converged rather than descending into noise.

A LOCAL HYPERCUBE, not axis moves: the two structural findings of this project (the
nematic_thresh x angle_noise interaction, and the branch/axis_spread degeneracy) are both
interactions that one-at-a-time screening cannot see.

PREDICTION, RECORDED BEFORE THE RUN.
  (1) rate_nucleate will settle NEAR the rescaled value, not near v12's, because the arm-B/arm-C
      pair showed the model is sensitive to nucleation EVENTS and only weakly to rate itself.
  (2) The tune will land at a LOWER thin:grow than v12's 0.2381. Reason: the fix already closes
      the mesh on its own (foam_ge8 at 44 hpf 0.583 -> 1.047 against traced 1.43), so the
      pressure that pushed v12 to a high thin:grow is partly satisfied by geometry, and the
      density/gap cost the fix incurs is what the rates now have to buy back.
  (3) The retuned L_dgo will land BETWEEN v12's held-out 3.536 and the +0.600 the fix costs
      unretuned -- i.e. the rates will recover part but not all of it. If it recovers all of it,
      the artefact was free and the fix is a strict improvement; if it recovers none, the dead
      band was doing real work and that has to be said plainly.

usage: python3 v13_tune.py r1
       python3 v13_tune.py r2 <best_case_name>
"""
from __future__ import annotations

import json, sys
import numpy as np
from scipy.stats import qmc

SITE_SCALE = 159.0 / 222.0

V12 = dict(rate_grow=0.0082531, rate_nematic_depoly=0.0068080, rate_nematic_poly=0.0015298,
           rate_branch=0.0019257, rate_nucleate=0.0788939, nematic_thresh=0.35,
           angle_noise=0.6950862, axis_spread=0.3882413, cadherin_nucleation_prob=0.3486433,
           rate_thin=0.0019649, n_sub=4, phi_max=None)

BASE = {**V12, "rate_nucleate": V12["rate_nucleate"] * SITE_SCALE}

NBR = 24
KEYS = ["rate_grow", "rate_nematic_depoly", "rate_nematic_poly", "rate_branch",
        "rate_nucleate", "angle_noise", "axis_spread", "cadherin_nucleation_prob"]
LIMITS = dict(angle_noise=(0.05, 0.785398), axis_spread=(0.05, 0.785398),
              cadherin_nucleation_prob=(0.0, 1.0))


def probe(tag, base, span, rng_seed):
    s = qmc.LatinHypercube(d=len(KEYS) + 2, scramble=True, seed=rng_seed,
                           optimization="random-cd").random(NBR)
    out = {tag: dict(base)}
    for i, row in enumerate(s):
        k = dict(base)
        for j, key in enumerate(KEYS):
            k[key] = float(base[key] * (1 - span + 2 * span * row[j]))
            if key in LIMITS:
                lo, hi = LIMITS[key]
                k[key] = float(min(max(k[key], lo), hi))
        # rate_thin moves through the RATIO: thin and grow are coupled and thin:grow is the axis
        # that trades mesh closure against fibre amount, so it must be free.
        k["rate_thin"] = float(base["rate_thin"] * (1 - span + 2 * span * row[-2])
                               * k["rate_grow"] / base["rate_grow"])
        k["nematic_thresh"] = 0.20 if row[-1] < 0.5 else 0.35
        out[f"{tag}_{i:02d}"] = k
    return out


if __name__ == "__main__":
    if sys.argv[1] == "r1":
        cases = probe("V13a", BASE, 0.40, 7301)
        cases["V13_v12asis"] = dict(V12)          # the old tune, unretuned, on the fixed code
        json.dump(cases, open("V13a_cases.json", "w"), indent=1)
        print(f"{len(cases)} configs -> V13a_cases.json  (+-40 % around v12 rescaled to 222 sites)")
    else:
        best = sys.argv[2]
        base = json.load(open("V13a_cases.json"))[best]
        json.dump(probe("V13b", base, 0.25, 7302), open("V13b_cases.json", "w"), indent=1)
        print(f"{NBR + 1} configs -> V13b_cases.json  (+-25 % around {best})")
