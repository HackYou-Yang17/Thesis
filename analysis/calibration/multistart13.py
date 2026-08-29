"""Multi-start on the FIXED nucleation rule (v13). Trimmed budget -- see BUDGET below.

    L_dgo = 2*X_density + 1.5*X_gap + 1*X_order      (moderated SDs, crossover held out)

THE QUESTION. Every round so far has been one pool -> one refinement, so every near-optimal
configuration on file is a descendant of one or two centres. That makes the existing data
useless for identifiability: parameters look pinned only because the search never started
anywhere else. Six INDEPENDENT starts, spread as far apart as the box allows, each descended
under an identical rule, gives three distinguishable outcomes:

  same parameters, same loss   -> a single well-defined optimum
  different parameters, same loss -> the model is NON-IDENTIFIABLE on this loss
  different parameters, different loss -> genuinely multimodal; the tune is a local optimum

COMMON RANDOM NUMBERS. Every evaluation uses the same seed set, and the seeds are FRESH
(5001-5004) rather than the 42/7/101/11/77/303 used to tune v10 and v11 -- reusing tuning
seeds would import their selection bias into a comparison that is supposed to be clean.
Measured over three configurations at twenty shared seeds, the paired SD of a difference in
L_dgo is 0.401 against 0.663 for the unpaired equivalent, so pairing buys a factor of 0.60.

PROBE AXES. All ten tunables are perturbed together. An earlier version of this script moved
six of them one at a time, chosen by their marginal rank correlation with L_dgo over the 167
gate-passing configurations on file (angle_noise -0.587, rate_grow +0.494, thin:grow -0.387,
rate_nematic_poly +0.323, rate_nucleate +0.243, rate_branch +0.207; nematic_thresh and
cadherin_nucleation_prob flat at p = 0.63 and 0.97). That was dropped: a marginal scan across
two heterogeneous pools is a weak guide, and the v7 round's central finding was an
INTERACTION (nematic_thresh 0.20 x wide angle_noise) that no one-at-a-time scheme can see.

NO GREEDY DESCENT, AND THAT IS A RESULT NOT A SHORTCUT. The 20-seed noise study
(NOISE_study.csv) puts the paired standard deviation of a difference in L_dgo at 0.401, so
the smallest difference detectable at 80 % power is 0.65 at three seeds, 0.46 at six and
0.36 at ten. Coordinate-refinement moves in this project are worth 0.05-0.30. A greedy
coordinate descent at any affordable seed count is therefore descending into noise -- which
is why refinements have repeatedly failed to beat their own centres (v10: 4.304 -> 4.308).

So each start gets ONE honest coarse step instead of several meaningless fine ones: a local
Latin-hypercube probe of 16 neighbours at +-40 % on all ten tunables, at four common seeds
(detectable difference 0.56). The best neighbour is that start's basin minimum. The six basin
minima then go to TWENTY seeds, where 0.25 is detectable, and are compared on loss AND on
distance in parameter space.

WINNER'S CURSE. The same study shows a tuning-seed estimate is optimistic by +0.37 to +0.53
because the candidate was selected for doing well on those seeds. The 20-seed re-scoring at
the end is not a formality; it is the only unbiased number in the procedure.


BUDGET, and what the trim costs. The v11 multi-start was 6 starts x 16 neighbours x 4 seeds
(408 runs) plus 6 basin minima x 20 seeds (120), about 540 runs. On the fixed code that is
roughly seven hours on this machine, so [stated] Luka asked for a trimmed run that is still
significant. This one is 6 starts x 12 neighbours x 3 seeds (234) plus 6 basin minima x 12
seeds (72) = 306 runs, a shade over half.

WHAT IS PROTECTED AND WHAT IS SPENT. The SIX INDEPENDENT STARTS are untouched -- the whole
identifiability claim rests on how far apart the starts are, and cutting starts would cut the
claim, not the cost of it. What is spent is RESOLUTION:
  * probe seeds 4 -> 3 raises the smallest detectable L_dgo difference from 0.56 to 0.65. A
    +-40 % hypercube produces differences of 0.5-1.5, so the probe still resolves the basin but
    its ranking of near-tied neighbours is noise. The basin MINIMUM is therefore a noisier
    pick than in the v11 run, which widens the parameter scatter this study reports -- i.e. the
    trim biases the result TOWARDS looking non-identifiable, not towards looking pinned.
  * neighbours 16 -> 12 thins the local hypercube in a 10-dimensional box that was already
    sparsely covered. Same direction of bias.
  * basin re-scoring 20 -> 12 seeds raises the detectable difference between basins from 0.25
    to 0.32. Basin-to-basin gaps in the v11 run were 0.1-0.9, so ties will be declared where
    the fuller run might have separated them. Same direction again.
Every one of those cuts makes agreement between starts HARDER to demonstrate. So a finding of
agreement survives the trim; a finding of disagreement has to be reported as "not resolved at
this budget" rather than as evidence of non-identifiability.

usage:  python3 multistart13.py starts     -> MS13_starts.json, MS13_r1_cases.json (78 configs)
        python3 multistart13.py basins     -> MS13_basins.json, after MS13_r1_runs.csv exists
"""
from __future__ import annotations

import json, sys
import numpy as np
import pandas as pd
from scipy.stats import qmc

K = 6                       # independent starts
SEEDS = [7401, 7402, 7403]           # fresh common random numbers  [v13: 3, was 4]
FINAL_SEEDS = list(range(7401, 7413))  # twelve, for the basin minima  [v13: was 20]

BOUNDS = dict(
    rate_grow=(0.0065, 0.0250),
    rate_nematic_depoly=(0.0070, 0.0160),
    rate_nematic_poly=(0.0020, 0.0110),
    rate_nucleate=(0.0200, 0.2400),
    angle_noise=(0.2000, 0.5500),
    axis_spread=(0.2800, 0.6800),
    cadherin_nucleation_prob=(0.0500, 0.9500),
)
BRANCH_LOG = (np.log(0.0010), np.log(0.0140))
THIN_GROW = (0.080, 0.330)
LIMITS = dict(angle_noise=(0.05, 0.785398), axis_spread=(0.05, 0.785398),
              cadherin_nucleation_prob=(0.0, 1.0))


def make_starts():
    """Maximin-spread starts over the full box, three at each nematic_thresh level."""
    keys = list(BOUNDS)
    s = qmc.LatinHypercube(d=len(keys) + 2, scramble=True, seed=131313,
                           optimization="random-cd").random(K)
    cases = {}
    for i, row in enumerate(s):
        c = {k: float(BOUNDS[k][0] + row[j] * (BOUNDS[k][1] - BOUNDS[k][0]))
             for j, k in enumerate(keys)}
        c["rate_branch"] = float(np.exp(BRANCH_LOG[0] + row[-2] * (BRANCH_LOG[1] - BRANCH_LOG[0])))
        c["rate_thin"] = float(c["rate_grow"]
                               * (THIN_GROW[0] + row[-1] * (THIN_GROW[1] - THIN_GROW[0])))
        c["nematic_thresh"] = 0.20 if i % 2 == 0 else 0.35
        c["n_sub"], c["phi_max"] = 4, None
        cases[f"S{i}"] = c
    return cases


NBR = 12          # neighbours probed per start  [v13: trimmed from 16]
SPAN = 0.40       # +-40 % on every tunable


def neighbours(name, base, rng_seed):
    """A local Latin hypercube of NBR points at +-SPAN around `base`, on all ten tunables.

    A hypercube rather than 2 x k axis moves: the same budget covers combinations as well as
    single-coordinate steps, and the v7 round's central finding was an INTERACTION
    (nematic_thresh 0.20 x wide angle_noise) that one-at-a-time screening cannot see.
    nematic_thresh is resampled between its two levels for the same reason.
    """
    keys = ["rate_grow", "rate_nematic_depoly", "rate_nematic_poly", "rate_branch",
            "rate_nucleate", "angle_noise", "axis_spread", "cadherin_nucleation_prob"]
    s = qmc.LatinHypercube(d=len(keys) + 2, scramble=True, seed=rng_seed,
                           optimization="random-cd").random(NBR)
    out = {name: dict(base)}
    for i, row in enumerate(s):
        k = dict(base)
        for j, key in enumerate(keys):
            k[key] = float(base[key] * (1 - SPAN + 2 * SPAN * row[j]))
            if key in LIMITS:
                lo, hi = LIMITS[key]
                k[key] = float(min(max(k[key], lo), hi))
        k["rate_thin"] = float(base["rate_thin"] * (1 - SPAN + 2 * SPAN * row[-2])
                               * k["rate_grow"] / base["rate_grow"])
        k["nematic_thresh"] = 0.20 if row[-1] < 0.5 else 0.35
        out[f"{name}_n{i:02d}"] = k
    return out


def best_of(runs_csv, prefix):
    """Lowest L_dgo among cases beginning with `prefix`, gates respected."""
    import loss_lab as L
    T = L.Targets()
    d = pd.read_csv(runs_csv)
    best, bl = None, np.inf
    for case, g in d.groupby("case"):
        if not case.startswith(prefix):
            continue
        a = {k: g.groupby("hpf")[k].mean().reindex(L.HPF).to_numpy() for k in L.METRICS}
        e = L.evaluate_all(a, T)
        if e["gate"] != "ok":
            continue
        if e["L_dgo"] < bl:
            best, bl = case, e["L_dgo"]
    return best, bl


if __name__ == "__main__":
    if sys.argv[1] == "starts":
        st = make_starts()
        cases = {}
        for i, (n, b) in enumerate(st.items()):
            cases.update(neighbours(n, b, 900 + i))
        json.dump(st, open("MS13_starts.json", "w"), indent=1)
        json.dump(cases, open("MS13_r1_cases.json", "w"), indent=1)
        print(f"{len(st)} starts, {len(cases)} configs -> MS13_r1_cases.json")
    else:
        # report each start's basin minimum
        import loss_lab as L
        picks = {}
        for i in range(K):
            b, bl = best_of("MS13_r1_runs.csv", f"S{i}")
            picks[f"S{i}"] = dict(case=b, L_dgo=bl)
            print(f"  S{i}: basin minimum {b}  L_dgo {bl:.3f}" if b else f"  S{i}: no gate-passing config")
        json.dump(picks, open("MS13_basins.json", "w"), indent=1)
