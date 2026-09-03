"""RECONSTRUCTS the v13 finalist set, and asserts it reproduces V13F_cases.json exactly.

WHY THIS FILE EXISTS. The finalist selection is the step that decided which tune the thesis
reports, and NO SCRIPT ON DISK PRODUCED IT -- V13F_cases.json was built ad hoc inline during the
v13 session and the code was never saved. Every other stage of the round has its script
(v13_tune.py, multistart13.py, sens_v13.py, aspect_sweep13.py); this one did not. That is a
provenance hole in the most decision-relevant step of the round, so it is closed here by
re-deriving the set from the archived runs and checking it against the file that was actually used.

THE RULE, recovered by matching parameter vectors back to their rounds:
    the TOP 2 of round 1 at 6 screening seeds        -> V13a_13, V13a_20
    the TOP 3 of round 2 at 6 screening seeds        -> V13b_03, V13b_07, V13b_18
    plus v12-as-is on the fixed code as a REFERENCE ARM (not a candidate)
Round 2's centre IS round 1's winner (V13b == V13a_13), so it is not double-counted; that is why
five distinct configurations come out of "2 + 3" rather than six.

Round-1 order (6 seeds): V13a_13 3.676, V13a_20 3.700, V13a_17 3.820, V13_v12asis 4.081
Round-2 order (6 seeds): V13b_03 3.644, V13b 3.676, V13b_07 3.795, V13b_18 3.795

WHAT THIS DOES NOT DO. It reproduces the SET; it does not reproduce the CHOICE within the set.
Re-scored at 14-20 seeds the five spanned 0.284 against a resolution floor of ~0.35, i.e. a
five-way tie, and the loss did not select. The tune was picked on two standing criteria recorded
in V13_TUNING.md: (i) NOT PINNED AT A BOUND -- two finalists sat exactly at angle_noise = pi/4,
the search-box ceiling, which is a statement about the box and not the data; (ii) CROSSOVER
PRECISION, the seed-count-invariant replacement for the retired "range < 4 h" rule. The adopted
tune, F_b07, was FOURTH of the five on the loss.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import loss_lab as L

KEYS = ["rate_grow", "rate_nematic_depoly", "rate_nematic_poly", "rate_branch",
        "rate_nucleate", "angle_noise", "axis_spread", "cadherin_nucleation_prob",
        "rate_thin", "nematic_thresh"]
N_R1, N_R2 = 2, 3


def score(runs_csv, cases, T):
    d = pd.read_csv(runs_csv)
    out = {}
    for c, g in d.groupby("case"):
        if c not in cases:
            continue
        arr = {k: g.groupby("hpf")[k].mean().reindex(L.HPF).to_numpy() for k in L.METRICS}
        v = L.evaluate_all(arr, T)["L_dgo"]
        if np.isfinite(v):
            out[c] = v
    return pd.Series(out).sort_values()


def main():
    T = L.Targets()
    A = json.load(open("V13a_cases.json"))
    B = json.load(open("V13b_cases.json"))
    r1 = score("V13a_runs.csv", A, T)
    r2 = score("V13b_runs.csv", B, T)
    print("round 1 (6 seeds):", ", ".join(f"{k} {v:.3f}" for k, v in r1.head(4).items()))
    print("round 2 (6 seeds):", ", ".join(f"{k} {v:.3f}" for k, v in r2.head(4).items()))

    def vec(c):
        return tuple(round(float(c[k]), 12) for k in KEYS)

    picked, seen = [], set()
    for name in [n for n in r1.index if n != "V13_v12asis"][:N_R1]:
        picked.append((name, A[name])); seen.add(vec(A[name]))
    for name in r2.index:
        if len(picked) >= N_R1 + N_R2:
            break
        if vec(B[name]) in seen:          # round 2's centre is round 1's winner
            continue
        picked.append((name, B[name])); seen.add(vec(B[name]))

    print(f"\n{len(picked)} distinct finalists: {[n for n, _ in picked]}")

    F = json.load(open("V13F_cases.json"))
    ref = {n: c for n, c in F.items() if n != "F_v12asis"}
    got = {vec(c) for _, c in picked}
    want = {vec(c) for c in ref.values()}
    print(f"reproduces V13F_cases.json (excluding the v12 reference arm): {got == want}")
    if got != want:
        raise SystemExit("MISMATCH — the recovered rule does not reproduce the file")
    json.dump({n: c for n, c in picked}, open("V13F_reconstructed.json", "w"), indent=1)
    print("-> V13F_reconstructed.json")


if __name__ == "__main__":
    main()
