"""CONVERGENCE AT FIXED RATES — does the fit depend on the discretisation or on the rates?

    [stated] Luka: "Establish whether the five measured statistics depend on the discretisation
    rather than on the rates. Every rate k is held FIXED at the calibrated values. This is a
    validation pass, not a search. Do not re-tune anything."

Run against v13, the tune on the corrected nucleation rule. NOTHING is fitted here.

WHY THE MEAN IS INVARIANT BY CONSTRUCTION. Growth and disassembly use lambda = k*dt*n_sub and
move monomer_quantum = monomers_per_seg/n_sub per event, so mean mass flux per unit time is
k*monomers_per_seg and carries no n_sub. Nucleation and branching use lambda = k*dt with NO
n_sub because they create objects rather than move mass. Halving dt halves every lambda and
doubles the step count (P.steps is a property: total_hours * 3600/dt), so simulated time and
mean flux are both unchanged. What is NOT invariant is everything downstream of discreteness:
realised morphology from partial point occupancy, how finely the rejection test
avail/monomers_per_point samples local availability, the two dt-linear boundary errors, and the
timing resolution of the nematic_thresh clearance gate.

CONDITIONS, common random seeds throughout, every comparison PAIRED
    A   dt 1.0  n_sub  4   cadherin_every 200    baseline, 72,000 steps
    B   dt 0.5  n_sub  4   cadherin_every 400    144,000 steps
    C   dt 1.0  n_sub  8   cadherin_every 200
    D   dt 1.0  n_sub 16   cadherin_every 200
    E   dt 1.0  n_sub  1   cadherin_every 200    THROWAWAY POWER PROBE, not part of the answer
    B2  dt 0.5  n_sub  4   cadherin_every 200    the dt trap the brief did not list -- see below

A TRAP THE BRIEF DID NOT LIST, and it would have silently broken condition B.
MultiCell.step fires cadherin coupling every `cadherin_every` STEPS, not every so many seconds.
At dt = 0.5 that is twice as often per simulated hour, which is a change in the MODEL, not in
the discretisation. Condition B therefore scales it to 400 so the coupling cadence is fixed in
simulated time. B2 leaves it at 200 and is included precisely to show what that trap costs, so
that if B and B2 differ the difference is attributable and not mixed into the dt result.

THE THREE TRAPS THE BRIEF DID LIST are all checked in code and reported:
  1. total simulated time is held at 20 h -- asserted, since P.steps is derived from dt;
  2. nucleation and branching lambdas carry no n_sub -- asserted against Params;
  3. total polymer mass against simulated time must overlay for A, C, D -- reported as phi at
     all six timepoints, which is exactly that quantity.

usage: python3 converge13.py [--seeds 6] [--procs 2]
"""
from __future__ import annotations

import json, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

V13 = json.load(open("V13_TUNE.json"))["V13"]
SEEDS = [7801, 7802, 7803, 7804, 7805, 7806]

CONDITIONS = {
    "A_dt1_ns4":   dict(dt=1.0, n_sub=4,  cad=200),
    "B_dt05_ns4":  dict(dt=0.5, n_sub=4,  cad=400),
    "C_dt1_ns8":   dict(dt=1.0, n_sub=8,  cad=200),
    "D_dt1_ns16":  dict(dt=1.0, n_sub=16, cad=200),
    "E_dt1_ns1":   dict(dt=1.0, n_sub=1,  cad=200),
    "B2_dt05_cad200": dict(dt=0.5, n_sub=4, cad=200),
}


def check_lambdas():
    """Trap 2, asserted rather than assumed."""
    import sim3
    from bundle_model.cell_particle import P
    sim3.configure({**V13, "n_sub": 4})
    P.dt = 1.0
    n4 = (P.k_nucleate, P.k_branch, P.k_grow, P.steps)
    sim3.configure({**V13, "n_sub": 16})
    P.dt = 1.0
    n16 = (P.k_nucleate, P.k_branch, P.k_grow, P.steps)
    assert np.isclose(n4[0], n16[0]) and np.isclose(n4[1], n16[1]), "nucleate/branch carry n_sub!"
    assert np.isclose(n16[2], 4 * n4[2]), "k_grow does not scale with n_sub"
    sim3.configure({**V13, "n_sub": 4})
    P.dt = 0.5
    half = (P.k_nucleate, P.k_grow, P.steps)
    assert np.isclose(half[0], n4[0] / 2) and np.isclose(half[1], n4[2] / 2), "dt not in lambda"
    assert half[2] == 2 * n4[3], f"step count did not double: {half[2]} vs {n4[3]}"
    return dict(k_nucleate_ns4=n4[0], k_nucleate_ns16=n16[0], k_grow_ns4=n4[2],
                k_grow_ns16=n16[2], steps_dt1=int(n4[3]), steps_dt05=int(half[2]))


def _one(args):
    cond, cfg, seed = args
    import sim3
    import measure as M
    from bundle_model import multicell_particle as mc
    from bundle_model.cell_particle import P
    t0 = time.time()
    sim3.configure({**V13, "n_sub": cfg["n_sub"]})
    P.dt = cfg["dt"]                     # P.steps and every lambda follow from this
    steps = int(P.steps)
    marks = {int(round(steps * (h - 32.0) / P.total_hours)): h for h in sim3.HPF}
    sheet = mc.MultiCell(base_seed=seed, cadherin_every=cfg["cad"])
    rows, last = [], max(marks)
    for step in range(last + 1):
        if step in marks:
            r = M.measure_all(sim3.field(sheet))
            r["phi"] = sim3.raw_occupancy(sheet)
            rows.append(dict(case=cond, seed=seed, hpf=marks[step], steps=steps,
                             dt=cfg["dt"], n_sub=cfg["n_sub"], cad=cfg["cad"], **r))
        if step < last:
            sheet.step()
    print(f"  {cond:16s} seed {seed}  {steps} steps  {time.time()-t0:6.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
    checks = check_lambdas()
    print("TRAP CHECKS")
    for k, v in checks.items():
        print(f"  {k:18s} {v}")
    print(f"  20 h at dt 1.0 = {checks['steps_dt1']} steps; at dt 0.5 = {checks['steps_dt05']}",
          flush=True)
    json.dump(checks, open("CONV13_checks.json", "w"), indent=1)

    jobs = [(c, cfg, s) for c, cfg in CONDITIONS.items() for s in SEEDS]
    done = []
    with Pool(procs) as pool:
        for rows in pool.imap_unordered(_one, jobs, chunksize=1):
            done.extend(rows)
            pd.DataFrame(done).to_csv("CONV13_runs.csv", index=False)
    print("wrote CONV13_runs.csv")
