"""Parallel sweep driver. One row per (case, seed); every metric at every timepoint kept.

Usage:  python3 sweep.py <name> <cases.json> [--seeds 42 7 101] [--procs 2]
`cases.json` is {"case name": {knob: value, ...}, ...}.
Writes <name>_runs.csv (one row per case x seed x hpf) and <name>_summary.csv
(one row per case, seed-averaged, with the v5 loss).
"""
from __future__ import annotations

import json, sys, time, os
from multiprocessing import Pool

import numpy as np
import pandas as pd


def _one(args):
    name, knobs, seed = args
    import sim3
    t0 = time.time()
    out = sim3.run_once(knobs, seed=seed)
    rows = []
    for h, r in out.items():
        rows.append(dict(case=name, seed=seed, hpf=h, **r))
    print(f"  done {name:28s} seed {seed:4d}  {time.time()-t0:6.1f}s", flush=True)
    return rows


def run(cases, seeds=(42,), procs=2, out_prefix="sweep"):
    jobs = [(n, k, s) for n, k in cases.items() for s in seeds]
    done = []
    with Pool(procs) as pool:
        for rows in pool.imap_unordered(_one, jobs, chunksize=1):
            done.extend(rows)                       # write after EVERY completion so a
            pd.DataFrame(done).to_csv(f"{out_prefix}_runs.csv", index=False)  # kill loses nothing
    df = pd.DataFrame(done)
    return df, summarise(df, cases)


def summarise(df, cases=None, targets_csv="traced_per_image.csv"):
    """Score every case under every candidate loss (loss_lab), seed-averaged."""
    import loss_lab as LL
    T = LL.Targets(targets_csv)
    rows = []
    for name, g in df.groupby("case", sort=False):
        arrays = {k: g.groupby("hpf")[k].mean().reindex(LL.HPF).to_numpy() for k in LL.METRICS}
        r = LL.evaluate_all(arrays, T)
        r["case"] = name
        r["n_seeds"] = int(g.seed.nunique())
        if len(g.seed.unique()) > 1:
            per = []
            for _, gs in g.groupby("seed"):
                a = {k: gs.sort_values("hpf")[k].to_numpy(float) for k in LL.METRICS}
                per.append(LL.evaluate_all(a, T))
            for nm in LL.LOSSES:
                r[f"sd_L_{nm}"] = float(np.std([p[f"L_{nm}"] for p in per], ddof=1))
        if cases:
            r["knobs"] = json.dumps(cases[name])
        rows.append(r)
    front = ["case", "n_seeds"] + [f"L_{k}" for k in LL.LOSSES] + ["L_ref", "xc", "gate"]
    df2 = pd.DataFrame(rows)
    return df2[front + [c for c in df2.columns if c not in front]]


if __name__ == "__main__":
    name = sys.argv[1]
    cases = json.load(open(sys.argv[2]))
    seeds = (42,)
    procs = 2
    if "--seeds" in sys.argv:
        i = sys.argv.index("--seeds") + 1
        out = []
        while i < len(sys.argv) and sys.argv[i].isdigit():   # stop at the next --flag
            out.append(int(sys.argv[i])); i += 1
        seeds = tuple(out)
    if "--procs" in sys.argv:
        procs = int(sys.argv[sys.argv.index("--procs") + 1])
    t0 = time.time()
    df, summ = run(cases, seeds=seeds, procs=procs, out_prefix=name)
    summ.to_csv(f"{name}_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    import loss_lab as LL
    cols = ["case"] + [f"L_{k}" for k in LL.LOSSES] + ["L_ref", "xc", "gate"]
    print("\n" + summ.sort_values("L_ref")[cols].to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))
    print(f"\ntotal {time.time()-t0:.0f}s")
