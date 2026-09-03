"""STEP 4 — re-run the calibrated v13 point so fib_len_um exists for the model.

The stored V13*_runs.csv predate fibres(), so the statistic cannot be back-filled from them.

ONE TRAP, and the brief walks into it. The brief says "parameters: bundle_model/parameters.py as
committed (v13)" and "harness: sim3.run_once(seed=S, hpf=[...])". But sim3.configure() sets EVERY
FLOAT_KEY from its own ANCHOR dict unless overridden, and ANCHOR is the v4/v5 rate set, not v13:

    rate_grow    ANCHOR 0.0283   parameters.py 0.0049036     (5.8x)
    angle_noise  ANCHOR 0.3142   parameters.py 0.7132        (2.3x)
    rate_nucleate ANCHOR 0.120   parameters.py 0.0590        (2.0x)

So run_once(seed=S) with no overrides would have quietly produced a v5 run labelled v13. The
overrides are therefore READ OFF P AT IMPORT, before any configure() call, so the run uses exactly
what parameters.py commits and no value is transcribed by hand.

Seeds: 7301-7320 (the v13 selection seeds) and 7501-7510 (held out). 30 in total, as specified.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import sim3
from bundle_model.cell_particle import P

# snapshot parameters.py BEFORE anything calls configure()
V13 = {k: float(getattr(P, k)) for k in sim3.FLOAT_KEYS}
V13.update({k: int(getattr(P, k)) for k in sim3.INT_KEYS})
V13["phi_max"] = None

SEEDS = list(range(7301, 7321)) + list(range(7501, 7511))
HPF = [32.0, 36.0, 40.0, 44.0, 48.0, 52.0]
COLS = ["case", "seed", "hpf", "density", "order", "foam", "foam_ge8", "gap_p90", "gap_p95",
        "coverage", "count", "seg_len_um", "frag", "junc", "fib_count", "fib_len_um",
        "fib_censored_frac", "phi", "n_nem", "n_mesh", "drift_pct"]


def _one(seed):
    t0 = time.time()
    out = sim3.run_once(V13, seed=seed, hpf=HPF)
    rows = [dict(case="V13FIB", seed=seed, hpf=h, **out[h]) for h in HPF]
    print(f"  seed {seed}  {time.time() - t0:5.0f}s  "
          f"fib52 {out[52.0]['fib_len_um']:.2f} um  seg52 {out[52.0]['seg_len_um']:.2f}",
          flush=True)
    return rows


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
    print("v13 overrides read from parameters.py:")
    for k in sim3.FLOAT_KEYS:
        print(f"  {k:26s} {V13[k]:.7g}")
    t0 = time.time()
    with Pool(procs) as p:
        out = p.map(_one, SEEDS)
    d = pd.DataFrame([r for rs in out for r in rs])
    d = d[[c for c in COLS if c in d.columns]]
    d.to_csv("V13FIB_runs.csv", index=False)
    print(f"\n{len(d)} rows -> V13FIB_runs.csv in {(time.time() - t0) / 60:.1f} min")
