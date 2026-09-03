"""RECOVERED WRITER for traced_per_image_fib.csv — the traced side of the fibre-length statistic.

WHY THIS FILE EXISTS. `traced_per_image_fib.csv` is one of the two tables the fibre-length
residual is computed from (the model side is V13FIB_runs.csv, written by run_fib_v13.py), and
NO SCRIPT ON DISK PRODUCED IT. It was built inline during the fibre-length session. Same class of
provenance hole as V13F_cases.json, and closed the same way: the procedure is re-derived here and
the claim is checked against the file that was actually used.

THE PROCEDURE, and the evidence for it. traced_per_image_fib.csv is traced_targets.py's
measurement re-run through the EXTENDED measure.py — the one carrying fibres(). Evidence: all
twelve columns the two tables share (density, order, foam, foam_ge8, gap_p90, gap_p95, coverage,
count, seg_len_um, frag, junc, hpf) are IDENTICAL to the last bit in traced_per_image.csv, row for
row, and the only additions are fib_count, fib_len_um and fib_censored_frac. That is exactly what
"additive, every other key byte-identical" predicts, and it is why the fibre-length work did not
need any earlier traced measurement re-checked.

    verified 3 Sep 2026: 18 rows, 12 shared columns, max |difference| = 0 on every one.

WHAT IS NOT VERIFIED. This script has NOT been re-run against the TIFFs — the images live on the
OneDrive side and were not staged for this recovery. It reproduces the recorded procedure and the
column set; it is not a re-measurement. Run it, and the assertion at the bottom will say whether
the recovery is exact. If it fails, believe the CSV and not this file.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

import measure as M                    # MUST be the extended one, the copy in A_fibrelength/

FIB_COLS = ["fib_count", "fib_len_um", "fib_censored_frac"]
COLS = ["hpf", "density", "order", "foam", "foam_ge8", "gap_p90", "gap_p95", "coverage",
        "count", "seg_len_um", "frag", "junc"] + FIB_COLS

if __name__ == "__main__":
    assert hasattr(M, "fibres"), (
        "measure.py has no fibres() — this is the base measurement layer from the analysis root, "
        "not the extended one in A_fibrelength/. fib_len_um cannot be computed with it.")

    from datapaths import ROOT as DATA_ROOT
    files = sorted(glob.glob(DATA_ROOT + "/*/*Copy.tif"),
                   key=lambda f: (int(re.search(r"(\d+)hpf", f).group(1)), f))
    files = [f for f in files if "Copy (2)" not in f]     # the RED traces, the 18-heart set
    assert len(files) == 18, files

    rows = []
    for f in files:
        hpf = float(re.search(r"(\d+)hpf", f).group(1))
        name = os.path.basename(f).replace(" - Copy.tif", "")
        m, s_in = M.load_trace(f)
        r = M.measure_all(m)
        r["hpf"] = hpf
        rows.append(r)
        print(f"  {name:9s} {hpf:3.0f} hpf  fib {r['fib_len_um']:6.2f} um  "
              f"n {r['fib_count']:3d}  censored {r['fib_censored_frac']:.3f}", flush=True)

    df = pd.DataFrame(rows)[COLS]
    df.to_csv("traced_per_image_fib_RECOVERED.csv", index=False)
    print(f"\n{len(df)} rows -> traced_per_image_fib_RECOVERED.csv")

    # ---- the check this file exists to make -------------------------------
    try:
        ref = pd.read_csv("traced_per_image_fib.csv")
    except FileNotFoundError:
        raise SystemExit("no traced_per_image_fib.csv beside this script — nothing to check against")
    # compare after an identical CSV round-trip: in-memory float64 differs from the
    # parsed file at the last bit, which is representation, not measurement
    a = pd.read_csv("traced_per_image_fib_RECOVERED.csv")[COLS].sort_values("hpf").reset_index(drop=True)
    b = ref[COLS].sort_values("hpf").reset_index(drop=True)
    worst = {c: float(np.abs(a[c].to_numpy() - b[c].to_numpy()).max()) for c in COLS}
    bad = {c: v for c, v in worst.items() if v > 0}
    print("reproduces traced_per_image_fib.csv exactly:", not bad)
    if bad:
        print("  columns that differ, and by how much:")
        for c, v in sorted(bad.items(), key=lambda kv: -kv[1]):
            print(f"    {c:20s} {v:.6g}")
        raise SystemExit("MISMATCH — the recovered procedure does not reproduce the file")
