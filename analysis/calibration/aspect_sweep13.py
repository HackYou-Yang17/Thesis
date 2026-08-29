"""Aspect ratio at CONSTANT CELL AREA, RE-RUN AT v13 ON THE FIXED NUCLEATION RULE.

    [stated] Luka: "do the aspect ratio one where area is constant and aspect ratio changes.
    I just want to see what happens"

WHY CONSTANT AREA IS THE WHOLE POINT. Every earlier aspect sweep in this project varied cell
HEIGHT at fixed LENGTH, so area varied with it — and cell area is the strongest single driver of
density in this model (r = +0.95 to +0.98 in the v1/v2 sweeps). Those sweeps therefore measured
area, not shape, and the recorded conclusion is that the cell-height response cannot sit in the
predictions list. Holding L x H fixed removes that confound: the number of cells per crop stays
roughly constant, the monomer budget per cell is unchanged, and the only thing that moves is the
SHAPE of the domain the fibres are laid down in.

    L = sqrt(A * area),  H = sqrt(area / A),  area = 38.9 * 12.5 = 486.25 um^2

The v13 tune sits at A = 3.112. The sweep spans 1.0 (square) to 6.2 (twice as elongated).

WHAT COULD MOVE, mechanistically. The seeded array is laid down on template rows spaced
nematic_gap 7-11 lu apart along the cell's short axis, so a flatter cell fits FEWER rows: at
A = 6.2 the half-height is 12.3 lu, giving only two or three rows. Cortical nucleation scales
with PERIMETER, and at constant area the perimeter grows with aspect ratio, so nucleation
pressure per unit area rises. Those pull in opposite directions and which wins is the question.

NOTE, 26 Aug 2026: the import cache checks source mtime in WHOLE SECONDS plus size, and
"%.4f" of a two-digit number keeps the size fixed, so a fast rewrite can be missed entirely.
set_cell now fsyncs and clears __pycache__. Runs minutes apart were never at risk.

WHY IT NEEDS A FRESH PROCESS PER ASPECT RATIO. hex_half_w/h are Params fields evaluated at class
definition, and multicell_particle computes CANVAS_W/H and clamps CROP_SIZE from the tiling at
IMPORT time. Overriding P at runtime would change the cell mask but leave the canvas and crop at
the old size. So each aspect ratio edits parameters.py and runs in a subprocess.

usage: python3 aspect_sweep.py            (runs everything, restores parameters.py at the end)
"""
from __future__ import annotations

import json, os, re, shutil, subprocess, sys
import numpy as np
import pandas as pd

PARAMS = "modelling/CARMA/carma_6_particle/parameters.py"
AREA = 38.9 * 12.5
ASPECTS = [1.0, 1.6, 2.2, 3.112, 4.4, 6.2]      # 3.112 is the v12 cell
SEEDS = [7701, 7702, 7703, 7704, 7705, 7706]

V12 = {"rate_grow": 0.004903568768146313, "rate_nematic_depoly": 0.0063040708394408885, "rate_nematic_poly": 0.0016555393492007537, "rate_branch": 0.0021422015778192914, "rate_nucleate": 0.05903938775253556, "nematic_thresh": 0.35, "angle_noise": 0.7131579655826681, "axis_spread": 0.35946663576674076, "cadherin_nucleation_prob": 0.2659229960288876, "rate_thin": 0.0013049314623728191, "n_sub": 4, "phi_max": None}

WORKER = r'''
import json, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

def _one(args):
    knobs, seed = args
    import sim3
    from modelling.CARMA.carma_6_particle import multicell_particle as mc
    t0 = time.time()
    out = sim3.run_once(knobs, seed=seed)
    rows = [dict(seed=seed, hpf=h, n_cells=mc.N_CELLS, crop=mc.CROP_SIZE,
                 canvas=mc.CANVAS_W, **r) for h, r in out.items()]
    print(f"    seed {seed:5d} {time.time()-t0:.0f}s  cells {mc.N_CELLS} crop {mc.CROP_SIZE}",
          flush=True)
    return rows

if __name__ == "__main__":
    knobs = json.loads(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]
    tag = sys.argv[3]
    with Pool(2) as pool:
        res = pool.map(_one, [(knobs, s) for s in seeds], chunksize=1)
    df = pd.DataFrame([r for rs in res for r in rs]); df["case"] = tag
    df.to_csv(f"ASPECT13_{tag}.csv", index=False)
'''


def set_cell(length, height):
    src = open(PARAMS).read()
    src = re.sub(r"^CELL_LENGTH = [0-9.]+", f"CELL_LENGTH = {length:.4f}", src, count=1, flags=re.M)
    src = re.sub(r"^CELL_HEIGHT = [0-9.]+", f"CELL_HEIGHT = {height:.4f}", src, count=1, flags=re.M)
    with open(PARAMS, "w") as fh:
        fh.write(src); fh.flush(); os.fsync(fh.fileno())
    shutil.rmtree("modelling/CARMA/carma_6_particle/__pycache__", ignore_errors=True)


if __name__ == "__main__":
    open("_worker13.py", "w").write(WORKER)
    try:
        for a in ASPECTS:
            L, H = np.sqrt(a * AREA), np.sqrt(AREA / a)
            tag = f"A{a:.3f}".replace(".", "p")
            print(f"=== aspect {a:.3f}  cell {L:.2f} x {H:.2f} um  (area {L*H:.1f}) ===", flush=True)
            set_cell(L, H)
            subprocess.run([sys.executable, "_worker13.py", json.dumps(V12),
                            ",".join(map(str, SEEDS)), tag], check=True)
        frames = [pd.read_csv("ASPECT13_" + f"A{a:.3f}".replace(".", "p") + ".csv")
                  for a in ASPECTS]
        pd.concat(frames, ignore_index=True).to_csv("ASPECT13_runs.csv", index=False)
        print("wrote ASPECT13_runs.csv")
    finally:
        set_cell(38.9, 12.5)      # always restore the v12 geometry
        print("parameters.py restored to 38.9 x 12.5")
