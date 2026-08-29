"""Crossing angles at the v13 tune, measured through the SAME skeleton route as the traces.

This is a CONSTRUCTION CHECK, not a prediction: branching imposes daughters at +-90 deg to
their mother, so orthogonal crossings are built in. What it checks is that the measurement
pipeline recovers what the model was told to do -- and it must be measured from the SKELETON,
not from the exact fibre objects, or it is not the same measurement as the traced one.
"""
import glob, re
import numpy as np, pandas as pd
from _bias_check import real_crossings

rows = []
for f in sorted(glob.glob("v13fields_s*.npz")):
    seed = int(re.search(r"s(\d+)", f).group(1))
    z = np.load(f)
    for k in z.files:
        hpf = int(k[1:])
        a = real_crossings(z[k].astype(bool))
        rows.append(dict(seed=seed, hpf=hpf, n=a.size,
                         frac60=float((a > 60).mean()) if a.size else np.nan))
pd.DataFrame(rows).sort_values(["seed", "hpf"]).to_csv("v13_crossings.csv", index=False)
print("wrote v13_crossings.csv", len(rows), "rows")
