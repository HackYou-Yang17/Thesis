"""Paired per-seed comparison of the three arms in seedangle_run.py."""
from __future__ import annotations

import numpy as np, pandas as pd
from scipy import stats
import loss_lab as LL

T = LL.Targets("traced_per_image.csv")


def per_seed(df):
    out = {}
    for (case, seed), g in df.groupby(["case", "seed"]):
        a = {k: g.sort_values("hpf")[k].to_numpy(float) for k in LL.METRICS}
        out[(case, seed)] = LL.evaluate_all(a, T)
    return out


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("SA_runs.csv"), pd.read_csv("SBC_runs.csv")], ignore_index=True)
    ps = per_seed(df)
    rows = [dict(case=c, seed=s, L_dgo=r["L_dgo"], X_density=r["X_density"],
                 X_gap=r["X_gap_p95"], X_order=r["X_order"], X_foam8=r.get("X_foam_ge8", np.nan),
                 xc=r["xc"], gate=r["gate"]) for (c, s), r in ps.items()]
    d = pd.DataFrame(rows).sort_values(["case", "seed"])
    d.to_csv("SEEDANGLE_perseed.csv", index=False)

    print(d.groupby("case").agg(n=("seed", "size"), L_dgo=("L_dgo", "mean"),
                                sd=("L_dgo", "std"), X_density=("X_density", "mean"),
                                X_gap=("X_gap", "mean"), X_order=("X_order", "mean"),
                                xc=("xc", "mean")).to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    piv = d.pivot(index="seed", columns="case", values="L_dgo")
    base = "A_centroid_159"
    print("PAIRED against the centroid rule, same seeds  (six-seed resolution = 0.46)")
    for c in piv.columns:
        if c == base:
            continue
        dd = piv[c] - piv[base]
        t, p = stats.ttest_rel(piv[c], piv[base])
        verdict = "NO DETECTABLE CHANGE" if abs(dd.mean()) < 0.46 else "*** MOVED ***"
        print(f"  {c:26s} delta {dd.mean():+.3f} +- {dd.std(ddof=1)/np.sqrt(len(dd)):.3f}"
              f"   p = {p:.3f}   {verdict}")
    print()
    print(piv.round(3).to_string())
