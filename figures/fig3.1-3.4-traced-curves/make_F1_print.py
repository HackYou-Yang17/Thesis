"""F1 — the traced curves with the v13 model on top, and what was fitted. SIX panels, 6.54 in.

WHAT CHANGED FROM THE PREVIOUS SIX-PANEL VERSION. Same panel count, different panels, because two
of the old six were not earning their place and one thing that was fitted was never plotted.

  DROPPED  raw foam index. It had to carry a four-line warning on its own face saying the
           agreement was not what it looked like: 62 % of the model's enclosed regions are under
           8 px against 19 % of the traces, so the two curves agreed for the wrong reason. The
           size-filtered version is the honest measurement and it is now a panel in its own right,
           so the raw index and its disclaimer both go.
  DROPPED  crossings above 60 deg. Its own annotation called it a construction check: the +-90 deg
           branch rule imposes orthogonal crossings, so agreement shows the pipeline measures what
           it was told, not that the model found anything. That belongs in the methods, not in the
           results figure.
  ADDED    the BAND STATISTIC. X_order is one of the three terms of the objective and its curve
           had never been plotted — the residual bar was the only trace of it. A fitted quantity
           should be visible as a fit.
  ADDED    mesh closure (>= 8 px) as a curve, not just a bar. It is the largest unfitted residual
           and the deficit WIDENS with age, which a single bar cannot show.
  SWAPPED  segment COUNT -> mean segment LENGTH (01 Sep 2026). Count is very nearly a restatement
           of line density (Spearman rho = +0.96 across the 18 traced fields, partial +0.96
           controlling for timepoint), and density is a fitted target, so count cannot serve as an
           unfitted prediction. Mean segment length measures the run between junctions rather than
           the amount of line (rho = -0.52 with density) and carries a systematic signal count
           hides: the model runs SHORTER than the tissue at every timepoint.
  DROPPED  fragmentation from panel F. It tracks mesh closure at rho = +0.91 on the traced set, so
           reporting both counted one failure twice.

Model is 30 seeds, mean with its own +-1 SD; traced is 3 hearts per age with +-1 SD.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import loss_lab as L
from thesisstyle import use_style, COL, FS, FW, age_axis, panel_title, audit
use_style()

T = L.Targets("traced_per_image_fib.csv")
g = pd.read_csv("V13FIB_runs.csv")
NS = g.seed.nunique()
m = {k: g.groupby("hpf")[k].mean().reindex(L.HPF).to_numpy() for k in L.METRICS}
sd = {k: g.groupby("hpf")[k].std().reindex(L.HPF).to_numpy() for k in L.METRICS}

fig, axes = plt.subplots(3, 2, figsize=(FW, 7.6))


def pair(ax, key, ylab, title, legend=False):
    tm, ts = T.mean[key], T.sd[key]
    ax.fill_between(L.HPF, tm - ts, tm + ts, color=COL["band"], alpha=.45, lw=0)
    ax.plot(L.HPF, tm, "-o", color=COL["traced"], ms=4, lw=1.7,
            label="traced (n=3)" if legend else None)
    ax.fill_between(L.HPF, m[key] - sd[key], m[key] + sd[key], color=COL["model"],
                    alpha=.18, lw=0)
    ax.plot(L.HPF, m[key], "-s", color=COL["model"], ms=4, lw=1.7,
            label=f"model (n={NS})" if legend else None)
    age_axis(ax, label=False)
    ax.set_ylabel(ylab)
    panel_title(ax, title, width=34)
    if legend:
        ax.legend(loc="lower right", fontsize=FS, handlelength=1.3, labelspacing=0.2,
                  borderpad=0.2)


pair(axes[0, 0], "density", "centreline px per field px",
     "A  line density against age", legend=True)
pair(axes[0, 1], "gap_p95", "gap (µm)", "B  95th-percentile inter-fibre gap")
pair(axes[1, 0], "order", "order parameter", "C  orientation order (\u00b145\u00b0 band)")
pair(axes[1, 1], "foam_ge8", "regions per unit area",
     "D  mesh closure (regions \u2265 8 px)")
pair(axes[2, 0], "fib_len_um", "length (µm)", "E  mean fibre length")
age_axis(axes[2, 0], label=True)
axes[1, 1].set_ylim(bottom=0)
axes[1, 1].set_xlabel("hours post fertilisation")

# ---------------- F : what was fitted ----------------
ax = axes[2, 1]
keys = ["density", "gap_p95", "order", "foam_ge8", "fib_len_um"]
LAB = {"density": "line density", "gap_p95": "gap p95", "order": "orientation",
       "foam_ge8": "mesh closure", "fib_len_um": "fibre length"}
X = {k: L.X(m[k], T.mean[k], T.sd_shrunk[k]) for k in keys}
y = np.arange(len(keys))
ax.barh(y, [X[k] for k in keys], height=.62,
        color=[COL["traced"]] * 3 + [COL["quartic"]] * 2)
for i, k in enumerate(keys):
    ax.annotate(f"{X[k]:.2f}", (X[k] + 0.06, i), va="center", fontsize=FS)
ax.axhline(2.5, color=COL["grey"], lw=1.0)
ax.set_yticks(y); ax.set_yticklabels([LAB[k] for k in keys])
ax.invert_yaxis(); ax.set_xlim(0, 4.0)
ax.set_xlabel("residual (heart-SD)")
ax.text(3.95, 0.05, "FITTED", fontsize=FS, color=COL["traced"], weight="bold", ha="right")
ax.text(3.95, 3.05, "NOT FITTED", fontsize=FS, color=COL["quartic"], weight="bold",
        ha="right")
panel_title(ax, "F  residual per statistic", width=34)
axes[2, 0].set_xlabel("hours post fertilisation")
ax.set_xlabel("residual (heart-SD units)")

fig.tight_layout(h_pad=1.2, w_pad=1.6)
fig.savefig("traced_curves_v13.png")
audit(fig, "traced_curves_v13")
print({k: round(v, 3) for k, v in X.items()})
