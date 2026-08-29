"""F_constrain — what the data constrain. TWO panels, authored at 6.54 in, 9 pt.

CUT FROM THE FOUR-PANEL VERSION, and why. Both reasons are about redundancy, not type size, so
they still hold now that the floor is 9 pt rather than 11:
  * the RESOLUTION panel (detectable dL_dgo against seed count) was a plot of 2.8*sd/sqrt(n).
    That is an arithmetic identity, not a measurement — the only number in it is the paired SD,
    0.472, and the only consequence is the 0.540 floor, which now appears as a labelled line on
    panel A where it is actually used.
  * the SENSITIVITY BAR panel was fully contained in panel A: the bar length IS panel A's
    x-coordinate, and panel A adds the second axis that decides identification. Showing the same
    ten numbers twice, once ranked and once positioned, is a repeat.

Point labels are placed OUT of the crowded region and joined to their points by leaders, so no
label sits on top of another point or of the resolution line. [stated] Luka asked for the leaders.
"""
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import loss_lab as LL
from thesisstyle import use_style, COL, FS, FW, panel_title, audit
use_style()

SHORT = {"thin_grow": "thin:grow", "angle_noise": "angle noise",
         "rate_nematic_depoly": "nematic depoly", "nematic_thresh": "nematic thresh*",
         "rate_branch": "branch", "rate_nucleate": "nucleate", "rate_grow": "grow",
         "rate_nematic_poly": "nematic poly", "axis_spread": "axis spread",
         "cadherin_nucleation_prob": "cadherin"}

# label anchor (data coords) and horizontal alignment; a leader joins it to the point
PLACE = {
    "cadherin_nucleation_prob": ((0.30, 1.66), "left"),
    "rate_branch":              ((0.72, 1.56), "left"),
    "rate_nematic_poly":        ((0.72, 1.36), "left"),
    "rate_grow":                ((0.72, 1.16), "left"),
    "nematic_thresh":           ((0.62, 0.94), "left"),
    "rate_nematic_depoly":      ((1.02, 0.72), "left"),
    "axis_spread":              ((0.02, 0.50), "left"),
    "rate_nucleate":            ((0.30, 0.10), "left"),
    "angle_noise":              ((1.08, 0.15), "left"),
    "thin_grow":                ((1.62, 0.52), "right"),
}

SD = json.load(open("NOISE13.json"))["sd_pair"]
RES6 = 2.8 * SD / np.sqrt(6)
AGREE = np.log2(1.4 / 0.6) / 2.0

s = pd.read_csv("SENSV13_scored.csv").set_index("param")
DIS = pd.read_csv("V13_tied_spread.csv").set_index("param")["log2"].to_dict()

fig, axes = plt.subplots(2, 1, figsize=(FW, 6.4), gridspec_kw={"height_ratios": [1.55, 1.0]})

# ------------------------------------------------ A : sensitivity vs disagreement
ax = axes[0]
ax.axvspan(0, RES6, color="#f2f2f2", zorder=0)
ax.axhspan(0, AGREE, color=COL["band"], alpha=.28, zorder=0)
for p in s.index:
    x, y = float(s.maxdev[p]), DIS[p]
    c = COL["grey"] if x <= RES6 else (COL["quartic"] if y <= AGREE else COL["model"])
    ax.scatter([x], [y], s=34, color=c, zorder=4, edgecolors="white", linewidths=.6)
    (lx, ly), ha = PLACE[p]
    ax.annotate(SHORT[p], xy=(x, y), xytext=(lx, ly), fontsize=FS, color=c,
                va="center", ha=ha, zorder=5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=c, alpha=.75,
                                shrinkA=2, shrinkB=3))
ax.axvline(RES6, color=COL["traced"], ls="--", lw=1.1, zorder=1)
ax.axhline(AGREE, color=COL["quartic"], ls="--", lw=1.1, zorder=1)
ax.set_xlim(-0.05, 2.15); ax.set_ylim(-0.10, 1.86)
ax.set_xlabel("sensitivity: largest change in loss at \u00b140 %")
ax.set_ylabel("multiple-tune disagreement\n(absolute log2 ratio)")
ax.text(RES6 + 0.05, 0.40, "6-seed resolution 0.54", fontsize=FS, color=COL["traced"],
        ha="left", va="center")
ax.text(2.12, AGREE + 0.05, "agreement line 1.53×", fontsize=FS, color=COL["quartic"],
        ha="right")
ax.text(2.12, 0.02, "FITTED", fontsize=FS, color=COL["quartic"], weight="bold", ha="right")
ax.text(2.12, 1.82, "* sensitive, not identified", fontsize=FS, color=COL["model"],
        ha="right", va="top")
ax.text(0.00, 1.82, "invisible to the fit", fontsize=FS, color=COL["grey"], ha="left", va="top")
panel_title(ax, "A   parameter sensitivity against disagreement among multiple tunes", width=80)

# ------------------------------------------------ B : multi-start basin minima
ax = axes[1]
T = LL.Targets("traced_per_image.csv")
d = pd.read_csv("MS13_final_runs.csv")
vals = {c: LL.evaluate_all({k: g.groupby("hpf")[k].mean().reindex(LL.HPF).to_numpy()
                            for k in LL.METRICS}, T)["L_dgo"] for c, g in d.groupby("case")}
v = pd.Series(vals).sort_values()
y = np.arange(len(v))
ax.barh(y, v.values, color=COL["gate"], height=.66)
ax.axvline(3.632, color=COL["quartic"], lw=1.8)
ax.text(3.52, 2.5, "v13  3.63", fontsize=FS, color=COL["quartic"], ha="right", va="center",
        weight="bold", bbox=dict(fc="white", ec="none", pad=1.2))
ax.set_yticks(y); ax.set_yticklabels([k.replace("M_S", "start ") for k in v.index])
ax.set_xlim(0, 7.6); ax.set_ylim(-0.7, len(v) - 0.3)
ax.set_xlabel("loss at 12 shared seeds")
panel_title(ax, "B   basin minimum of each of six independent starts", width=80)

fig.tight_layout(h_pad=1.4)
fig.savefig("F_constrain_v13.png")
audit(fig, "F_constrain_v13")
