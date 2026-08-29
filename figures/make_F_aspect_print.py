"""F_aspect — cell SHAPE at constant cell AREA, on the corrected nucleation rule. TWO panels.

CUT FROM THE THREE-PANEL VERSION:
  * the old panel A (32 hpf density against aspect) was a single column of the old panel B
    (density against age, one curve per aspect). One of the two had to go. Panel A here keeps
    BOTH ends — 32 and 52 hpf — against aspect, which is the whole finding now that the late
    field is known to respond too, and which the old panel A could not show at all.
  * the old panel B also redrew the traced density curve, which already appears in F1. A repeat
    across figures is still a repeat.

WHY THE SWEEP WAS RE-RUN. Under the old _seed_angle the nucleation dead band's width was the CELL
HEIGHT, so the nucleation-competent FRACTION of the cortex was itself a function of aspect ratio
(0.433 at aspect 1.0 to 0.861 at 6.2). That sweep varied shape and nucleation count together.
Under the corrected rule every cortical site nucleates at every aspect, leaving only the perimeter
term a real cell also has.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
import loss_lab as L
from thesisstyle import use_style, COL, FS, FW, panel_title, audit
use_style()

T = L.Targets()
d = pd.read_csv("ASPECT13_runs.csv")
ASP = {"A1p000": 1.0, "A1p600": 1.6, "A2p200": 2.2, "A3p112": 3.112, "A4p400": 4.4, "A6p200": 6.2}

rows = []
for c, a in ASP.items():
    g = d[d.case == c]
    arr = {k: g.groupby("hpf")[k].mean().reindex(L.HPF).to_numpy() for k in L.METRICS}
    sd = g.groupby("hpf")["density"].std().reindex(L.HPF).to_numpy()
    e = L.evaluate_all(arr, T)
    rows.append(dict(a=a, L=e["L_dgo"], den=e["Xs_density"], gap=e["Xs_gap_p95"],
                     order=e["Xs_order"], d32=arr["density"][0], s32=sd[0],
                     d52=arr["density"][5], s52=sd[5]))
r = pd.DataFrame(rows)
rho32 = stats.spearmanr(r.a, r.d32)[0]
rho52, p52 = stats.spearmanr(r.a, r.d52)
rhoL = stats.spearmanr(r.a, r.L)[0]

fig, axes = plt.subplots(2, 1, figsize=(FW, 6.2))

# ---------------- A ----------------
ax = axes[0]
n = np.sqrt(6)
ax.errorbar(r.a, r.d32, yerr=r.s32 / n, fmt="o-", color=COL["nematic"], ms=5, capsize=2.5,
            label="model 32 hpf")
ax.errorbar(r.a, r.d52, yerr=r.s52 / n, fmt="s-", color=COL["quartic"], ms=5, capsize=2.5,
            label="model 52 hpf")
for j, col, lab in ((0, COL["nematic"], "traced 32"), (5, COL["quartic"], "traced 52")):
    ax.fill_between([0.8, 6.6], T.mean["density"][j] - T.sd["density"][j],
                    T.mean["density"][j] + T.sd["density"][j], color=col, alpha=.13, lw=0)
    ax.axhline(T.mean["density"][j], color=col, lw=1.1, ls=":")
ax.axvline(3.112, color=COL["grey"], ls="--", lw=1.1)
ax.text(3.24, 0.0245, "measured cell", fontsize=FS, color=COL["grey"], va="bottom")
ax.text(6.5, T.mean["density"][0] + 0.0035, "traced 32 hpf", fontsize=FS,
        color=COL["nematic"], ha="right")
ax.text(6.5, T.mean["density"][5] + 0.005, "traced 52 hpf", fontsize=FS, color=COL["quartic"],
        ha="right")
ax.set_xlim(0.8, 6.6); ax.set_ylim(0.022, 0.104)
ax.set_xlabel("aspect ratio (length / height)")
ax.set_ylabel("line density")
ax.legend(loc="lower left", ncol=1, fontsize=FS, handlelength=1.4, borderpad=0.2,
          labelspacing=0.25)
panel_title(ax, "A   line density at 32 and 52 hpf against aspect ratio", width=80)

# ---------------- B ----------------
ax = axes[1]
ax.plot(r.a, r.L, "o-", color=COL["model"], ms=5, label="total loss")
ax.plot(r.a, 2 * r.den, "s--", color=COL["quartic"], ms=4, label="2 \u00d7 density")
ax.plot(r.a, 1.5 * r.gap, "^:", color=COL["nematic"], ms=4, label="1.5 \u00d7 gap")
ax.plot(r.a, r.order, "v-.", color=COL["grey"], ms=4, label="1 \u00d7 order")
ax.axvline(3.112, color=COL["grey"], ls="--", lw=1.1)
ax.set_xlim(0.8, 6.6)
ax.set_xlabel("aspect ratio (length / height)")
ax.set_ylabel("loss contribution")
ax.legend(loc="upper right", ncol=2, fontsize=FS, handlelength=1.5, columnspacing=1.0,
          labelspacing=0.3, borderpad=0.3, borderaxespad=0.6)
ax.set_ylim(0, 8.6)
panel_title(ax, "B   the loss and its three weighted terms against aspect ratio",
            width=80)

fig.tight_layout(h_pad=2.4)
fig.savefig("F_aspect_v13.png")
audit(fig, "F_aspect_v13")
print(f"  rho32 {rho32:+.3f}  rho52 {rho52:+.3f} (p {p52:.4f})  rhoL {rhoL:+.3f}")
