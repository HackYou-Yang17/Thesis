"""crossing_check, re-authored to the thesisstyle print rule.

WHY IT WAS REDONE. The old figure was authored 13.5 x 4.1 in. Placed at \\textwidth = 6.54 in
that is a 0.48x reduction, so its 10 pt panel titles printed at 4.8 pt and its 8 pt annotations
at 3.9 pt — exactly the failure thesisstyle.py was written to eliminate, and this figure was
never brought over with the others.

WHAT CHANGED
  * authored at exactly FW = 6.54 in, so authored pt == printed pt, and every text object
    is >= MIN_PT (9 pt). Verified by thesisstyle.audit().
  * three panels stacked instead of side by side. thesisstyle: "Figures may be TALLER than FW;
    height costs a reader nothing and width costs legibility."
  * the two traced annotations in panel a were HARD-CODED strings ("traced 52 hpf = 0.43",
    "traced 32 hpf = 0.00") while the lines they label were drawn from the data. They now read
    from the data, so the figure cannot silently disagree with itself.
  * panel b's twin axis kept: it is the evidence that the fraction is stable where the median
    is not, which is the claim the appendix makes.
  * colours taken from thesisstyle.COL so the figure matches the rest of the set.

NOT CHANGED: every number. full.py reproduces the published values exactly
(52 hpf frac>60 = 0.43, per-heart 0.42/0.48/0.40; median 32.2).
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/uploads/thesis--thesis/pipeline")
import thesisstyle as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts.use_style()

d = json.load(open("res.json"))
cal, real = d["cal"], d["real"]
seps = sorted(int(k) for k in cal)
hpfs = sorted(int(k) for k in real)
cf = [cal[str(s)]["f60"] for s in seps]
rf = [real[str(h)]["f60"] for h in hpfs]
rm = [real[str(h)]["med"] for h in hpfs]
rper = [[p["f60"] for p in real[str(h)]["per"]] for h in hpfs]
rperm = [[p["med"] for p in real[str(h)]["per"]] for h in hpfs]

K = ts.COL["grey"]
G = ts.COL["quartic"]     # 52 hpf / mesh
O = ts.COL["nematic"]     # 32 hpf / striated

fig, ax = plt.subplots(3, 1, figsize=(ts.FW, 8.4))

# ---------------------------------------------------------------- a. calibration ladder
a = ax[0]
a.plot(seps, cf, "o-", color=K, ms=5, label="synthetic fields, known separation")
for s, f in zip(seps, cf):
    a.annotate(f"{s}°", (s, f), textcoords="offset points",
               xytext=(6, 6) if s <= 30 else (6, -12), fontsize=ts.FS, color=K)
lo, hi = sorted([rf[hpfs.index(44)], rf[hpfs.index(52)]])   # the late-window band, as before
a.axhspan(lo, hi, color=G, alpha=0.10, lw=0)
a.axhline(rf[hpfs.index(52)], color=G, lw=1.6)
a.annotate(f"traced 52 hpf = {rf[hpfs.index(52)]:.2f}", (24, rf[hpfs.index(52)]),
           textcoords="offset points", xytext=(0, 7), fontsize=ts.FS, color=G, ha="center")
a.axhline(rf[hpfs.index(32)], color=O, lw=1.6)
a.annotate(f"traced 32 hpf = {rf[hpfs.index(32)]:.2f}", (70, rf[hpfs.index(32)]),
           textcoords="offset points", xytext=(0, 7), fontsize=ts.FS, color=O, ha="center")
a.set_xlabel("true separation between families (°)")
a.set_ylabel("fraction of crossings > 60°")
ts.panel_title(a, "a  Calibration: the 52 hpf tissue interpolates between the 60° and 75° references", width=64)
a.set_ylim(-0.04, 0.62)
a.legend(loc="upper left")

# ---------------------------------------------------------------- b. stability across stages
a = ax[1]
a.plot(hpfs, rf, "o-", color=G, ms=5.5, label="fraction > 60°  (stable)")
for h, ps in zip(hpfs, rper):
    a.plot([h] * len(ps), ps, "o", color=G, ms=3.5, alpha=0.45)
a2 = a.twinx()
a2.spines["right"].set_visible(True)
a2.plot(hpfs, rm, "s--", color=O, lw=1.4, ms=4.5, label="median (°)  (unstable)")
for h, ms_ in zip(hpfs, rperm):
    a2.plot([h] * len(ms_), ms_, "s", color=O, ms=3.5, alpha=0.45)
ts.age_axis(a)
a.set_ylabel("fraction > 60°", color=G)
a2.set_ylabel("median crossing angle (°)", color=O)
a2.tick_params(axis="y", labelsize=ts.FS)
ts.panel_title(a, "b  The fraction is the stable statistic; the median is not", width=64)
h1, l1 = a.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="upper left")

# ---------------------------------------------------------------- c. why the median fails
a = ax[2]
bins = np.linspace(0, 90, 19)
a.hist(cal["90"]["ang"], bins=bins, density=True, color=K, alpha=0.30,
       label="synthetic, true 90°")
a.hist(real["52"]["ang"], bins=bins, density=True, histtype="step", color=G, lw=1.8,
       label="traced 52 hpf")
a.hist(real["32"]["ang"], bins=bins, density=True, histtype="step", color=O, lw=1.8,
       label="traced 32 hpf")
a.axvline(60, color="k", ls=":", lw=1)
a.annotate("60° threshold", (60, a.get_ylim()[1] * 0.55), textcoords="offset points",
           xytext=(-6, 0), fontsize=ts.FS, color="k", ha="right")
a.set_xlabel("crossing angle (°)")
a.set_ylabel("density")
ts.panel_title(a, "c  Bimodal: skeletonisation spurs fill the low mode, which is what drags the median down", width=64)
a.legend(loc="upper center")

fig.tight_layout(h_pad=1.6)
fig.savefig("crossing_check.png")
fig.savefig("crossing_check.pdf")
ts.audit(fig, "crossing_check")

# overlap check, same idea as check_box.py
import itertools, matplotlib
fig.canvas.draw(); r = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
bb = [(t, t.get_window_extent(r)) for t in texts]
ov = [(a.get_text()[:26], b.get_text()[:26]) for (a, x), (b, y) in itertools.combinations(bb, 2)
      if x.overlaps(y) and a.axes is b.axes]
print("[overlap]", "OK" if not ov else ov)
print("52 hpf frac>60 = %.2f  (per-heart %s)" % (rf[-1], ", ".join("%.2f" % p for p in rper[-1])))
print("32 hpf frac>60 = %.2f   median at 52 hpf = %.1f deg" % (rf[0], rm[-1]))
