"""crossing_check, JUNCTION-TANGENT definition, authored to the thesisstyle print rule.

Branch direction is the LOCAL TANGENT at the junction (spacing.tangent_angles, arc-local PCA
over +-6 px), not the whole-branch principal axis that bias_check.real_crossings uses. The
model's +-90 deg rule is defined at the branch point, so this is the like-for-like quantity.
Data from full_tan.py -> res_tan.json.

WHY IT WAS REDONE. The old figure was authored 13.5 x 4.1 in. Placed at \\textwidth = 6.54 in
that is a 0.48x reduction, so its 10 pt panel titles printed at 4.8 pt and its 8 pt annotations
at 3.9 pt — exactly the failure thesisstyle.py was written to eliminate, and this figure was
never brought over with the others.

WHAT CHANGED
  * authored at exactly FW = 6.54 in, so authored pt == printed pt, and every text object
    is >= MIN_PT (9 pt). Verified by thesisstyle.audit().
  * two panels stacked instead of three side by side. thesisstyle: "Figures may be TALLER than FW;
    height costs a reader nothing and width costs legibility."
  * the two traced annotations in panel a were HARD-CODED strings ("traced 52 hpf = 0.43",
    "traced 32 hpf = 0.00") while the lines they label were drawn from the data. They now read
    from the data, so the figure cannot silently disagree with itself.
  * the old panel b (fraction and median against hpf, twin axis) is DROPPED at Luka's request.
    What goes with it: the per-heart scatter and the 48 hpf dip. The remaining two panels carry
    the appendix's actual claim -- where the tissue sits on the ladder, and why the median cannot
    be used to put it there.
  * colours taken from thesisstyle.COL so the figure matches the rest of the set.

NOT CHANGED: every number. full.py reproduces the published values exactly
(52 hpf frac>60 = 0.43, per-heart 0.42/0.48/0.40; median 32.2).
"""
import json, sys
import numpy as np
import thesisstyle as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts.use_style()

d = json.load(open("res_tan.json"))
cal, real = d["cal"], d["real"]
seps = sorted(int(k) for k in cal)
hpfs = sorted(int(k) for k in real)
cf = [cal[str(s)]["f60"] for s in seps]
rf = [real[str(h)]["f60"] for h in hpfs]
rm = [real[str(h)]["med"] for h in hpfs]
rper = [[p["f60"] for p in real[str(h)]["per"]] for h in hpfs]

K = ts.COL["grey"]
G = ts.COL["quartic"]     # 52 hpf / mesh
O = ts.COL["nematic"]     # 32 hpf / striated

fig, ax = plt.subplots(2, 1, figsize=(ts.FW, 7.4))

# ---------------------------------------------------------------- a. calibration ladder
a = ax[0]
a.plot(seps, cf, "o-", color=K, ms=5, label="synthetic fields, known separation")
for s, f in zip(seps, cf):
    a.annotate(f"{s}°", (s, f), textcoords="offset points",
               xytext=(2, 9) if s < 45 else (7, -13), fontsize=ts.FS, color=K)
lo, hi = sorted([rf[hpfs.index(44)], rf[hpfs.index(52)]])   # the late-window band, as before
a.axhspan(lo, hi, color=G, alpha=0.10, lw=0)
a.axhline(rf[hpfs.index(52)], color=G, lw=1.6)
a.annotate(f"traced 52 hpf = {rf[hpfs.index(52)]:.2f}", (18, rf[hpfs.index(52)]),
           textcoords="offset points", xytext=(0, 7), fontsize=ts.FS, color=G, ha="center")
a.axhline(rf[hpfs.index(32)], color=O, lw=1.6)
a.annotate(f"traced 32 hpf = {rf[hpfs.index(32)]:.2f}", (22, rf[hpfs.index(32)]),
           textcoords="offset points", xytext=(0, -16), fontsize=ts.FS, color=O, ha="center")
a.axhline(cf[0], color=K, lw=1.0, ls=(0, (4, 3)))
a.annotate("single-family floor = %.2f" % cf[0], (46, cf[0]), textcoords="offset points",
           xytext=(0, 8), fontsize=ts.FS, color=K, ha="center")
a.axvspan(60, 92, color=K, alpha=0.07, lw=0)
a.annotate("90° cap", (76, 0.185), fontsize=ts.FS, color=K, ha="center", va="center")
a.set_xlabel("true separation between families (°)")
a.set_ylabel("fraction of crossings > 60°")
ts.panel_title(a, "a  Calibration: the 52 hpf tissue reads at the top of the ladder", width=95, pad=12)
a.set_ylim(-0.04, 0.78)
a.legend(loc="upper left")

# ---------------------------------------------------------------- c. why the median fails
a = ax[1]
bins = np.linspace(0, 90, 19)
a.hist(cal["90"]["ang"], bins=bins, density=True, color=K, alpha=0.30,
       label="synthetic, true 90°")
a.hist(real["52"]["ang"], bins=bins, density=True, histtype="step", color=G, lw=1.8,
       label="traced 52 hpf")
a.hist(real["32"]["ang"], bins=bins, density=True, histtype="step", color=O, lw=1.8,
       label="traced 32 hpf")
a.set_ylim(0, a.get_ylim()[1] * 1.58)          # headroom so the legend never sits on a bar
a.axvline(60, color="k", ls=":", lw=1)
ymax = a.get_ylim()[1]
a.annotate("60° threshold", (60, ymax * 0.82), textcoords="offset points",
           xytext=(-6, 0), fontsize=ts.FS, color="k", ha="right", va="center")
a.set_xlabel("crossing angle (°)")
a.set_ylabel("density")
ts.panel_title(a, "b  At 52 hpf the traced and true-90° distributions nearly coincide (median 67.5° in both)", width=95, pad=12)
a.legend(loc="upper center", ncol=3, columnspacing=1.1, handlelength=1.4)

fig.tight_layout(h_pad=3.2)
fig.savefig("crossing_check_tan.png")
fig.savefig("crossing_check_tan.pdf")
ts.audit(fig, "crossing_check_tan")

# ---------------------------------------------------------------- collision checks
import itertools, matplotlib
import numpy as _np
fig.canvas.draw(); r = fig.canvas.get_renderer()


def _axes_of(t):
    if t.axes is not None:
        return t.axes
    fp = getattr(t, "figure", None)
    for axx in fig.axes:                      # legend texts carry no .axes
        lg = axx.get_legend()
        if lg is not None and t in lg.get_texts():
            return axx
    return None


texts = [t for t in fig.findobj(matplotlib.text.Text)
         if t.get_text().strip() and t.get_visible()]
bb = [(t, t.get_window_extent(r), _axes_of(t)) for t in texts]

# 1. text on text
ov = [(a.get_text()[:26], b.get_text()[:26])
      for (a, x, ax1), (b, y, ax2) in itertools.combinations(bb, 2)
      if x.overlaps(y) and ax1 is ax2 and ax1 is not None]
print("[text-text]", "OK" if not ov else ov)

# 2. text on a DRAWN STROKE. A line's bbox spans the whole curve, so bbox-vs-bbox is
#    meaningless here: sample points along each segment and ask whether any of them
#    actually lands inside the label. Background shading (the axhspan and the filled
#    histogram) is excluded -- text over a pale wash is legible and intended.
def _stroke_points(line):
    xy = line.get_xydata()
    if len(xy) < 2:
        return _np.empty((0, 2))
    pts = line.axes.transData.transform(xy)
    out = [pts]
    for i in range(len(pts) - 1):
        out.append(pts[i] + (pts[i + 1] - pts[i]) * _np.linspace(0, 1, 40)[:, None])
    return _np.vstack(out)


hits = []
for t, tb, ax_ in bb:
    if ax_ is None:
        continue
    pad = tb.expanded(1.02, 1.02)
    for ln in ax_.lines:
        if not ln.get_visible() or (ln.get_alpha() or 1) < 0.6:
            continue
        p = _stroke_points(ln)
        if len(p) and _np.any((p[:, 0] >= pad.x0) & (p[:, 0] <= pad.x1) &
                              (p[:, 1] >= pad.y0) & (p[:, 1] <= pad.y1)):
            hits.append((t.get_text()[:30], "line"))
    for pt in ax_.patches:
        if (pt.get_alpha() or 1) < 0.6:        # the pale wash and the grey fill
            continue
        pbb = pt.get_window_extent(r)
        if pbb.width and pbb.height and pad.overlaps(pbb):
            hits.append((t.get_text()[:30], "bar"))
print("[text-data]", "OK" if not hits else sorted(set(hits)))

print("52 hpf frac>60 = %.2f  (per-heart %s)" % (rf[-1], ", ".join("%.2f" % p for p in rper[-1])))
print("32 hpf frac>60 = %.2f   median at 52 hpf = %.1f deg" % (rf[0], rm[-1]))
