"""F_schematic.png -- what the model is. Two panels, drawn from REAL model objects at the
v13 tune rather than as a cartoon, so nothing in the picture is an idealisation.

A  ONE CELL at 44 hpf, the age where both fibre families coexist. The elongated-hexagon
   mask, the seeded striated array laid on template rows spaced along the cell's SHORT
   axis, the mesh grown on the cell's own axis with daughters at +-90 deg to their mother,
   the cortex, and the cortical sites where new bundles nucleate.
   NOTE the sites now run unbroken around the whole membrane. Until 26 Aug 2026 a site's
   firing direction was snapped to whichever lattice direction pointed at the cell
   CENTROID, which silenced a band one cell-height wide in the middle of each flat
   membrane -- an artefact of that construction, not a modelling choice. The rule is now
   "along the long axis, into the cell": 222 sites, not 159.
B  THE SHEET. Cells tiled confluently, each with its own mesh axis drawn from axis_spread
   and coupled to its neighbours at shared boundaries every 200 steps. The box is the
   61.4 um measurement window -- the same physical area as a traced crop, and the ONLY
   region any statistic is computed on.

Both panels are rasterised at lattice resolution and shown with imshow, so one drawn pixel
is one lattice point and nothing is smoothed or resized.
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi
import sim3
from bundle_model import multicell_particle as mc
from bundle_model.cell_particle import P
from thesisstyle import use_style, COL, FS, FW, panel_title, audit
use_style()

V12 = {'rate_grow': 0.004903568768146313, 'rate_nematic_depoly': 0.0063040708394408885, 'rate_nematic_poly': 0.0016555393492007537, 'rate_branch': 0.0021422015778192914, 'rate_nucleate': 0.05903938775253556, 'nematic_thresh': 0.35, 'angle_noise': 0.7131579655826681, 'axis_spread': 0.35946663576674076, 'cadherin_nucleation_prob': 0.2659229960288876, 'rate_thin': 0.0013049314623728191, 'n_sub': 4, 'phi_max': None}
sim3.configure(V12)
LU = 0.361
sheet = mc.MultiCell(base_seed=42)
for _ in range(int(round(int(P.steps) * (44.0 - 32.0) / P.total_hours))):
    sheet.step()

FAM = [("cortex", "#b0b0b0", "cortex"),
       ("nematic", COL["nematic"], "seeded striated array"),
       ("mesh", COL["quartic"], "mesh — the second family")]
INSIDE, EDGE = np.array([0.949, 0.961, 0.976]), np.array([0.78, 0.82, 0.87])

def outline(mask):
    return mask & ~ndi.binary_erosion(mask, np.ones((3, 3), bool))

fig, axes = plt.subplots(2, 1, figsize=(FW, 7.9),
                         gridspec_kw={"height_ratios": [0.82, 2.05]})

# ---------- A : one cell ----------
ax = axes[0]
img = np.ones((P.H, P.W, 3))
sim, _, _ = sheet.cells[len(sheet.cells) // 2]
img[sim.mask] = INSIDE
img[outline(sim.mask)] = EDGE
for fam, c, _ in FAM:
    col = np.array(to_rgb(c))
    for f in sim.fibres[fam]:
        if not f.occupied_pts:
            continue
        p = np.array(list(f.occupied_pts), int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < P.W) & (p[:, 1] >= 0) & (p[:, 1] < P.H)
        img[p[ok, 1], p[ok, 0]] = col
ax.imshow(img, extent=[0, P.W * LU, P.H * LU, 0], interpolation="nearest")
site = np.array(sim.nuc_sites, float)
ax.scatter(site[:, 0] * LU, site[:, 1] * LU, s=8, facecolors="none",
           edgecolors=COL["traced"], lw=.65, zorder=4)
for _, c, lab in FAM:
    ax.plot([], [], "s", color=c, ms=8, label=lab)
ax.plot([], [], "o", mfc="none", mec=COL["traced"], ms=7,
        label=f"cortical nucleation sites (n={len(sim.nuc_sites)})")
gap = np.mean([P.nematic_gap_min, P.nematic_gap_max]) * LU
x0, y0 = 6.2, 9.2
ax.annotate("", xy=(x0, y0), xytext=(x0, y0 + gap),
            arrowprops=dict(arrowstyle="<->", color=COL["nematic"], lw=1.8))
ax.text(x0 + 0.9, 2.2, f"template pitch {P.nematic_gap_min}–{P.nematic_gap_max} lu",
        fontsize=FS, color=COL["nematic"], va="bottom", ha="left")
ax.set_xlabel("µm"); ax.set_ylabel("µm")
ax.legend(loc="upper center", fontsize=FS, ncol=2, bbox_to_anchor=(0.5, -0.22),
          handlelength=1.2, columnspacing=1.2, labelspacing=0.25)
panel_title(ax, f"A  one cell at 44 hpf, {P.hex_half_w*2*LU:.1f} \u00d7 {P.hex_half_h*2*LU:.1f} µm", width=60)

# ---------- B : the sheet ----------
ax = axes[1]
canvas = np.ones((mc.CANVAS_H, mc.CANVAS_W, 3))
def paste(px, py, col):
    ok = (px >= 0) & (px < mc.CANVAS_W) & (py >= 0) & (py < mc.CANVAS_H)
    canvas[py[ok], px[ok]] = col
for sim_i, cx, cy in sheet.cells:
    yy, xx = np.nonzero(sim_i.mask)
    paste(xx - sheet.hcx + cx, yy - sheet.hcy + cy, INSIDE)
for sim_i, cx, cy in sheet.cells:
    yy, xx = np.nonzero(outline(sim_i.mask))
    paste(xx - sheet.hcx + cx, yy - sheet.hcy + cy, EDGE)
for sim_i, cx, cy in sheet.cells:
    for fam, c, _ in FAM:
        col = np.array(to_rgb(c))
        for f in sim_i.fibres[fam]:
            if not f.occupied_pts:
                continue
            p = np.array(list(f.occupied_pts), int)
            paste(p[:, 0] - sheet.hcx + cx, p[:, 1] - sheet.hcy + cy, col)
ax.imshow(canvas, extent=[0, mc.CANVAS_W * LU, mc.CANVAS_H * LU, 0], interpolation="nearest")
r0, r1, c0, c1 = sheet.crop_bounds()
ax.add_patch(Rectangle((c0 * LU, r0 * LU), (c1 - c0) * LU, (r1 - r0) * LU,
                       fill=False, ec=COL["model"], lw=2.8, zorder=5))
ax.annotate(f"the {(c1-c0)*LU:.1f} µm measurement window", (0.5, -0.10),
            xycoords="axes fraction", fontsize=FS, color=COL["model"],
            ha="center", va="top")
ax.set_xlabel("µm"); ax.set_ylabel("µm")
panel_title(ax, f"B  the confluent sheet, {len(sheet.cells)} cells", width=60)

fig.tight_layout(h_pad=2.4); fig.savefig("F_schematic_v13.png")
audit(fig, "F_schematic_v13")
print("wrote F_schematic_v13.png:", len(sheet.cells), "cells, crop", c1 - c0, "lu, sites", len(sim.nuc_sites))
