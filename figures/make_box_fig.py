"""
make_box_fig.py -- fibre spacing at the two endpoints, new yellow-green traces.

LANDSCAPE LAYOUT. Plot on the left; two arrangement panels in a column on the far right,
nematic on top and mesh below. No speech tails: the association is carried by COLOUR --
each panel's border, tint and labels are the same colour as its box in the plot, and the
heading names the stage as well as the arrangement.

Panels are near-square (1.36 x 1.09 in, 1.25:1), the proportion the schematics were drawn
for: four fibre lines in a wide, flat box read as a barcode rather than as tissue. The plot
takes the width that frees up, which also puts the median and IQR back over their own boxes.

Sized for Word: 6.5 in wide, i.e. a full text column at 100 %, saved WITHOUT a tight
bounding box so the placed width is exactly 6.5 in and every text element renders at its
true point size. Minimum used here is 9 pt; every other size is scaled from it by the
same 9/11 factor (title 10.5, axis label 9.5, stage ticks 10).
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NEM, MESH = '#C4762A', '#388561'
INK, MUTE = '#2E2E2E', '#8C8C8C'
FIBRE, CROSS, ARROW = '#3A3A3A', '#C6C6C6', '#C2410C'

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5,
                     'xtick.labelsize': 10, 'ytick.labelsize': 9,
                     'axes.edgecolor': '#B8B8B8'})

store = np.load('green_dists.npy', allow_pickle=True).item()


def schematic(ax, kind):
    """Which distance is measured. Crossing family drawn ORTHOGONAL to the measured one."""
    ys = [0.12, 0.39, 0.66, 0.93]
    xr = 0.40 if kind == 'nematic' else 0.96      # lines stop early to leave room for text
    if kind == 'mesh':
        for x in [0.14, 0.37, 0.60, 0.83]:
            ax.plot([x, x], [0.06, 0.99], '-', color=CROSS, lw=2.6,
                    solid_capstyle='round', zorder=1)
    for y in ys:
        ax.plot([0.04, xr], [y, y], '-', color=FIBRE, lw=2.6,
                solid_capstyle='round', zorder=2)
    ax.annotate('', xy=(0.24 if kind == 'nematic' else 0.48, ys[1]),
                xytext=(0.24 if kind == 'nematic' else 0.48, ys[2]), zorder=4,
                arrowprops=dict(arrowstyle='<->', color=ARROW, lw=1.8,
                                mutation_scale=10, shrinkA=0, shrinkB=0))
    if kind == 'nematic':
        # right-aligned to the drawing edge, so the label can never leave the panel
        ax.text(1.0, (ys[1] + ys[2]) / 2, 'spacing', color=ARROW, fontsize=9,
                va='center', ha='right', fontweight='bold', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()


FW, FH = 6.5, 4.40
fig = plt.figure(figsize=(FW, FH))

ax = fig.add_axes([0.095, 0.150, 0.605, 0.630])

for i, (h, c) in enumerate([(32, NEM), (52, MESH)]):
    x = np.concatenate(store[h])
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    lo, hi = np.percentile(x, [10, 90])

    ax.plot([i, i], [lo, q1], '-', color=c, lw=1.3, zorder=2)
    ax.plot([i, i], [q3, hi], '-', color=c, lw=1.3, zorder=2)
    for y in (lo, hi):
        ax.plot([i - 0.07, i + 0.07], [y, y], '-', color=c, lw=1.3, zorder=2)
    ax.add_patch(FancyBboxPatch((i - 0.23, q1), 0.46, q3 - q1,
                                boxstyle='round,pad=0,rounding_size=0.030',
                                facecolor=c, alpha=0.22, edgecolor=c, lw=1.5, zorder=3))
    ax.plot([i - 0.23, i + 0.23], [med, med], '-', color=c, lw=2.6, zorder=4,
            solid_capstyle='butt')

    rng = np.random.default_rng(4 + i)
    ax.plot(i + 0.325 + rng.uniform(-0.022, 0.022, 3),
            [np.median(d) for d in store[h]], 'o', color=c, ms=6.5,
            mfc='white', mew=1.6, zorder=5)

    ax.text(i, 5.40, 'median %.2f' % med, ha='center', va='bottom',
            fontsize=9, color=c, fontweight='bold')
    ax.text(i, 4.93, 'IQR %.2f–%.2f' % (q1, q3), ha='center', va='bottom',
            fontsize=9, color=c)

ax.set_xticks([0, 1]); ax.set_xticklabels(['32 hpf', '52 hpf'])
ax.set_xlim(-0.62, 1.62); ax.set_ylim(1.35, 5.95)
ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
ax.set_ylabel('spacing (µm)')
ax.grid(axis='y', alpha=0.13, lw=0.8)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.tick_params(axis='both', length=3.5, color='#B8B8B8')

# ---- arrangement panels, far-right column: nematic on top, mesh below
PW_IN, PH_IN = 1.36, 1.088                  # near-square (1.25:1), as the schematics were drawn
PW, PH = PW_IN / FW, PH_IN / FH
PX = 0.985 - PW                             # flush to the right margin
for c, kind, head, sub, py in [(NEM, 'nematic', '32 hpf · nematic', 'one family', 0.562),
                               (MESH, 'mesh', '52 hpf · mesh', 'two families', 0.147)]:
    fig.patches.append(FancyBboxPatch((PX, py), PW, PH,
        boxstyle='round,pad=0.008,rounding_size=0.016', transform=fig.transFigure,
        facecolor=c, alpha=0.09, edgecolor='none', zorder=4))
    fig.patches.append(FancyBboxPatch((PX, py), PW, PH,
        boxstyle='round,pad=0.008,rounding_size=0.016', transform=fig.transFigure,
        facecolor='none', edgecolor=c, linewidth=1.6, zorder=6,
        mutation_aspect=FW / FH))

    pad = 0.075
    sax = fig.add_axes([PX + pad * PW, py + pad * PH,
                        PW * (1 - 2 * pad), PH * (1 - 2 * pad)])
    schematic(sax, kind); sax.set_zorder(8); sax.patch.set_alpha(0)

    # right-aligned to the panel edge: the heading is wider than the panel, and centring
    # it would push it off the figure
    fig.text(PX + PW, py + PH + 0.020, head, ha='right', va='bottom', fontsize=9,
             fontweight='bold', color=c)
    fig.text(PX + PW / 2, py - 0.022, sub, ha='center', va='top', fontsize=9, color=c)

fig.text(0.035, 0.988, 'Fibre spacing at the two endpoints',
         fontsize=10.5, fontweight='bold', va='top', color=INK)
fig.text(0.035, 0.940, 'spacing to the nearest parallel fibre  ·  three hearts per stage',
         fontsize=9, va='top', color='#6E6E6E')
fig.text(0.035, 0.018, 'whiskers = 10th–90th percentile  ·  open circles = per-heart medians',
         fontsize=9, va='bottom', color='#6E6E6E')

fig.savefig('spacing_box.png', dpi=400)
fig.savefig('spacing_box.pdf')
print('figure %.2f x %.2f in (aspect %.2f) — place at 100%% in Word'
      % (FW, FH, FW / FH))
print('panel %.2f x %.2f in (aspect %.2f)' % (PW_IN, PH_IN, PW_IN / PH_IN))
for h in (32, 52):
    q = np.percentile(np.concatenate(store[h]), [25, 50, 75])
    print('%d hpf  IQR %.2f-%.2f  median %.2f' % (h, q[0], q[2], q[1]))
