"""
make_order_fig.py -- orientational order in the traced atrium, two ways.

Replaces c2c4_transition.png. Panel A is the band statistic against stage, as
before. Panel B replaces the old C2/C4-against-stage panel with the C2-C4 plane,
which carries the reference corners, the distance from them and the direction of
travel in one picture.

House rules followed: no text-text overlaps, the title carries no numbers or
metrics, no dead space.

SIZING. Authored at the printed size. main.tex is a4paper with margin=2.2cm, so
\\textwidth = 21.0 - 4.4 = 16.6 cm = 6.54 in. A figure* included at
width=\\textwidth is therefore NOT rescaled, and FS pt here is FS pt on the page.
Changing FW or FH breaks that, so change the font size with it.

LAYOUT. A and B side by side. Text is 8.5 pt as printed -- the midpoint between
the old figure (13 pt on a 14.6 in canvas, which width=\textwidth scaled down to
about 5.9 pt on the page) and the 11 pt version, which read as oversized. At
8.5 pt the legends fit inside the axes and the series can be labelled directly.

Inputs
  order_traced.json    per-heart band and [C2, C4] by stage, from fft_c4.py
  model_c2c4.json      OPTIONAL, {stage: [[C2, C4], ...]} one entry per seed.
                       Present -> the model path is drawn in panel B.

Output
  order_transition.png
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTE, GRID = '#2E2E2E', '#8C8C8C', '#D0D0D0'
NEM, MESH, MODEL = '#C4762A', '#388561', '#C0392B'
CMAP = plt.get_cmap('viridis')
STAGES = [32, 36, 40, 44, 48, 52]
CROSS = 40.8
REFS = {'nematic': (0.98, 0.93), 'tetratic': (0.11, 0.93),
        '$60^\\circ$': (0.45, 0.56), 'isotropic': (0.28, 0.42)}
REF_DX = {'nematic': -0.045, 'tetratic': 0.045, '$60^\\circ$': -0.045, 'isotropic': 0.045}
REF_HA = {'nematic': 'right', 'tetratic': 'left', '$60^\\circ$': 'right', 'isotropic': 'left'}
FS = 8.5
plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS,
                     'ytick.labelsize': FS, 'axes.edgecolor': '#B8B8B8'})

D = json.load(open('order_traced.json'))
band, spec = D['band'], D['spectral']
norm = matplotlib.colors.Normalize(vmin=32, vmax=52)

FW, FH = 6.54, 2.85
fig = plt.figure(figsize=(FW, FH))
axA = fig.add_axes([0.080, 0.180, 0.375, 0.735])
axB = fig.add_axes([0.565, 0.180, 0.323, 0.735])

# ---- A  the band statistic ----
b = np.array([band[str(h)] for h in STAGES])
m, sd = b.mean(axis=1), b.std(axis=1, ddof=1)
axA.fill_between(STAGES, m - sd, m + sd, color=NEM, alpha=0.18, lw=0)
axA.fill_between(STAGES, 1 - m - sd, 1 - m + sd, color=MESH, alpha=0.18, lw=0)
axA.plot(STAGES, m, '-o', color=NEM, lw=1.6, ms=4.5, mec='white', mew=1.0)
axA.plot(STAGES, 1 - m, '-s', color=MESH, lw=1.6, ms=4.5, mec='white', mew=1.0)
axA.axhline(0.5, color=MUTE, ls=':', lw=1.5, zorder=1)
axA.plot([CROSS], [0.5], 'o', ms=5.5, mfc='white', mec=INK, mew=1.4, zorder=5)
axA.annotate('crossing', xy=(CROSS, 0.5), xytext=(CROSS + 0.2, 0.10), fontsize=FS,
             color=INK, ha='center', va='bottom',
             arrowprops=dict(arrowstyle='-', color=INK, lw=0.9, shrinkA=2, shrinkB=5))
axA.text(52.4, 0.96, 'above $\\pm45^\\circ$', color=MESH, fontsize=FS,
         ha='right', va='top')
axA.text(52.4, 0.04, 'within $\\pm45^\\circ$', color=NEM, fontsize=FS,
         ha='right', va='bottom')
axA.set_xlabel('hours post fertilisation')
axA.set_ylabel('fraction of oriented power')
axA.set_xlim(31.2, 52.9); axA.set_ylim(0.0, 1.0)
axA.set_xticks(STAGES); axA.set_yticks([0.0, 0.5, 1.0])
axA.set_title('A  angular power leaves the dominant axis', fontsize=FS, pad=4, loc='left')
axA.grid(True, color='#F0F0F0', lw=0.9, zorder=0); axA.set_axisbelow(True)

# ---- B  the C2-C4 plane ----
d = np.linspace(0, np.pi / 2, 400)
axB.plot(np.abs(np.cos(d)), np.abs(np.cos(2 * d)), '-', color=GRID, lw=1.9, zorder=1)
for name, (c2, c4) in REFS.items():
    axB.plot(c2, c4, marker='s', ms=5.5, mfc='white', mec=INK, mew=1.2, ls='none', zorder=5)
    axB.text(c2 + REF_DX[name], c4, name, color=INK, fontsize=FS,
             ha=REF_HA[name], va='center', zorder=6)
for h in STAGES:
    p = np.array(spec[str(h)])
    axB.plot(p[:, 0], p[:, 1], 'o', ms=3.0, color=CMAP(norm(h)), alpha=0.32,
             mec='none', ls='none', zorder=2)
mean = np.array([np.array(spec[str(h)]).mean(axis=0) for h in STAGES])
axB.plot(mean[:, 0], mean[:, 1], '-', color=INK, lw=1.5, alpha=0.55, zorder=3)
for i, h in enumerate(STAGES):
    axB.plot(mean[i, 0], mean[i, 1], 'o', ms=6.0, color=CMAP(norm(h)),
             mec='white', mew=1.0, zorder=4)
axB.annotate('', xy=mean[-1], xytext=mean[-2], zorder=4,
             arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.5, mutation_scale=10,
                             shrinkA=5, shrinkB=5))
axB.text(mean[0, 0] + 0.035, mean[0, 1] - 0.025, '32', color=INK, fontsize=FS,
         ha='left', va='top', fontweight='bold', zorder=6)
axB.text(mean[-1, 0] - 0.040, mean[-1, 1] + 0.025, '52', color=INK, fontsize=FS,
         ha='right', va='bottom', fontweight='bold', zorder=6)
axB.set_xlabel('$C_2$   single-axis order')
axB.set_ylabel('$C_4$   four-fold order')
axB.set_xlim(-0.05, 1.10); axB.set_ylim(-0.05, 1.10)
axB.set_xticks([0, 0.5, 1.0]); axB.set_yticks([0, 0.5, 1.0])
axB.set_title('B  where it sits, and where it travels', fontsize=FS, pad=4, loc='left')
axB.grid(True, color='#F0F0F0', lw=0.9, zorder=0); axB.set_axisbelow(True)

# ---- legend for B, inside the axes ----
handles = [Line2D([], [], marker='o', ms=6.0, color=CMAP(norm(42)), ls='-', lw=1.6,
                  mec='white', mew=1.0, label='stage mean'),
           Line2D([], [], marker='o', ms=3.2, color=CMAP(norm(42)), ls='none',
                  alpha=0.5, label='one heart'),
           Line2D([], [], marker='s', ms=5.5, mfc='white', mec=INK, mew=1.2, ls='none',
                  label='reference'),
           Line2D([], [], color=GRID, lw=1.9, label='ideal locus')]
if os.path.exists('model_c2c4.json'):
    mspec = json.load(open('model_c2c4.json'))
    mm = np.array([np.array(mspec[str(h)]).mean(axis=0) for h in STAGES])
    axB.plot(mm[:, 0], mm[:, 1], '-', color=MODEL, lw=2.0, zorder=3)
    axB.plot(mm[:, 0], mm[:, 1], 's', ms=7, color=MODEL, mec='white', mew=1.2, zorder=4)
    handles.insert(3, Line2D([], [], marker='s', ms=6, color=MODEL, ls='-', lw=1.9,
                             mec='white', mew=1.2, label='model'))
else:
    print('model_c2c4.json not found -- traced path only')

axB.legend(handles=handles, loc='lower left', fontsize=FS, frameon=True,
           framealpha=0.95, edgecolor='#CCCCCC', handlelength=1.4,
           labelspacing=0.4, handletextpad=0.5, borderpad=0.4)

cax = fig.add_axes([0.910, 0.180, 0.016, 0.735])
cb = matplotlib.colorbar.ColorbarBase(cax, cmap=CMAP, norm=norm)
cb.set_label('hpf', fontsize=FS)
cb.set_ticks(STAGES); cax.tick_params(labelsize=FS)

fig.savefig('order_transition.png', dpi=220)
print('wrote order_transition.png')
