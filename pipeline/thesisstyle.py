"""Thesis figure style: authored at the printed size, so nothing is ever scaled down.

    printed_pt = authored_pt * (textwidth_in / FW)        [stated] Luka's rule, FW = 6.54 in

Every figure here is authored at EXACTLY FW inches wide, so the ratio is 1 and authored point
sizes ARE printed point sizes. Nothing may be smaller than MIN_PT.

WHY THIS FORCED A REDESIGN RATHER THAN A FONT BUMP. The v13 figures were authored 15-20 in wide.
Placed at 6.54 in that is a 0.32-0.44x reduction, so their 12 pt axis labels printed at 3.9-5.2 pt
and their 8 pt annotations at 2.6-3.5 pt. Recovering 11 pt printed by raising the authored size
would have needed 20-28 pt type on the original canvases, which does not fit. The width has to
come down instead. audit() is the check that the result actually complies.

MIN_PT was 11 and is now 9 at [stated] Luka's request. The PANEL CUTS were made on redundancy
grounds -- a panel fully contained in the one beside it, a panel plotting an arithmetic identity,
a panel needing a disclaimer saying its own agreement was spurious -- and those reasons do not
depend on type size, so lowering the floor does not restore any of them.

Figures may be TALLER than FW; height costs a reader nothing and width costs legibility.

savefig.bbox is deliberately NOT "tight". Tight bbox trims the canvas to the drawn content,
so a figure authored at 6.54 in can be saved at 5.3 in and then be scaled UP by 1.24x when
placed at \textwidth -- harmless for legibility but it makes type size differ between
figures in the same document. Saving the full canvas keeps every figure at exactly 1.000x.
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FW = 6.54          # \textwidth in inches -- the printed width of every figure here
MIN_PT = 9.0       # nothing on any figure may print smaller than this
FS = 9             # base size: authored == printed, so this IS 9 pt on the page

COL = dict(traced="#1f4e9c", model="#c0392b", model2="#e08214",
           nematic="#e08214", quartic="#2e8b57", grey="#5a5a5a",
           band="#9ecae1", gate="#b0b0b0")
AGES = [32, 36, 40, 44, 48, 52]


def use_style():
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "figure.dpi": 130, "savefig.dpi": 400, "savefig.bbox": None,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })


def panel_title(ax, text, width=34, pad=4):
    """Titles WRAP rather than shrink. Shrinking is what produced sub-11 pt text before."""
    return ax.set_title("\n".join(textwrap.wrap(text, width)), fontsize=FS, pad=pad, loc="left")


def age_axis(ax, label=True):
    ax.set_xticks(AGES)
    ax.set_xlim(31, 53)
    if label:
        ax.set_xlabel("hours post fertilisation")


def audit(fig, name):
    """Every Text object on the figure, checked against MIN_PT. Prints a verdict.

    Reports the figure's authored width too, because the whole guarantee rests on it being FW.
    """
    fig.canvas.draw()
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        pt = t.get_fontsize()
        if pt < MIN_PT - 1e-6:
            bad.append((round(pt, 2), s.replace("\n", " ")[:48]))
    w = fig.get_size_inches()[0]
    scale = FW / w
    print(f"[audit] {name}: authored {w:.2f} x {fig.get_size_inches()[1]:.2f} in, "
          f"printed scale {scale:.3f}")
    if abs(w - FW) > 0.02:
        print(f"        !! authored width is not {FW} in — printed sizes will be "
              f"{scale:.3f}x authored")
    if bad:
        print(f"        !! {len(bad)} text objects below {MIN_PT} pt:")
        for pt, s in sorted(set(bad))[:12]:
            print(f"           {pt:5.2f} pt  {s}")
    else:
        print(f"        all text >= {MIN_PT:.0f} pt printed")
    return bad
