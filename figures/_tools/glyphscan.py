"""Smallest type in a rendered figure, measured from the pixels.

For figures whose build script is not to hand, the only way to check the type floor is to measure
the glyphs. Connected dark components are collected, rules/markers/axis lines are filtered out by
shape, and the height distribution is reported. In DejaVu Sans a digit or capital is ~0.70 em and
a lowercase x-height is ~0.52 em, so a run of h px at ppi = width_px / 6.54 implies

    printed_pt = 72 * (h / 0.70) / ppi      for a cap-height glyph

CALIBRATION: run it on a figure whose authored size is known (mine are 9 pt at scale 1.000). If it
returns ~9 pt there, the estimate is trustworthy on the others.
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

FW = 6.54
im = Image.open(sys.argv[1]).convert("L")
W, H = im.size
a = np.asarray(im) < 150
lab, n = ndi.label(a)
objs = ndi.find_objects(lab)
hs = []
for sl in objs:
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    if h < 4 or h > 120 or w < 2 or w > 120:      # rules, markers, long lines
        continue
    if w > 6 * h or h > 8 * w:                    # dashes and bars
        continue
    hs.append(h)
hs = np.array(sorted(hs))
ppi = W / FW
def pt(h, r=0.70):
    return 72.0 * (h / r) / ppi
print(f"{sys.argv[1].split('/')[-1]}: {W}x{H} px -> {ppi:.0f} px/in at {FW} in, {len(hs)} glyph-like components")
for q in (5, 10, 25, 50, 75, 95):
    h = np.percentile(hs, q)
    print(f"  p{q:<2d} height {h:5.1f} px   cap-basis {pt(h):5.1f} pt   x-height-basis {pt(h,0.52):5.1f} pt")
