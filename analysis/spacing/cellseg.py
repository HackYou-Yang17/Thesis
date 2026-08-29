"""
cellseg.py -- cell areas, lengths and widths from the 52 hpf cell-boundary image.

Membrane-marker segmentation: cell interiors are dark, junctions are bright, so the image is
used directly as a watershed elevation map. Seeds are the regional minima of the smoothed
image after an h-minima transform, which is what stops one cell being split into several by
noise.

"Isotropic cells only" is applied as a DECLARED filter with a reported sensitivity, because
dropping elongated regions is a selection and the threshold decides the answer.
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import h_minima, binary_closing, disk, remove_small_objects
from skimage.segmentation import watershed, clear_border
from skimage.measure import label, regionprops
from datapaths import ROOT as DATA_ROOT

PATH = DATA_ROOT + '/52hpf cell size.tif'
UM = 0.4815


def load():
    a = tifffile.imread(PATH).astype(float)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def tissue_mask(img, um=UM):
    """The heart does not fill the frame; the black surround must be excluded."""
    sm = gaussian(img, 6.0 / um)
    thr = sm.max() * 0.18
    m = sm > thr
    m = binary_closing(m, disk(int(round(4.0 / um))))
    m = ndi.binary_fill_holes(m)
    return remove_small_objects(m, int(500 / um ** 2))


def segment(img, h_rel=0.055, smooth_um=1.1, um=UM):
    sm = gaussian(img, smooth_um / um)
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    tis = tissue_mask(img, um)
    seeds = label(h_minima(sm, h_rel) & tis)
    lab = watershed(sm, markers=seeds, mask=tis)
    return lab, tis


def measure(lab, tis, um=UM, min_um2=25.0, max_um2=1500.0):
    inner = clear_border(lab)                       # a cell cut by the frame is not measurable
    rows = []
    for r in regionprops(inner):
        a = r.area * um ** 2
        if not (min_um2 <= a <= max_um2):
            continue
        L = r.axis_major_length * um
        W = r.axis_minor_length * um
        if W <= 0:
            continue
        rows.append(dict(px=r.area, area=a, length=L, width=W, aspect=L / W,
                         solidity=r.solidity, label=r.label))
    return rows
