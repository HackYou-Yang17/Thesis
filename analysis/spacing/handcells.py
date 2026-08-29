"""
handcells.py -- cell dimensions from the HAND-TRACED 52 hpf outlines.

Extraction is exact: ImageJ red (237,28,36) on a greyscale image, so (R - max(G,B)) > 60
recovers the annotation with no threshold to choose. Cells are the enclosed regions of that
outline network. No watershed, no seed parameter, no segmentation quality to argue about --
the tracer decided where every boundary is.

Pixel counts are computed TWICE by different routes and required to agree exactly.
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops

PATH = '/mnt/user-data/uploads/analysis/52hpf cell size - Copy.tif'
UM = 0.4815


def trace():
    a = tifffile.imread(PATH)
    r = a[..., 0].astype(np.int16); g = a[..., 1].astype(np.int16); b = a[..., 2].astype(np.int16)
    return (r - np.maximum(g, b)) > 60


def cells(close_px=0):
    m = trace()
    if close_px:
        m = binary_closing(m, disk(close_px))
    lab = label(~m, connectivity=1)
    return lab, m


def measure(lab, um=UM, min_um2=20.0):
    h, w = lab.shape
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    counts = np.bincount(lab.ravel())               # route 1: explicit pixel count per label
    rows = []
    for r in regionprops(lab):
        if r.label in edge:
            continue
        n_bincount = int(counts[r.label])
        n_props = int(r.area)
        n_direct = int((lab == r.label).sum())      # route 2: count the mask, one by one
        assert n_bincount == n_props == n_direct, (r.label, n_bincount, n_props, n_direct)
        a = n_direct * um ** 2
        if a < min_um2:
            continue
        L = r.axis_major_length * um
        W = r.axis_minor_length * um
        rows.append(dict(label=r.label, px=n_direct, area=a, length=L,
                         w_ellipse=W, w_equiv=a / L if L > 0 else np.nan,
                         aspect=L / W if W > 0 else np.nan, solidity=r.solidity))
    return rows
