"""
raw_detect.py -- fibre detection on the UNTRACED fluorescence, feeding the identical
spacing measure used on the traces.

WHY THIS IS A REAL INDEPENDENT CHECK
------------------------------------
The traced result carries the tracer's eye inside it: which fibres were drawn, where a
myofibril was judged to start and stop, and whether a cell boundary counted. A detector
has none of those decisions. If both substrates give the same spacing, the traced number
is not an artefact of tracing. If they disagree, the disagreement localises the cause.

WHY IT IS ALSO THE HARDER SUBSTRATE, and the record says so
-----------------------------------------------------------
On raw fluorescence the project already measured: additive pixel grain is a MAJOR confound
(dominance 0.91 -> 0.43) and binarising first does not fix it; blur on a two-family texture
pushes the measure the other way; and at 32 hpf the bright structure is partly the CORTICAL
HONEYCOMB, which the tracer excluded by judgement and a detector cannot.

ORDER OF OPERATIONS, and the reason
-----------------------------------
Resample to the common 0.40 um/px grid BEFORE detecting, not after. Input pixel scale is
correlated with stage (0.361 um/px early, 0.5675 late), so a detector run at native scale
would use a different effective kernel at each timepoint -- a systematic change in the
measured object along the axis under test. Detecting on a common grid makes the kernel
identical everywhere. This is the same argument that fixed the pen width on the traces.

DETECTION
---------
Sato tubeness at the fibre scale (0.4-0.9 um, the scale band the project already established
for fibres as opposed to the 1.5-3 um boundary scale), then a per-image threshold at a fixed
PERCENTILE of the tubeness response, then skeletonise.

The percentile, not an absolute value, is the free parameter: absolute intensity varies 3x
across this set (image means 24.8 to 85.9) because brightness was never normalised between
acquisitions, so any absolute threshold would be a different effective threshold per image --
the exact confound the project's ImageJ rule ("never auto-adjust brightness per image") exists
to prevent. The percentile is swept and reported, never fitted.
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects

import spacing as sp

TARGET = 0.40
FIELD_UM = 61.4
FIBRE_SCALES_UM = (0.4, 0.9)     # established fibre scale band, not fitted here
MIN_OBJ_PX = 12                  # drop specks below ~5 um of skeleton


def load_raw(path, target=TARGET, field_um=FIELD_UM):
    """Read, resample to the common grid, centre-crop to the common field."""
    a = tifffile.imread(path).astype(float)
    if a.ndim == 3:
        a = a[..., 0]
    um = sp.read_scale(path)
    zoom = um / target
    if abs(zoom - 1.0) > 1e-6:
        a = ndi.zoom(a, zoom, order=1)
    n = int(round(field_um / target))
    h, w = a.shape
    if h < n or w < n:
        a = np.pad(a, ((max(0, (n - h + 1) // 2), max(0, n - h - (n - h + 1) // 2)),
                       (max(0, (n - w + 1) // 2), max(0, n - w - (n - w + 1) // 2))))
        h, w = a.shape
    i0, j0 = (h - n) // 2, (w - n) // 2
    return a[i0:i0 + n, j0:j0 + n], um


def tubeness(img, target=TARGET, scales_um=FIBRE_SCALES_UM):
    lo, hi = [s / target for s in scales_um]
    sigmas = np.linspace(lo, hi, 3)
    return sato(img, sigmas=sigmas, black_ridges=False)


def detect(img, percentile=80.0, target=TARGET, min_obj=MIN_OBJ_PX):
    """Tubeness -> percentile threshold -> skeleton."""
    t = tubeness(img, target)
    thr = np.percentile(t, percentile)
    m = t > thr
    m = remove_small_objects(m, min_obj)
    sk = skeletonize(m)
    sk = remove_small_objects(sk, min_obj, connectivity=2)
    return sk, t


def agreement(sk_det, sk_trace, tol_um=1.2, target=TARGET):
    """How much of the detected skeleton lies within tol of the traced one, and vice versa.
    The trace is the reference standard for WHERE fibres are, so this scores the detector
    without reference to any spacing number."""
    tol_px = tol_um / target
    d_to_trace = ndi.distance_transform_edt(~sk_trace)
    d_to_det = ndi.distance_transform_edt(~sk_det)
    prec = float((d_to_trace[sk_det] <= tol_px).mean()) if sk_det.any() else np.nan
    rec = float((d_to_det[sk_trace] <= tol_px).mean()) if sk_trace.any() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
    return prec, rec, f1
