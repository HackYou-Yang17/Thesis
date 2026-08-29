"""
green.py -- the new yellow-green traces (32 and 52 hpf), measured through the identical
pipeline used for the red traces.

Extraction is exact and has no free parameter: the annotation is ImageJ yellow-green
(181, 230, 29) and the underlying image is greyscale, so any pixel with green != blue is
trace. Verified per file: every pixel with ANY channel difference is a trace pixel, and none
falls between the pure colour and grey except a handful of blend pixels at g-b = 111.
"""
import glob, os
import numpy as np
import tifffile
import spacing as sp, nn

OFF = 0.20
R_UM = 5.0


def extract_green(path):
    a = tifffile.imread(path)
    g = a[..., 1].astype(np.int16); b = a[..., 2].astype(np.int16)
    return (g - b) > 60


def load(path):
    return sp.normalise(extract_green(path), sp.read_scale(path))


def files():
    out = []
    for h in (32, 52):
        for f in sorted(glob.glob('/mnt/user-data/uploads/analysis/%dhpf/*Copy (2).tif' % h)):
            out.append((h, f))
    return out


if __name__ == '__main__':
    print('%-12s %5s %8s %8s %8s %8s %8s %8s'
          % ('heart', 'hpf', 'lineDens', 'n_pairs', 'censored', 'p25', 'median', 'p75'))
    store = {}
    for h, f in files():
        name = os.path.basename(f).replace(' - Copy (2).tif', '')
        sk = load(f)
        d, _, cens = nn.nn_spacings(sk, max_um=R_UM, step_px=0.1)
        d = d + OFF
        store.setdefault(h, []).append(d)
        q = np.percentile(d, [25, 50, 75])
        print('%-12s %5d %8.3f %8d %8.2f %8.2f %8.2f %8.2f'
              % (name, h, sp.line_density(sk), len(d), cens, q[0], q[1], q[2]))
    np.save('green_dists.npy', store, allow_pickle=True)
    print()
    for h in (32, 52):
        x = np.concatenate(store[h])
        q = np.percentile(x, [25, 50, 75])
        print('%d hpf pooled: n = %d   IQR %.2f - %.2f um   median %.2f   mean %.2f'
              % (h, len(x), q[0], q[2], q[1], x.mean()))
