"""Where do nucleation sites actually sit, under the centroid rule vs the membrane rule?

    OLD  _seed_angle(col,row) = argmax over the four cardinals of (cx-col)cos a + (cy-row)sin a
         -> the direction that points at the CELL CENTROID, snapped to a lattice axis.
         A site is kept only if that direction is primary, so the kept/dropped boundary is
         |col - cx| = |row - cy|.  Every point on the flat top/bottom membranes has
         |row - cy| = hex_half_h, so the whole middle |col - cx| < hex_half_h of those
         membranes is dropped: a dead band exactly one CELL HEIGHT wide, produced by an
         accident of the centroid construction rather than by anything local.

    NEW  the inward normal of the membrane segment the site came from, snapped the same way.
         _seed_sites() ALREADY computes that normal -- (-sin f.angle, cos f.angle) -- to place
         the site one pixel inside the cortex line, and then throws it away.

This script only counts and maps. It does not run the simulation.
"""
from __future__ import annotations

import numpy as np
from bundle_model import cell_particle as CP
from bundle_model.parameters import Params, UM_PER_LATTICE as LU

P = CP.P


def snap(dcol, drow):
    return max((0.0, np.pi / 2, np.pi, 3 * np.pi / 2),
               key=lambda a: dcol * np.cos(a) + drow * np.sin(a))


def build():
    s = CP.Simulation.__new__(CP.Simulation)
    s._cx, s._cy = P.W / 2.0, P.H / 2.0
    s.mask = s._make_hex_mask()
    s.field = CP.MonomerField(s.mask)
    s.mesh_axis = P.mesh_axis
    s.fibres = {'nematic': [], 'cortex': [], 'mesh': [], 'dead': []}
    s._lines_cache = {}
    s.phi = {True: 0.0, False: 0.0}
    if not hasattr(s, "phase"):
        s.phase = s.phi
    s.seed_cortex()
    return s


if __name__ == "__main__":
    s = build()
    rows = []
    for f in s.fibres['cortex']:
        ncol, nrow = -np.sin(f.angle), np.cos(f.angle)
        for (col, row) in f.centre_pts:
            pt = (round(col + ncol), round(row + nrow))
            a_old = snap(s._cx - pt[0], s._cy - pt[1])
            a_new = snap(ncol, nrow)
            # rule C: keep the ASSUMPTION (fibres run along the long axis, into the cell)
            # and drop only the accident (the choice of axis coming from the centroid).
            # Candidates are the primary pair only; the interior direction picks between them.
            a_axis = 0.0 if (s._cx - pt[0]) >= 0 else np.pi
            rows.append(dict(col=pt[0], row=pt[1],
                             edge=round(np.degrees(f.angle)) % 360,
                             old=a_old, new=a_new, axis=a_axis,
                             keep_old=CP.is_primary(a_old + s.mesh_axis),
                             keep_new=CP.is_primary(a_new + s.mesh_axis),
                             keep_axis=CP.is_primary(a_axis + s.mesh_axis)))
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv("SEEDANGLE_sites.csv", index=False)
    print(f"cell {P.hex_half_w*2*LU:.1f} x {P.hex_half_h*2*LU:.1f} um   "
          f"pool {len(d)} sites ({d[['col','row']].drop_duplicates().shape[0]} unique)")
    print(f"  OLD rule (centroid) kept {d.keep_old.sum():3d}   dropped {(~d.keep_old).sum():3d}")
    print(f"  NEW rule (membrane) kept {d.keep_new.sum():3d}   dropped {(~d.keep_new).sum():3d}")
    print(f"  RULE C  (long axis) kept {d.keep_axis.sum():3d}   dropped {(~d.keep_axis).sum():3d}")
    print(f"  old vs new disagree on {int((d.keep_old != d.keep_new).sum())} sites; "
          f"old vs C on {int((d.keep_old != d.keep_axis).sum())}")
    same_dir = (d.old == d.axis)
    print(f"  of the {d.keep_old.sum()} sites the OLD rule keeps, rule C fires "
          f"{int(same_dir[d.keep_old].sum())} in the SAME direction")
    print()
    print("per membrane edge (edge = its own angle in deg):")
    g = d.groupby("edge").agg(n=("col", "size"), kept_old=("keep_old", "sum"),
                              kept_new=("keep_new", "sum"))
    print(g.to_string())
    print()
    kd = d[~d.keep_old]
    print(f"OLD dropped sites span |col-cx| = {np.abs(kd.col - s._cx).min():.0f}"
          f"..{np.abs(kd.col - s._cx).max():.0f} lu "
          f"(= {np.abs(kd.col-s._cx).max()*LU:.1f} um);  hex_half_h = {P.hex_half_h} lu")
    kn = d[~d.keep_new]
    print(f"NEW dropped sites: edges {sorted(kn.edge.unique().tolist())}, "
          f"|row-cy| = {np.abs(kn.row - s._cy).min():.0f}..{np.abs(kn.row - s._cy).max():.0f} lu")
