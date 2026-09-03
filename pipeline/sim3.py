"""v5 run harness: run the model sheet and measure it through measure.py.

Replaces the lost sim2.py. Differences from sim_dominance.py (v1/v2), all deliberate:

  * every tunable is plumbed, including cadherin_nucleation_prob and n_sub. The handover
    records that sim2.py silently ran these at baseline for a whole 44-run sweep because
    they were not in TUNED_V1. DEFAULTS below is the authority; _configure sets EVERY key
    on every call so nothing leaks between runs in one process.
  * the crop comes from sheet.crop_bounds(), the model's own declared measurement window,
    rather than a centre-crop of the canvas. multicell_particle documents crop_bounds as
    "single source of truth for where the render cuts", and using it means the statistic
    and the rendered figure look at the same pixels.
  * gap_percentiles, foam and fibre count are recorded at every timepoint (v5 loss needs
    gap; foam and p95 are the two hard gates).
  * check_conservation() runs at the end of every run -- it was written and never called.
"""
from __future__ import annotations

import numpy as np

import measure as M
from bundle_model import multicell_particle as mc
from bundle_model.cell_particle import P
from bundle_model.parameters import UM_PER_LATTICE as LU

HPF = np.array([32.0, 36.0, 40.0, 44.0, 48.0, 52.0])

# HISTORICAL ONLY -- DO NOT USE AS A DEFAULT. Kept under its own name for provenance.
# Until 2 Sep 2026 this dict WAS the default: configure() with no overrides set every rate from
# it, silently overwriting whatever bundle_model/parameters.py commits. parameters.py has held
# the v13 tune since 27 Aug, so run_once(seed=S) would have produced a v5 run that every log,
# CSV and figure caption would have called v13 -- rate_grow off by 5.8x, angle_noise by 2.3x.
# Nothing in the repo was actually affected (every caller passed an explicit overrides dict, and
# that was checked before this was changed), but the trap was one careless call from firing.
# THE DEFAULT IS NOW parameters.py, so the file a reader inspects is the file that runs.
# v4 tune 2 -- the rate anchor (handover S5). NOT a fit at this geometry.
# v5 tune (parameters_v5_tuned.py) -- the anchor for the SATURATED counterfactual, since it
# is the best rate set known at this geometry. phi_max is carried here as an explicit knob.
ANCHOR_V5 = dict(rate_nematic_depoly=0.011, rate_nematic_poly=0.006,
              rate_grow=0.0283, rate_thin=0.006226, rate_branch=0.0020,
              rate_nucleate=0.120, nematic_thresh=0.35,
              angle_noise=float(np.pi / 10), axis_spread=float(np.pi / 6),
              cadherin_nucleation_prob=0.4, n_sub=4, phi_max=None)

# phi_max = monomers_per_point / monomers_per_seg -- the fraction of the cell that can be
# filled with saturated bundle before the actin runs out. It is the ONLY route through which
# MONOMER_CONC enters, so setting it is exactly equivalent to setting MONOMER_CONC, and it is
# dimensionless. phi_max=None leaves parameters.py alone (300 uM, phi_max = 1.056).
PHI_MAX_PER_UM = 1.0557 / 300.0        # phi_max per uM of MONOMER_CONC, at this geometry

FLOAT_KEYS = ("rate_nematic_depoly", "rate_nematic_poly", "rate_grow", "rate_thin",
              "rate_branch", "rate_nucleate", "nematic_thresh", "angle_noise",
              "axis_spread", "cadherin_nucleation_prob")
INT_KEYS = ("n_sub",)
BASE_MPP = float(P.monomers_per_point)      # as parameters.py has it, before any override


KNOBS = FLOAT_KEYS + INT_KEYS + ("phi_max",)


def _read_committed():
    """Every tunable exactly as bundle_model/parameters.py commits it."""
    d = {k: float(getattr(P, k)) for k in FLOAT_KEYS}
    d.update({k: int(getattr(P, k)) for k in INT_KEYS})
    d["phi_max"] = None
    return d


# Snapshotted AT IMPORT, before any configure() call can mutate P. Reading it later would return
# whatever the last configure() left behind, which is exactly the class of bug this replaces.
COMMITTED = _read_committed()


def committed():
    """The committed tune, as a fresh dict. This is what configure() defaults to."""
    return dict(COMMITTED)


def configure(overrides=None):
    k = {**COMMITTED, **(overrides or {})}
    unknown = set(k) - set(KNOBS)
    if unknown:
        raise KeyError(f"unknown knob(s): {sorted(unknown)}")
    for name in FLOAT_KEYS:
        setattr(P, name, float(k[name]))
    for name in INT_KEYS:
        setattr(P, name, int(k[name]))
    # supply. Written straight onto monomers_per_point because that is the single field
    # MONOMER_CONC feeds; monomers_per_seg (demand) is untouched, as it must be.
    P.monomers_per_point = (BASE_MPP if k["phi_max"] is None
                            else float(k["phi_max"]) * float(P.monomers_per_seg))
    return k


def paste(sheet):
    """Occupied lattice points of mesh + nematic on the canvas. Cortex excluded, as in
    every earlier round -- the cortex is a cell wall, not a myofibril, and the traces
    do not annotate it."""
    canvas = np.zeros((mc.CANVAS_H, mc.CANVAS_W), bool)
    for sim, cx, cy in sheet.cells:
        for fam in ("mesh", "nematic"):
            for f in sim.fibres[fam]:
                if not f.occupied_pts:
                    continue
                pts = np.array(list(f.occupied_pts), int)
                X = pts[:, 0] - sheet.hcx + cx
                Y = pts[:, 1] - sheet.hcy + cy
                ok = (X >= 0) & (X < mc.CANVAS_W) & (Y >= 0) & (Y < mc.CANVAS_H)
                canvas[Y[ok], X[ok]] = True
    return canvas


def field(sheet):
    """The measured field: model crop -> the shared normalise()."""
    r0, r1, c0, c1 = sheet.crop_bounds()
    return M.normalise(paste(sheet)[r0:r1, c0:c1], scale_in=LU)


def raw_occupancy(sheet):
    """sim-side phi: total fibre mass / (mask points x monomers_per_seg).
    NOT comparable with the post-pipeline line density -- handover S11.4."""
    mass = pts = 0.0
    for sim, _, _ in sheet.cells:
        mass += sum(f.mass for fam in ("mesh", "nematic", "cortex")
                    for f in sim.fibres[fam])
        pts += float(sim.mask.sum())
    return mass / (pts * P.monomers_per_seg) if pts else np.nan


def run_once(overrides=None, seed=42, hpf=HPF, verbose=False, keep_fields=False):
    configure(overrides)
    sheet = mc.MultiCell(base_seed=seed)
    steps = int(P.steps)
    marks = {int(round(steps * (h - 32.0) / P.total_hours)): h for h in np.asarray(hpf, float)}
    out, fields = {}, {}
    steps = max(marks)                     # never step past the last requested timepoint
    for step in range(steps + 1):
        if step in marks:
            h = marks[step]
            m = field(sheet)
            r = M.measure_all(m)
            r["phi"] = raw_occupancy(sheet)
            r["n_nem"] = sum(len(s.fibres["nematic"]) for s, _, _ in sheet.cells)
            r["n_mesh"] = sum(len(s.fibres["mesh"]) for s, _, _ in sheet.cells)
            out[h] = r
            if keep_fields:
                fields[h] = m
            if verbose:
                print(f"    {h:.0f} hpf  p {r['order']:.3f}  dens {r['density']:.4f}  "
                      f"p95 {r['gap_p95']:5.2f}  foam {r['foam']:.2f}  n {r['count']:3d}  "
                      f"phi {r['phi']:.3f}  nem {r['n_nem']:3d} mesh {r['n_mesh']:4d}",
                      flush=True)
        if step < steps:
            sheet.step()
    drift = 0.0
    for sim, _, _ in sheet.cells:
        stored = sum(f.mass for fam in sim.fibres.values() for f in fam)
        tot = sim.field.pool.sum() + stored
        drift = max(drift, abs(tot - sim.field.total_init) / sim.field.total_init * 100)
    for h in out:
        out[h]["drift_pct"] = drift
    return (out, fields) if keep_fields else out


def as_arrays(out, keys=("density", "order", "gap_p95", "gap_p90", "foam", "count",
                         "coverage", "phi", "n_nem", "n_mesh", "drift_pct")):
    return {k: np.array([out[h][k] for h in HPF], float) for k in keys}
