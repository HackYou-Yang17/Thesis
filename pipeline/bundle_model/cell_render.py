"""
CARMA — single-cell render layer (PARTICLE model).
Per-frame state capture (label image, brightness/intensity maps, junctions) +
free-monomer particle sampling. No physics here: reads a Simulation / MonomerField
from cell_particle.py and returns arrays for plotting/animation.

Ported from the mean-field cell_render to the cell_particle API. Key differences,
all forced by the particle representation (a fibre is a set of pixels, not an
object with length/width/maturity):

  * A fibre has NO .length / .width / .mature / .monomers_stored. Instead it carries
    `occupied_pts` (live filament pixels) and `lattice_pts` (its potential envelope).
  * DRAW `occupied_pts`, not `lattice_pts` — only occupied pixels are real filament.
    (The mean-field renderer drew the full lattice band because every fibre was solid.)
  * Brightness is OCCUPANCY-driven: `fibre_fill` = occupied / potential envelope, so a
    sparsely-grown or partly-depolymerised fibre reads dim and a filled one reads bright,
    reproducing the old width/monomer look without needing width or stored-monomer state.
  * Orientation binning uses the module-level is_primary(angle); families come from the
    sim.fibres dict key exactly as in the mean-field renderer.
  * MonomerField has only `pool` (released monomers go straight back into the pool), so
    there is no separate `freed` population and no monomer-transit dots.

The returned dict keys are IDENTICAL to the mean-field cell_render.record_state, so any
downstream consumer (multicell_render, analysis) works unchanged.
"""

import numpy as np
from bundle_model.cell_particle import P, is_primary


# ── BRIGHTNESS CONSTANTS ────────────────────────────────────────────────────────
# fibre_brightness() reads these four. They were not defined anywhere -- Params has no
# bright_mature / bright_growing_min / mature_ref_length / bright_monomer_gamma -- so this
# module could not import and run as shipped. They live HERE rather than in Params because
# they are a greyscale look-up and nothing else: no rate, no geometry, no physical quantity
# reads them, and changing one cannot move a fitted number. Keeping them out of Params also
# keeps the tuned parameter file free of values that only affect how a frame looks.
BRIGHT_MATURE = 1.0        # ceiling for seeded-full fibres (nematic, cortex)
BRIGHT_GROWING_MIN = 0.35  # floor of the mesh ramp: a just-nucleated fibre's ceiling
MATURE_REF_LENGTH = 40.0   # occupied PIXELS at which a mesh fibre reaches the mature ceiling.
                           # Particle fibres have no .length, so pixel count is the extent
                           # proxy. Scale this with the cell: 40 px suits the 34.7 x 12.5 um
                           # cell (at fibre_width 4 that is ~10 lu of centre line); the
                           # 93.7 x 39.8 um cell used 100. Set it longer than a whole fibre
                           # and every mesh fibre renders at the floor.
BRIGHT_MONOMER_GAMMA = 1.0 # >1 darkens partially-filled fibres faster; 1.0 = linear in fill


# ── Per-fibre brightness (single source of truth) ───────────────────────────────


def fibre_fill(f):
    """Occupancy fraction of a fibre in [0, 1]: occupied pixels / potential-envelope pixels.

    In the particle model a fibre's `lattice_pts` is its full potential band (centre line
    expanded to fibre_width and clipped to the cell); `occupied_pts` is what has actually
    polymerised. Their ratio is how "filled" the fibre is — the particle analogue of the
    mean-field width/max_width. A freshly-seeded fibre (one pixel) is near 0; a fully grown,
    full-width fibre is near 1; depolymerisation lowers it. Guards against an empty envelope.
    """
    cap = len(f.lattice_pts)
    if cap <= 0:
        return 0.0
    return float(min(1.0, max(0.0, len(f.occupied_pts) / cap)))


def fibre_brightness(f, family):
    """Greyscale brightness in [0.08, 1.0] for a fibre in the given family.

    Family sets the ceiling (`base`); occupancy sets how much of that ceiling the fibre
    reaches (`fill`). Growing mesh fibres additionally ramp their ceiling with how far they
    have extended (occupied-pixel count toward mature_ref_length) so freshly-nucleated
    fibres start faint — mirroring the mean-field renderer's length ramp, but using pixel
    count as the length proxy (particle fibres have no .length).
    """
    is_nematic = family == "nematic"
    is_cortex = family == "cortex"
    # Particle fibres have no .mature flag; treat nematic/cortex (seeded full) as "mature",
    # and ramp mesh fibres by extent.
    if is_nematic or is_cortex:
        base = BRIGHT_MATURE
    else:
        extent = len(f.occupied_pts)
        prog = min(1.0, extent / max(1.0, MATURE_REF_LENGTH))
        base = (BRIGHT_GROWING_MIN + (BRIGHT_MATURE - BRIGHT_GROWING_MIN) * prog) * 0.9

    fill = fibre_fill(f)
    gamma = BRIGHT_MONOMER_GAMMA   # >1 darkens partials faster
    if gamma != 1.0:
        fill = fill ** gamma
    return float(max(0.08, min(1.0, base * fill)))


# ── Free-monomer particle sampling ─────────────────────────────────────────────


def particle_positions(field, quantum, rng=None):
    """Discrete particle positions for the resting G-actin pool (coarse sub-grid sample).
    The particle model keeps released monomers in `pool`, so there is no separate freed set."""
    step = P.pool_grid_step
    sub = field.pool[::step, ::step]
    rr, cc = np.nonzero(sub >= P.monomers_per_point * 0.4)
    pool_cols = (cc * step).astype(float)
    pool_rows = (rr * step).astype(float)
    return {
        "pool": (pool_cols, pool_rows),
        "freed": (np.empty(0), np.empty(0)),
    }


# ── Per-frame state capture ─────────────────────────────────────────────────────


def record_state(sim, rng=None):
    """Capture a lightweight snapshot of `sim` for later animation.
    Returns dict with labels (uint8), intensity, mesh/static/contractile intensity, junction,
    and particle positions. Draws OCCUPIED pixels (real filament) per family."""
    parts = particle_positions(sim.field, P.monomer_quantum, rng)
    labels = np.zeros((P.H, P.W), dtype=np.uint8)
    intensity = np.zeros((P.H, P.W), dtype=np.float32)
    mesh_intensity = np.zeros((P.H, P.W), dtype=np.float32)
    static_intensity = np.zeros((P.H, P.W), dtype=np.float32)
    contractile_intensity = np.zeros((P.H, P.W), dtype=np.float32)
    h_cov = np.zeros((P.H, P.W), dtype=np.float32)
    v_cov = np.zeros((P.H, P.W), dtype=np.float32)

    # Iterate families explicitly (the model tracks family by dict key, not by fibre flags).
    for family, flist in sim.fibres.items():
        if family == "dead":
            continue
        is_nematic = family == "nematic"
        is_cortex = family == "cortex"
        is_mesh = family == "mesh"
        for f in flist:
            if is_cortex:
                lab = 5
            elif is_nematic:
                lab = 1
            else:
                # particle mesh fibre: "mature" proxy = has grown past the reference extent
                lab = 2 if len(f.occupied_pts) >= MATURE_REF_LENGTH else 3

            b = fibre_brightness(f, family)
            is_h = is_primary(f.angle)

            for c, r in f.occupied_pts:                 # DRAW occupied pixels, not the envelope
                if 0 <= c < P.W and 0 <= r < P.H:
                    labels[r, c] = lab
                    if b > intensity[r, c]:
                        intensity[r, c] = b
                    if not is_cortex:
                        if b > contractile_intensity[r, c]:
                            contractile_intensity[r, c] = b
                    if is_mesh:
                        if b > mesh_intensity[r, c]:
                            mesh_intensity[r, c] = b
                        if is_h:
                            if b > h_cov[r, c]:
                                h_cov[r, c] = b
                        else:
                            if b > v_cov[r, c]:
                                v_cov[r, c] = b
                    else:
                        if b > static_intensity[r, c]:
                            static_intensity[r, c] = b

    junction = h_cov * v_cov
    return {
        "p_pool": parts["pool"],
        "p_freed": parts["freed"],
        "p_transit": (np.empty(0), np.empty(0)),
        "p_transit_cortex": (np.empty(0), np.empty(0)),
        "labels": labels,
        "intensity": intensity,
        "mesh_intensity": mesh_intensity,
        "static_intensity": static_intensity,
        "contractile_intensity": contractile_intensity,
        "junction": junction,
    }
