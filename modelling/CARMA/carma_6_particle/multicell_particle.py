"""
CARMA — multicell physics module (PARTICLE model).
A confluent sheet of independent single-cell particle Simulations (cell_particle.py):
tissue tiling, neighbour adjacency, per-cell stepping, cross-cell cadherin nucleation.
No rendering, no run loop, no diagnostics here.

Design constraints for this port:
  * The model (cell_particle.py) is NOT reimplemented here. Cadherin nucleation calls the
    model's OWN Simulation.nucleate(col, row, angle) — the single nucleation code path —
    with a directed contact point. Everything cadherin needs to COMPUTE that point (canvas
    tiling, neighbour masks, local<->canvas transforms, axis conversion) is owned and stored
    HERE in MultiCell, not in the model.
  * nucleate() derives its own direction from (col, row) — the cardinal pointing toward the
    receiving cell's centre — and adds that cell's axis internally (angle + P.mesh_axis). So
    cadherin passes ONLY the contact point; no absolute angle crosses the cell boundary, which
    removes the cross-cell axis-double-application hazard by construction.

FLAG LEGEND:
  [BIO]    cell biology / actin physics — the actual model
  [SCAF]   scaffolding: config / bookkeeping that only exists to make the model run
  [MM]     "middle-man": coordinate transforms, global-state juggling, geometric searches
"""

import numpy as np
from scipy import ndimage
from modelling.CARMA.carma_6_particle.cell_particle import P, Simulation, is_primary

CROP_SIZE = 170      # [SCAF] final 1:1 crop side (lu)
CROP_MARGIN = 30     # [SCAF] overhang cells kept beyond the crop
N_CELLS_TARGET = None  # [SCAF] None = AUTOMATIC: count follows from cell size + crop size.
                       #        Set an int to force exactly that many (N nearest the canvas centre).

# ── CROP PLACEMENT ──────────────────────────────────────────────────────────────
# The crop was previously hard-centred on the canvas. With the current tiling a cell
# centre lands exactly on the canvas centre, so the centred crop is dominated by ONE
# cell interior and only clips the two horizontal membranes above and below it. The
# traced images are crops of a confluent sheet, not of a single cell, so the crop is
# now PLACED rather than assumed: it slides within the canvas to the window that sees
# the most cells. CROP_SIZE is unchanged — only the window's origin moves.
CROP_OFFSET = None       # [SCAF] (dcol, drow) in lu from the canvas centre.
                         #        None = AUTOMATIC (MultiCell._fit_crop_offset).
                         #        Set a tuple to force a placement.
CROP_SLIDE = 0           # [SCAF] how far (lu) the crop is allowed to move off the canvas centre.
                         #        The canvas and the simulated sheet grow by this on every side so
                         #        there is real tissue under every candidate window; CROP_SIZE is
                         #        untouched. 0 restores the original hard-centred behaviour.
                         #        Was 90 for the 93.7 x 39.8 um cells, where one cell was longer
                         #        than the crop and the window had to travel a whole column period to
                         #        find a vertex. At 34.7 x 12.5 um a cell is well inside the 170 lu
                         #        crop, so even the centred window spans many cells and the slide is
                         #        no longer needed: back to 0, which keeps the canvas at 230 and the
                         #        simulated sheet at 17 cells instead of 57.
CROP_MIN_CELL_FRAC = 0.02  # [SCAF] a cell counts as "seen" once it fills this fraction of the crop
                           #        (2% of 170x170 = 578 lu^2, ~76 um^2 — a real wedge, not a corner
                           #        pixel or an anti-aliased membrane sliver).


# ── CELL TESSELLATION ───────────────────────────────────────────────────────────

def cell_centres(crop=CROP_SIZE, margin=CROP_MARGIN, n_cells=N_CELLS_TARGET, slide=CROP_SLIDE):
    """[BIO] Confluent elongated-hexagon tiling: staggered columns, pointed ends interlocking,
    odd columns shifted down by B. [SCAF] Over-covers then prunes so the central crop is full.
    If n_cells is given, keep exactly that many cells — the ones nearest the canvas centre — so the
    sheet is a fixed-size central cluster (deterministic count, independent of cell/crop dimensions).

    `slide` grows the canvas and the keep-rectangle by that many lu on every side, so the crop can be
    PLACED off-centre (see CROP_SLIDE / MultiCell._fit_crop_offset). It does not change the crop size,
    the tiling, or any cell; it only means more of the sheet is simulated so there is confluent tissue
    for the crop to move onto. slide=0 reproduces the original behaviour exactly."""
    A = float(P.hex_half_w)            # [BIO] cell half-length
    B = float(P.hex_half_h)            # [BIO] cell half-width
    E = float(P.hex_end_frac) * A      # [BIO] pointed-end inset
    DX = 2 * A - E                     # [BIO] column period (ends interlock)
    DY = 2 * B                         # [BIO] row period within a column

    canvas = crop + 2 * (margin + slide)             # [SCAF] slide widens the simulated sheet only
    cx0 = cy0 = canvas / 2.0                          # [SCAF]
    n_cols = int(canvas / DX) + 3                     # [SCAF] over-cover margin
    n_rows = int(canvas / DY) + 3                     # [SCAF]

    centres = []
    for ci in range(-n_cols, n_cols + 1):
        x = cx0 + ci * DX
        stagger = B if (ci % 2 != 0) else 0.0         # [BIO] alternate-column half-shift
        for ri in range(-n_rows, n_rows + 1):
            y = cy0 + ri * DY + stagger
            centres.append((x, y))
    # [SCAF] Keep ONLY cells whose body intersects the region the crop can REACH: the centred 1:1
    #        crop rectangle grown by `slide` on every side. At slide=0 that is the crop itself, i.e.
    #        the original rule.
    half = crop // 2
    lo_x = canvas / 2.0 - half - slide; hi_x = lo_x + crop + 2 * slide
    lo_y = canvas / 2.0 - half - slide; hi_y = lo_y + crop + 2 * slide
    kept = [(x, y) for (x, y) in centres
            if (x - A < hi_x and x + A > lo_x and y - B < hi_y and y + B > lo_y)]
    if n_cells is not None and len(kept) > n_cells:
        # [SCAF] reduce to exactly n_cells: keep those nearest the canvas centre (a central cluster)
        kept.sort(key=lambda p: (p[0] - canvas / 2.0) ** 2 + (p[1] - canvas / 2.0) ** 2)
        kept = kept[:n_cells]
    return kept, canvas


CENTRES, _CANVAS = cell_centres()                       # [SCAF]
N_CELLS = len(CENTRES)
CANVAS_W = CANVAS_H = int(round(_CANVAS))
CROP_SIZE = min(CROP_SIZE, CANVAS_W, CANVAS_H)          # [SCAF]


def n_cells_for(crop, margin=CROP_MARGIN, slide=CROP_SLIDE):
    """[SCAF] How many cells a given crop needs, for the CURRENT P.hex_half_w/h. Pure geometry
    (no Simulation is built), so it is cheap to call when choosing a crop size."""
    return len(cell_centres(crop=crop, margin=margin, n_cells=None, slide=slide)[0])


def crop_for_n_cells(target, margin=CROP_MARGIN, lo=80, hi=2000):
    """[SCAF] Inverse of n_cells_for: the largest crop whose automatic cell count stays <= target.
    Use when you want 'about N cells' but still want the sheet to fill the crop (full confluence).
    Returns (crop, n_cells_at_that_crop)."""
    best = (lo, n_cells_for(lo, margin))
    for crop in range(lo, hi + 1, 2):
        n = n_cells_for(crop, margin)
        if n <= target:
            best = (crop, n)
        else:
            break
    return best



# ── MULTICELL ───────────────────────────────────────────────────────────────────

class MultiCell:
    """[BIO] Confluent sheet of independent cardiomyocyte particle Simulations with cadherin coupling."""
    def __init__(self, base_seed=42, cadherin_every=200):
        self.hcx = P.W // 2                  # [MM] local cell-frame centre (for local<->canvas transforms)
        self.hcy = P.H // 2                  # [MM]
        self.cells = []                      # [BIO] (Simulation, canvas_col, canvas_row) per cell
        self.axes = []                       # [MM] per-cell mesh axis, stored HERE (model reads P.mesh_axis live)
        self.cadherin_every = cadherin_every # [BIO] cadherin coupling evaluated every N steps
        self._iter = 0                       # [SCAF] step counter (drives the cadherin cadence)

        for i, (cx, cy) in enumerate(CENTRES):
            np.random.seed(base_seed + 7919 * i)   # [SCAF] independent fibre seeding per cell
            # FIXED: was the literal np.random.uniform(-np.pi/12, np.pi/12). parameters.py states
            # axis_spread "was hard-coded as a literal inside multicell_particle.MultiCell.__init__
            # and now a Params field so there is one source of truth" -- but the literal was never
            # replaced, so the tuned pi/8 was not reaching the sheet and every run used pi/12.
            axis = np.random.uniform(-P.axis_spread, P.axis_spread)  # [BIO] this cell's own mesh axis
            P.mesh_axis = axis                     # [MM] Simulation.__init__ captures self.mesh_axis = P.mesh_axis
            sim = Simulation()
            self.cells.append((sim, int(round(cx)), int(round(cy))))
            self.axes.append(axis)
        self._neighbours = self._find_neighbours()  # [BIO] which cells are adjacent (cadherin contacts)
        self.crop_size = self._fit_crop()           # [SCAF] largest centred crop these cells fully cover
        self.crop_offset = (tuple(CROP_OFFSET) if CROP_OFFSET is not None
                            else self._fit_crop_offset())   # [SCAF] where that crop sits on the canvas

    def _fit_crop_offset(self, size=None, min_frac=CROP_MIN_CELL_FRAC):
        """[SCAF] Slide the (fixed-size) crop over the canvas and return the (dcol, drow) offset from
        the canvas centre whose window sees the MOST distinct cells.

        Why this exists: the tiling puts a cell centre on the canvas centre, so the centred crop is
        one cell's interior plus two clipped membranes. The traced images are windows onto a
        confluent sheet — mostly partial cells meeting at membranes and tricellular vertices — so the
        crop needs to be placed there instead. Partial cells are the POINT; nothing here asks for a
        whole cell.

        Constraints kept: crop side is untouched (self.crop_size), the window stays inside the
        canvas, and it must be FULLY covered by the sheet so no black non-tissue wedge enters the
        frame. Coverage is tested on the closed union for the same reason _fit_crop does it — a 1 lu
        rounding seam along a shared membrane is a tiling artifact, not a hole.

        Ranking: (number of cells covering at least `min_frac` of the crop, then the most even split,
        i.e. smallest single-cell share). The second term is what stops it settling for "one big cell
        plus n slivers" when a genuine junction is reachable.
        """
        size = self.crop_size if size is None else size
        half = size // 2
        union = np.zeros((CANVAS_H, CANVAS_W), bool)
        for m in self._cell_masks:
            union |= m
        covered = ndimage.binary_closing(union, np.ones((3, 3), bool))

        def integral(a):                       # [MM] summed-area table -> O(1) window counts
            return np.pad(np.cumsum(np.cumsum(a.astype(np.int64), 0), 1), ((1, 0), (1, 0)))

        def window(ii, r0, c0):
            return int(ii[r0 + size, c0 + size] - ii[r0, c0 + size] - ii[r0 + size, c0] + ii[r0, c0])

        gaps_ii = integral(~covered)
        cell_ii = [integral(m) for m in self._cell_masks]
        area = float(size * size)
        max_dc = (CANVAS_W - size) // 2
        max_dr = (CANVAS_H - size) // 2

        best, best_key = (0, 0), None
        for dr in range(-max_dr, max_dr + 1):
            r0 = CANVAS_H // 2 - half + dr
            for dc in range(-max_dc, max_dc + 1):
                c0 = CANVAS_W // 2 - half + dc
                if window(gaps_ii, r0, c0):                       # [SCAF] any uncovered pixel -> reject
                    continue
                fracs = sorted((window(ii, r0, c0) / area for ii in cell_ii), reverse=True)
                seen = [f for f in fracs if f >= min_frac]
                key = (len(seen), -fracs[0])                      # more cells, then most even split
                if best_key is None or key > best_key:
                    best_key, best = key, (dc, dr)
        return best

    def crop_bounds(self, size=None, offset=None):
        """[SCAF] (r0, r1, c0, c1) of the crop on the canvas, clipped to stay in bounds.
        Single source of truth for where the render cuts; multicell_render calls this."""
        size = self.crop_size if size is None else size
        dc, dr = self.crop_offset if offset is None else offset
        half = size // 2
        r0 = int(np.clip(CANVAS_H // 2 - half + dr, 0, CANVAS_H - size))
        c0 = int(np.clip(CANVAS_W // 2 - half + dc, 0, CANVAS_W - size))
        return r0, r0 + size, c0, c0 + size

    def crop_cell_fractions(self, size=None, offset=None):
        """[SCAF] Fraction of the crop occupied by each cell — how the placement is reported/checked."""
        r0, r1, c0, c1 = self.crop_bounds(size, offset)
        area = float((r1 - r0) * (c1 - c0))
        return sorted((m[r0:r1, c0:c1].sum() / area for m in self._cell_masks), reverse=True)

    def _fit_crop(self, cap=CROP_SIZE):
        """[SCAF] Largest centred square fully covered by the cell sheet (no black gaps in the render).
        With a reduced cell count the sheet no longer fills the nominal CROP_SIZE, so the crop is
        shrunk to the confluent region. Recomputed from the actual masks, so it stays correct for any
        cell size / count."""
        union = np.zeros((CANVAS_H, CANVAS_W), bool)
        for m in self._cell_masks:
            union |= m
        # [MM] Close hairline seams: adjacent cells can leave a 1px rounding gap along a shared
        #      membrane. That is a tiling artifact (the PSF blur renders it as a cell wall), not a
        #      real hole, so dilate before testing or _fit_crop would shrink the crop drastically.
        union = ndimage.binary_closing(union, np.ones((3, 3), bool))
        best = 0
        for size in range(40, min(cap, CANVAS_W, CANVAS_H) + 1, 2):
            h = size // 2
            r0, c0 = CANVAS_H // 2 - h, CANVAS_W // 2 - h
            if r0 < 0 or c0 < 0:
                break
            if union[r0:r0 + size, c0:c0 + size].all():
                best = size
            else:
                break
        return best or min(cap, CANVAS_W, CANVAS_H)

    def _find_neighbours(self):
        """[BIO] Adjacency between cells that share a membrane. [MM] via pasted per-cell canvas masks + dilation."""
        masks = []
        for sim, cx, cy in self.cells:
            m = np.zeros((CANVAS_H, CANVAS_W), bool)          # [MM] paste this cell onto the canvas
            ys, xs = np.nonzero(sim.mask)
            CX = xs - self.hcx + cx
            CY = ys - self.hcy + cy
            ok = (CX >= 0) & (CX < CANVAS_W) & (CY >= 0) & (CY < CANVAS_H)
            m[CY[ok], CX[ok]] = True
            masks.append(m)
        self._cell_masks = masks                              # [MM] reused by the renderer for walls/haze
        self._cell_masks_dil = [ndimage.binary_dilation(m, iterations=4) for m in masks]  # [MM]
        nb = {i: [] for i in range(len(self.cells))}
        for i in range(len(self.cells)):
            di = ndimage.binary_dilation(masks[i], iterations=3)   # [MM]
            for j in range(len(self.cells)):
                if i == j:
                    continue
                if (di & masks[j]).sum() > 5:                 # [BIO] cells touch -> neighbours
                    nb[i].append(j)
        return nb

    def step(self):
        """[BIO] Advance every cell's KMC by one step, then evaluate cadherin coupling on cadence."""
        for i, (sim, _, _) in enumerate(self.cells):
            P.mesh_axis = self.axes[i]     # [MM] is_primary() reads the global -> use this cell's axis
            sim.kmc_steps()
        if self._iter % self.cadherin_every == 0:        # [BIO] cadherin contacts checked periodically
            self._cadherin_check()
        self._iter += 1

    def _cadherin_check(self):
        """[BIO] A grown mesh fibre whose leading pixel reaches a shared membrane nucleates a
        superfibre in the neighbour cell — via Simulation.cadherin(), which builds the fibre on
        the SOURCE cell's axis so it threads the junction carrying the neighbour's orientation."""
        for i, (sim_i, cxi, cyi) in enumerate(self.cells):
            nbrs = self._neighbours.get(i, [])
            if not nbrs:
                continue
            for f in list(sim_i.fibres['mesh']):
                if not f.occupied_pts:                        # [BIO] skip inert (never-seeded) fibres
                    continue
                # [MM] particle fibres have no .tip -> leading occupied pixel along the fibre axis
                tcol, trow = max(f.occupied_pts,
                                 key=lambda p: p[0] * np.cos(f.angle) + p[1] * np.sin(f.angle))
                tx = tcol - self.hcx + cxi                    # [MM] local -> canvas
                ty = trow - self.hcy + cyi                    # [MM]
                if not (0 <= tx < CANVAS_W and 0 <= ty < CANVAS_H):
                    continue
                for j in nbrs:
                    sim_j, cxj, cyj = self.cells[j]
                    if not self._cell_masks_dil[j][ty, tx]:   # [BIO] tip actually reaches cell j's membrane
                        continue
                    if np.random.random() > P.cadherin_nucleation_prob:  # [BIO] per-attempt cadherin chance
                        continue
                    lc = tx - cxj + self.hcx                  # [MM] canvas -> cell-j local
                    lr = ty - cyj + self.hcy                  # [MM]
                    pt = self._nearest_inside(sim_j, lc, lr)  # [MM] snap to nearest interior pixel
                    if pt is None:
                        continue
                    # [BIO] Superfibre reflects the SOURCE cell i's axis + fibre direction, threading the
                    #       junction with the neighbour's orientation rather than snapping to cell j's grid.
                    #       cadherin() sets/restores P.mesh_axis internally.
                    sim_j.cadherin(pt[0], pt[1], f.angle, mesh_noise=self.axes[i])

    @staticmethod
    def _nearest_inside(sim, lc, lr, R=6):
        """[MM] Nearest interior lattice point to (lc,lr) within radius R (snaps a wall hit to a valid seed site)."""
        best, bd = None, 1e9
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                c, r = lc + dc, lr + dr
                if 0 <= c < P.W and 0 <= r < P.H and sim.mask[r, c]:
                    d = dc * dc + dr * dr
                    if d < bd:
                        bd, best = d, (c, r)
        return best
