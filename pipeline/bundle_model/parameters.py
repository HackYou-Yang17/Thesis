import numpy as np
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# v13 — THE TUNE, RETUNED ON THE CORRECTED NUCLEATION RULE.  26 Aug 2026.
# Fitted against the hand-traced curves (18 hearts, 3 per age, 32-52 hpf).  Model and
# images measured through the IDENTICAL pipeline: skeletonise -> 0.40 um/px ->
# re-skeletonise -> 1.2 um width -> 61.4 um crop -> +-45 deg whole-crop band statistic
# with a surrogate-matched null.
#
#   objective   L_dgo = 2*X_density + 1.5*X_gap + 1*X_order
#               X(obs) = sqrt(mean_ages z^2),  z = (model - traced)/sd_hearts,
#               sd MODERATED (empirical Bayes, df0 = df = 2) because every traced sd
#               comes from three hearts and carries ~52 % relative uncertainty.
#   NOT in the loss, and therefore reportable as diagnostics: foam_ge8, fragmentation,
#   segment count, and the crossover.
#
# WHY THERE IS A v13 AT ALL.  cell_particle._seed_angle used to snap a cortical site's
# firing direction to whichever of four LATTICE directions pointed nearest the cell
# CENTROID.  On the flat membranes the short axis wins that contest wherever
# |col - cx| < hex_half_h, so those sites were assigned a short-axis direction, failed the
# is_primary filter and never nucleated -- a dead band one CELL HEIGHT wide with no
# biological content.  The rule is now "along the long axis, into the cell", which is the
# assumption the old rule was trying to express.  159 -> 222 nucleation-competent sites,
# and every site the old rule kept fires in exactly the same direction as before.
#   NOT the local membrane normal, which was the obvious fix and is wrong: the taper
#   e = hex_end_frac*hex_half_w = 18.9 lu exceeds hex_half_h = 17.3 lu, so EVERY membrane
#   segment's inward normal points more across the short axis than along it and a
#   normal-based rule leaves ZERO primary sites.  In an elongated cell cortical fibres run
#   ALONG the long axis, not perpendicular to the membrane they start from.
#
# WHAT THE FIX COST, AND WHAT RETUNING RECOVERED.  Ten held-out seeds (7501-7510), all
# four arms on the SAME seeds, paired:
#       v12 rates on the OLD code          L_dgo 3.596
#       v13 rates on the FIXED code        L_dgo 3.632   (+0.014 vs v12/old, p = 0.85)
#       v12 rates on the FIXED code        L_dgo 4.079   (+0.453 vs v12/old, p = 0.0025)
#   So the artefact was worth 0.45 while the rates were held at v12, and retuning gives
#   ALL of it back.  The dead band was not doing anything the rates cannot do; it was
#   simply a constraint the old rates had been fitted around.
#
#   fit, 30 seeds:  L_dgo 3.654   X_density 0.613   X_gap 1.207   X_order 0.619
#                   X_frag 1.697   X_count 0.833
#   HELD-OUT ten seeds that took no part in selection: L_dgo 3.632
#   QUOTE THE HELD-OUT VALUE.  A tune is optimistic by +0.34 to +0.53 on the seeds that
#   selected it; the round-2 winner measured 3.551 at its twenty selection seeds and 3.895
#   on ten it had never seen, a curse of +0.344.
#
# HOW THIS TUNE WAS CHOSEN, because the loss did not choose it.  Five finalists sat within
# 0.27 at twenty seeds and within 0.26 held out, against resolutions of 0.25 and 0.36 --
# a five-way tie.  The tie was broken on two standing project criteria, not on the loss:
#   (i)  NOT PINNED AT A BOUND.  Two finalists sat exactly at angle_noise = pi/4, the
#        search's ceiling.  A parameter at its bound is a statement about the box, not
#        about the data.
#   (ii) CROSSOVER PRECISION, the seed-count-invariant replacement for the retired
#        "range < 4 h" rule.  Among the interior finalists this one has the tightest and
#        most accurate crossing: 41.66 +- 2.30 h over 30 seeds against a traced
#        40.81 +- 1.00, i.e. 0.85 heart-SD, crossing on 30 of 30 seeds.
#
# THE CROSSOVER IS NOT A PREDICTION.  X_order fits the dominance curve at all six
# timepoints and the crossover is just where that fitted curve passes 0.5, so fitting the
# curve fits the crossing.  Report it as a derived feature of a calibration target.
#
# THE RESOLUTION, measured natively at this tune from 15 configuration pairs over 20
# shared seeds: the paired SD of a difference in L_dgo is 0.472, so the smallest
# detectable difference is 0.76 at 3 seeds, 0.54 at 6, 0.42 at 10 and 0.30 at 20.  The
# 0.401 / 0.46 quoted in the v12 files was measured on the OLD code from three
# configurations and should not be used here.
#
# WHAT IS AND IS NOT IDENTIFIED (a +-40 % sensitivity sweep at this tune, crossed with how
# far the FIVE STATISTICALLY TIED finalists disagree; agreement line = half the probe box,
# 1.53x).  The v12 sweep and multi-start were run on the old nucleation rule and DO NOT
# TRANSFER -- a sensitivity ranking is a property of the point AND the model.
#   FITTED (moves the fit AND agreed on by tunes that fit equally well):
#       thin:grow (sensitivity 1.83, tied set spans 1.24x)
#       angle_noise (0.99, 1.10x)
#   SENSITIVE BUT NOT IDENTIFIED -- report as one point on a degenerate manifold:
#       rate_nematic_depoly (0.91, 1.62x)
#       nematic_thresh (0.55, MARGINAL against the 0.54 floor; and the tied set holds both
#                       0.20 and 0.35, which is disagreement in the only sense a two-level
#                       gate has)
#   INVISIBLE TO THE FIT -- all below the 0.54 resolution, so agreement between searches is
#   uninformative and these must NOT be quoted as fitted values:
#       rate_grow (0.40)  rate_nematic_poly (0.33)  axis_spread (0.23)  rate_branch (0.13)
#       cadherin_nucleation_prob (0.10)  rate_nucleate (0.08)
#   NOTE the swap against v12, which identified angle_noise and nematic_thresh: at v13 it is
#   angle_noise and THIN:GROW, and nematic_thresh has fallen to marginal.  The ORDER of the
#   sensitivity ranking is unchanged; what moved is which parameters the tied set agrees on.
#
# THE MULTI-START DOES NOT CONTRIBUTE TO THE ABOVE, and that is itself a result.  Six
# maximin-spread independent starts, one coarse +-40 % probe each (12 neighbours, 3 seeds),
# basin minima re-scored at 12 seeds: 4.88, 5.43, 5.51, 5.79, 6.11, 6.82 against this
# tune's 3.63.  NONE of them reaches it, so there is no tied pair to measure disagreement
# on and the finalist set is used instead.  Do NOT read this as "the optimum is isolated":
# the probe was trimmed from the v11 run's budget (16 -> 12 neighbours, 4 -> 3 probe seeds,
# 20 -> 12 re-scoring seeds) and every one of those cuts makes a start LESS likely to
# descend into the basin.  The honest statement is that no independent start found it at
# this budget.
#
# phi_max = monomers_per_point / monomers_per_seg = 1.0901.
# ─────────────────────────────────────────────────────────────────────────────

# find pixel size to fibre length (nanometres)
#       recrop reconfig stopped heart images and find a larger images for 40hpf. recrop to exactly sim size
#       sim pixels = 170x170
# find intensity to actin concentration (DONE)
# Used heart cell images and measured the greatest distances for width and length of cells
# Then took an average of the measurements (remember to calculate S.D. and error bars)
CELL_LENGTH = 38.9000 # μm from cell boundary images (re-measured 23 Aug 2026; was 34.7)
CELL_HEIGHT = 12.5000 # μm

UM_PER_LATTICE = 0.361 # μm/lu (re-measured 23 Aug 2026; was 0.362)
MONOMER_CONC = 300.0 # μM (per cell total)
FIAMENT_LENGTH_PER_MONOMER = 2.70 # nm (https://pmc.ncbi.nlm.nih.gov/articles/PMC3130349/)
FILAMENT_WIDTH = 7 # nm (Dominguez & Holmes 2011 Annu Rev Biophys)
FIBRE_BUNDLE_WIDTH = 1.23 # μm (re-measured 23 Aug 2026; was 1.27). NOTE fibre_width = round(W/UM_PER_LATTICE)
                          # flips from 4 lu to 3 lu at W = 1.267 μm, so 1.27 sat 0.2% above a rounding
                          # boundary and 1.23 sits 2.7% below it. A 3% change in the measured value is a
                          # 25% change in the modelled fibre cross-section -- quote it with its error bar.
MESH_SPACING = 2.80 # μm (from image pixel analysis)
NEMATIC_GAP_MIN = 2.60 # μm (from image pixel analysis)
NEMATIC_GAP_MAX = 4.04 # μm

# ── Actin budget ─────────────────────────────────────────────────────────────
# supply depth = cell depth; demand depth = bundle depth (bundle taken as circular in section)
CELL_THICKNESS = 3.0 # μm (one-cell-thick myocardium at 32-52 hpf; ASSUMPTION, not a measured value)
MYOFIBRIL_DEPTH = FIBRE_BUNDLE_WIDTH # μm
MYOSIN_SPACING = 46e-3 # μm, a = 2*d10/sqrt(3) with d10 ~ 40 nm (Millman 1998; Irving 2000)
ACTIN_PER_MYOSIN = 2.0 # thin filaments sit on the trigonal points: 6 neighbours shared 3 ways
# mature sarcomeric packing -> UPPER bound on filament density during myofibrillogenesis
FILAMENT_AREA_DENSITY = ACTIN_PER_MYOSIN / (np.sqrt(3) / 2 * MYOSIN_SPACING ** 2) # filaments/μm^2
FILAMENTS_PER_POINT = FILAMENT_AREA_DENSITY * UM_PER_LATTICE * MYOFIBRIL_DEPTH
FILAMENTS_PER_BUNDLE = FILAMENT_AREA_DENSITY * FIBRE_BUNDLE_WIDTH * MYOFIBRIL_DEPTH # reporting only

@dataclass
class Params:
    hex_end_frac: float = 0.35 # taper (shape knob — stays independent)
    canvas_pad: int = 10 # blank margin around the hexagon (what does it do for the physics?)
    # For mask and shape creation
    hex_half_w: float = (CELL_LENGTH / UM_PER_LATTICE / 2)
    hex_half_h: float = (CELL_HEIGHT / UM_PER_LATTICE / 2)
    W: int = 2 * round(hex_half_w) + 2 * canvas_pad
    H: int = 2 * round(hex_half_h) + 2 * canvas_pad
    fibre_width = round(FIBRE_BUNDLE_WIDTH / UM_PER_LATTICE) # bundle width
    # Supply: mol/L * (lattice point volume in L) * Avogadro. 1 μm^3 = 1e-15 L
    monomers_per_point: float = (MONOMER_CONC * 1e-6) * (UM_PER_LATTICE ** 2 * CELL_THICKNESS * 1e-15) * 6.022e23
    # Demand: one lattice point of a BUNDLE cross-section, not a single filament
    monomers_per_seg: float = FILAMENTS_PER_POINT * (UM_PER_LATTICE / (FIAMENT_LENGTH_PER_MONOMER * 1e-3))
    dt: float = 1.0 # seconds per step
    # Rates are per second at RESOLUTION=1, rescaled to per-step probabilities below.
    rate_nematic_depoly: float = 0.0063041 # [TUNED v13] was 0.0068080
                                       # when the branch gate opens, so it owns the crossover: at -40 %
                                       # the order curve never reaches 0.5 at all. It is SENSITIVE
                                       # (0.91 on L_dgo, against a 0.54 floor) but NOT IDENTIFIED --
                                       # the five tied finalists span 1.62x -- so the timing of clearance
                                       # is constrained only in COMBINATION with nematic_thresh.
    rate_nematic_poly: float = 0.0016555 # [TUNED v13] was 0.0015298
                                     # array never clears. Near-inert at phi_max 1.09: seeded points start
                                     # SATURATED so the frontier is empty, and the channel only opens once
                                     # depoly unsaturates a point. Sensitivity 0.33, below the 0.54
                                     # resolution -- INVISIBLE to the fit, do not quote as fitted.
    rate_grow: float = 0.0049036 # [TUNED v13] was 0.0082531 -- DOWN 41 %
                              # rate_grow is invisible to the fit (sensitivity 0.40 against a 0.54
                              # resolution); it is the RATIO that matters, and the ratio is the single
                              # most sensitive quantity in the model.
    rate_thin: float = 0.0013049 # [TUNED v13] thin:grow 0.2661 (v12 0.2381)
                               # QUANTITY IN THE MODEL: max |dL_dgo| = 1.83 at +-40 %, against a six-seed
                               # resolution of 0.54. At v13 it is also ONE OF THE TWO IDENTIFIED
                               # PARAMETERS: the five tied finalists span only 1.24x in it, well
                               # inside the 1.53x half-box agreement line. This is a CHANGE from
                               # v12, where it was sensitive but not identified. The monomer pool now bites (phi runs 0.53 -> 0.10
                               # against phi_max 1.09) and polymerise accepts with probability
                               # avail/monomers_per_point, so the EFFECTIVE ratio is ~1.55x nominal and
                               # the usable window is 0.15-0.30, not the near-1.0 the untuned file used.
                               # Thinning is also the ONLY knob that moves mesh closure (foam_ge8): a
                               # sparser field has bigger voids and closes fewer circuits, so gap and
                               # closure trade against each other on this one parameter.
    rate_branch: float = 0.0021422 # [TUNED v13] was 0.0019257
                                # pair at +-90 deg) and, because daughters grow from an OCCUPIED point of
                                # the mother, the only route to a CONNECTED one. Each daughter is also new
                                # fibre mass, so it is a density knob as well.
                                # DO NOT REPORT THIS AS A FITTED VALUE. Sensitivity 0.13 -- the third
                                # least sensitive field in the model, far below the 0.54 resolution --
                                # and the five tied finalists span 2.13x in it. The fit cannot see it.
    rate_nucleate: float = 0.0590394 # [TUNED v13] was 0.0788939 -- see note
                                # is_primary cortical sites of a 38.9 x 12.5 um cell -- NOT per cell per
                                # step -- so nucleation pressure scales with PERIMETER, not area. It makes
                                # only primary-axis fibres, so a high rate floods the field on the seeded
                                # array's own axis and lets nucleation rather than branching set the family
                                # balance. Sensitivity 0.08 -- the LEAST sensitive field in the model, and lower
                                # than at v12 (0.18) for a structural reason: with 222 competent sites
                                # instead of 159, nucleation pressure is set by the geometry rather
                                # than by this knob. Not quotable as a fitted value.
    @property
    def monomer_quantum(self): return self.monomers_per_seg / self.n_sub # monomers moved per event
    @property
    def k_nematic_depoly(self): return self.rate_nematic_depoly * self.dt * self.n_sub
    @property
    def k_nematic_poly(self): return self.rate_nematic_poly * self.dt * self.n_sub
    @property
    def k_grow(self): return self.rate_grow * self.dt * self.n_sub
    @property
    def k_decay(self): return self.rate_thin * self.dt * self.n_sub
    @property
    def k_branch(self): return self.rate_branch * self.dt
    @property
    def k_nucleate(self): return self.rate_nucleate * self.dt
    # Mesh spacing = crystalline lattice pitch (snap-to-grid).
    exclusion_len: int = round(MESH_SPACING / UM_PER_LATTICE) # lattice points. Measured (crosslinker span); never swept
    nematic_gap_min: int = round((NEMATIC_GAP_MIN) / UM_PER_LATTICE) # initial nematic array spacing (variable)
    nematic_gap_max: int = round((NEMATIC_GAP_MAX) / UM_PER_LATTICE)
    seed_gap: int = 0 # LINE-INDEX units. 0 => one fibre per template line, pitch == exclusion_len
    nematic_thresh = 0.35 # [TUNED v13 -- unchanged from v12] Fraction of the seeded nematic array that must still be present
                          # for branching to stay SHUT. AT v13 IT IS NO LONGER IDENTIFIED, and that is a
                          # change from v12: sensitivity 0.546 against a 0.54 floor is MARGINAL, and
                          # the five tied finalists hold BOTH 0.20 and 0.35 -- disagreement in the
                          # only sense a two-level gate has. It also owns the largest crossover response in the
                          # model, 6.4 h across +-40 %. Note this REVERSES three earlier rounds, which
                          # ranked it 8th-11th and called it nearly inert -- a sensitivity ranking is a
                          # property of the point it was measured at, not of the parameter.
    total_hours: float = 20.0
    # Noise
    axis_spread: float = 0.3594666 # [TUNED v13] = 20.6 deg (v12 22.2)
                                   # distribution. It is the only knob that acts LATE ONLY: the seeded
                                   # array is laid down in the canvas frame while mesh fibres use the
                                   # cell's own axis, so widening the spread lowers whole-crop order only
                                   # once the mesh dominates.
                                   # NOT IDENTIFIED and NOT SENSITIVE: 0.23 on L_dgo, and the five tied
                                   # finalists span 1.32x. It trades against rate_branch at constant loss --
                                   # more branching with a narrower spread fits as well as less branching
                                   # with a wider one, because both deliver the same off-axis fibre at the
                                   # same time and density, gap and order see only the total.
    mesh_axis: float = np.random.uniform(-np.pi / 8, np.pi / 8) # follows axis_spread
    angle_noise: float = 0.7131580 # [TUNED v13] = 40.9 deg (v12 39.8)
                                    # plateau, since the traces read 0.782 at 32 hpf rather than the 0.93
                                    # a clean array gives.
                                    # THE OTHER IDENTIFIED PARAMETER: sensitivity 0.99 and the five tied
                                    # finalists agree to 1.10x, the tightest agreement of any field. It is close to the pi/4 that earlier rounds
                                    # tested and REJECTED -- correctly, at THEIR settings, where a wide
                                    # spread flattened the order curve until the 0.5 crossing was seed
                                    # noise. It is survivable here only because it is PAIRED with a
                                    # nematic_thresh that holds branching shut until the array is gone:
                                    # the curve stays steep (0.804 -> 0.250) and crosses on 30 of 30
                                    # seeds. The interaction, not the value, is what makes it work.
    # Tau-leap steps
    @property
    def steps(self): return int(round(self.total_hours * self.steps_per_hour))
    record_every: int = 31
    @property
    def steps_per_hour(self): return 3600.0 / self.dt
    # Rendering params (read by cell_render.py; single Params source of truth).
    n_sub: int = 4 # sub-steps to fill one lattice point. 1 recovers the binary-pixel model
    pool_grid_step: int = 8 # sub-sampling stride for the free-monomer dot overlay
    pool_show_frac: float = 0.4 # draw a dot where the pool is at least this fraction of full
    cadherin_nucleation_prob: float = 0.2659230 # [TUNED v13] was 0.3486433
                                          # seed nucleates; a Bernoulli draw every 200 steps, not a Poisson
                                          # channel. Sensitivity 0.10, far below the 0.54 resolution, and the five tied
                                          # finalists span 2.72x in it -- the WIDEST disagreement of any
                                          # field. Invisible to the fit and free to roam the box.

# ── v13 fit, 30 seeds (7301-7320 selection, 7501-7510 held out) ─────────────
#
#   | hpf | density | traced |  p95 | traced | order | traced | foam_ge8 | traced | count | traced |
#   |----:|--------:|-------:|-----:|-------:|------:|-------:|---------:|-------:|------:|-------:|
#   |  32 |  0.054  | 0.047  | 5.91 |  8.14  | 0.785 | 0.782  |   0.07   |  0.01  |  44.7 |   29.3 |
#   |  36 |  0.079  | 0.095  | 3.18 |  3.64  | 0.672 | 0.717  |   0.20   |  0.77  |  74.8 |   71.0 |
#   |  40 |  0.087  | 0.090  | 2.80 |  5.18  | 0.566 | 0.555  |   0.30   |  1.12  |  82.3 |   71.7 |
#   |  44 |  0.095  | 0.091  | 2.57 |  3.70  | 0.393 | 0.279  |   0.55   |  1.43  |  84.1 |   77.0 |
#   |  48 |  0.090  | 0.099  | 2.71 |  4.28  | 0.256 | 0.371  |   0.54   |  2.02  |  80.3 |   77.0 |
#   |  52 |  0.090  | 0.078  | 2.66 |  5.82  | 0.207 | 0.273  |   0.45   |  1.57  |  83.0 |   62.0 |
#
#   IN THE LOSS      X_density 0.613   X_gap 1.207   X_order 0.619   ->  L_dgo 3.654
#   NOT IN THE LOSS  X_frag 1.697   X_count 0.833   crossover 41.66 +- 2.30 h
#   gates            foam_ge8 max 1.01 (ceiling 2.446), p95 max 8.10 (ceiling 10.32),
#                    no extinction -- 30 of 30 seeds pass all three, and all 30 cross
#
# THE 32 hpf DENSITY IS NOT A LOCKED FLOOR, and the earlier claim that it was has been
# withdrawn. At 32 hpf the field IS the seeded array, but the array's density is set by
# how many template rows fit across the cell's SHORT axis, i.e. by cell SHAPE. The
# constant-area aspect sweep that established this was run on the OLD nucleation rule,
# where the nucleation-competent FRACTION of the cortex was itself a function of aspect
# ratio (0.433 at aspect 1 rising to 0.861 at aspect 6.2, because the dead band's width
# was the cell height) -- so that sweep confounded shape with a nucleation-count ramp and
# is being re-run here. Under the corrected rule every cortical site nucleates at every
# aspect ratio, so only the perimeter term survives and the shape claim can be made
# cleanly for the first time.
# This is NOT licence to retune the cell: length and height are MEASURED. It means the
# 32 hpf residual sits inside the uncertainty of the measured aspect ratio, and that
# uncertainty has not yet been propagated.
#
# WHAT REMAINS UNREACHED, all three systematic and same-signed at every age -- the
# signature of a missing mechanism rather than a mis-set rate:
#   * THE FIELD IS TOO FULL. p95 2.7-5.6 um against a traced 3.6-8.1, in every tune
#     across six rounds and three geometries.
#   * THE MESH DOES NOT CLOSE. foam_ge8 at a third of traced, and the deficit WIDENS
#     with age. In this model circuits can only be closed by ADDING FIBRE -- closure and
#     line density are one knob -- whereas in a real heart they are two:
#     foam_ge8/density is 12.5 traced against 5.8 model -- the corrected
#     nucleation rule closed some of that gap for free, but not the shape of it.
#   * Segments are too short early: 10.4 / 7.5 um at 32 / 36 hpf against 14.6 / 11.3.
#   The standing hypothesis, untested, is that the first two are ONE failure and the
#   missing structure is the fibre-free nuclear / perinuclear zone that a real
#   cardiomyocyte has and this uniformly seedable hexagonal mask does not.
