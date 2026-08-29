"""Loss laboratory — six candidate objectives, scored on the same runs.

Every candidate is built from the same per-age residuals, so ONE pool of parameter
configurations can be scored under all of them. That is not just cheaper than tuning each
loss separately, it is the fair comparison: every loss searches exactly the same space with
exactly the same budget, so a difference between tunes is a difference between objectives
rather than between search efforts.

    z_i = (model_i - traced_i) / sd_i ;  X(obs) = sqrt(mean_i z_i^2)

--------------------------------------------------------------------------------------
WHAT WENT IN, AND WHY
--------------------------------------------------------------------------------------
density    total skeleton centreline length per unit area. How MUCH fibre.
gap_p95    95th percentile distance to the nearest fibre. How big the VOIDS are.
order      +-45 deg band statistic. ORIENTATION.
crossover  when order passes 0.5. TIMING.
count      number of skeleton segments >= 3.2 um. How MANY fibres.
frag       share of skeleton length in runs < 3.2 um. How BROKEN the field is.

COVERAGE IS NOT A CANDIDATE, and this is provable rather than a judgement.
`normalise()` skeletonises and then dilates to a fixed 1.2 um stroke, so coverage is
skeleton length times a constant: measured ratio 2.947 +- 0.059 on the 18 traced hearts and
3.208 +- 0.131 over 2545 model run-timepoints, r = 0.998 and 0.993. Because X is computed on
z = (model - traced)/sd_hearts, ANY constant factor cancels in numerator and denominator, so
X_coverage == X_density identically for a perfectly constant ratio. The only thing swapping
them would introduce is the 8.8 % offset between the model's ratio (3.21) and the traced one
(2.95) -- and that offset is an artefact of the model having more junction and end pixels per
unit length, not information about myofibrils. Swapping gap for coverage would therefore
delete the one spatial term and replace it with a noisier copy of density.

FOAM IS NOT A CANDIDATE either, for the weaker version of the same reason: foam =
39.3*density - 1.40 in the model (r = 0.745) and 32.4*density - 1.27 in the traces
(r = 0.835). It stays a hard gate, where a monotone function of density is exactly what is
wanted, and out of the loss, where it would double-count.

--------------------------------------------------------------------------------------
SD MODERATION -- the reason `shrunk` exists
--------------------------------------------------------------------------------------
Every SD here is estimated from THREE hearts, so it carries 2 degrees of freedom and a
relative uncertainty of about 52 %. Weighting by 1/sd^2 on such an estimate hands enormous
authority to whichever age happened to draw three similar hearts: traced order SD is 0.0169 at
44 hpf and 0.2172 at 36 hpf, a 13x span that is mostly sampling noise. `sd_shrunk` applies
standard empirical-Bayes variance moderation with the prior given the same weight as the data
(df0 = df = 2):

    s2_shrunk = 0.5 * s2_age + 0.5 * mean_over_ages(s2)

Only the `shrunk` loss uses it, so the comparison isolates the weighting change. The referee
score uses it too, so no single fragile SD can decide the winner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HPF = np.array([32.0, 36.0, 40.0, 44.0, 48.0, 52.0])
TRACED_XC = 40.81
XC_SD_H = 1.00
METRICS = ("density", "order", "gap_p95", "gap_p90", "foam", "foam_ge8", "count",
           "coverage", "seg_len_um", "frag", "junc")


class Targets:
    def __init__(self, per_image_csv="traced_per_image.csv"):
        df = pd.read_csv(per_image_csv)
        g = df.groupby("hpf")
        self.mean = {k: g[k].mean().reindex(HPF).to_numpy() for k in METRICS}
        self.sd = {k: g[k].std().reindex(HPF).to_numpy() for k in METRICS}
        # empirical-Bayes variance moderation, df0 = df = 2 (see module docstring)
        self.sd_shrunk = {k: np.sqrt(0.5 * self.sd[k] ** 2 + 0.5 * np.nanmean(self.sd[k] ** 2))
                          for k in METRICS}
        self.foam_ceiling = float(df.foam.max())
        self.foam8_ceiling = float(df.foam_ge8.max()) if "foam_ge8" in df else np.nan
        self.p95_ceiling = float(df.gap_p95.max())
        self.per_image = df


def crossover(order, hpf=HPF, level=0.5):
    v = np.asarray(order, float)
    for i in range(len(v) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and (v[i] - level) * (v[i + 1] - level) < 0:
            return float(hpf[i] + (hpf[i + 1] - hpf[i]) * (v[i] - level) / (v[i] - v[i + 1]))
    return np.nan


def X(model, mean, sd, use=None):
    z = (np.asarray(model, float) - mean) / sd
    if use is not None:
        z = z[np.asarray(use, bool)]
    z = z[np.isfinite(z)]
    return float(np.sqrt(np.mean(z ** 2))) if z.size else np.nan


# name -> (weights on per-age metrics, crossover weight, use shrunk sd, ages used)
LOSSES = {
    # the v5 objective. density + the spatial term + orientation + timing.
    "v5":     (dict(density=2.0, gap_p95=1.5, order=1.0), 0.5, False, None),
    # density dropped; the voids constrain the amount of fibre from the other side.
    "v6":     (dict(gap_p95=1.5, order=1.0), 0.5, False, None),
    # density replaced by COUNT: how many fibres rather than how much fibre. Count is a
    # different question from length -- a field can lose length by shortening every fibre
    # (count unchanged) or by losing fibres (count falls), and only the second is what
    # "the mesh is less complete" usually means.
    "count":  (dict(count=2.0, gap_p95=1.5, order=1.0), 0.5, False, None),
    # v5 terms, moderated SDs. Isolates the weighting change from the term change.
    "shrunk": (dict(density=2.0, gap_p95=1.5, order=1.0), 0.5, True, None),
    # v6 plus the structure term the visual gate exposed and no scalar in v5 could see.
    "struct": (dict(gap_p95=1.5, order=1.0, frag=1.0), 0.5, False, None),
    # --- VISUAL-PRIORITY family, added 23 Aug 2026 -------------------------------------
    # Rationale (Luka): the model should LOOK like the traces. Discrepancies in density and
    # order are defensible on grounds the model already admits -- the cell mask is a hexagon
    # and real cardiomyocytes are not, which moves density; and real fibres bend, curve and
    # cross at angles a little off 90 deg, which moves the +-45 deg band statistic. A field
    # made of short broken stubs is NOT defensible on any of those grounds: it is a statement
    # about the fibres themselves. So fragmentation outranks both.
    # ALL THREE USE MODERATED SDs. Raw frag SD is 0.0072 at 44 hpf -- the tightest number
    # anywhere in the traced set -- so on raw SDs a frag term would be almost entirely the
    # 44 hpf point.
    "vis":      (dict(frag=2.5, gap_p95=1.5, order=1.0), 0.5, True, None),
    "vis_d":    (dict(frag=2.5, gap_p95=1.5, order=1.0, density=0.5), 0.5, True, None),
    "vis_hard": (dict(frag=4.0, gap_p95=1.0, order=0.5), 0.5, True, None),
    # --- DENSITY + GAP + ORDER, NO CROSSOVER (added 23 Aug 2026, Luka) ------------------
    # The crossover comes OUT of the objective so it can be used to VALIDATE the tune: a
    # crossover the loss never saw is an independent prediction, and reproducing 40.81 +- 1.00
    # then means something. Weights are the v5 weights with the crossover term deleted, not
    # renormalised, so L_dgo = L_shrunk - 0.5*X_crossover and the two stay comparable.
    # Moderated SDs, on the same n = 3 argument as `shrunk`.
    # ---- foam IN THE LOSS, added 24 Aug 2026 at Luka's request -----------------------
    # "density + order + gap + foam". The foam term is foam_ge8, NOT raw foam, and that is
    # the whole point of the correction in measure.foam2(): raw foam counts 1-3 px
    # skeletonisation loops, which in the model are 48-65 % of all enclosed regions and carry
    # no biological meaning. Putting raw foam in a loss would tell the tuner to manufacture
    # rendering artefacts. foam_ge8 counts only regions of at least 8 px (1.13 um across),
    # which in the traces is unambiguously mesh.
    #
    # IT IS NOT A COPY OF DENSITY, and this was checked the same way coverage was:
    #   foam_ge8/density ratio   traced 12.51 +- 8.22   model 4.03 +- 3.53   (66 % / 88 % CV)
    #   coverage/density ratio   traced  2.95 +- 0.06   model  3.21 +- 0.13  ( 2 % /  4 % CV)
    # Coverage was rejected because a near-constant ratio cancels under z-normalisation.
    # foam_ge8 has no such relation: r(foam_ge8, density) = +0.77 traced / +0.68 model, and
    # ACROSS CANDIDATE TUNES the rank correlation of the two X terms is only rho = +0.53. It
    # also carries the one thing density cannot: the ratio differs 3x between model and
    # traces, which is the mesh-closure deficit.
    #
    # WHAT IT WILL TRADE AGAINST: the gap term. rho(Xs_foam8, Xs_gap) = -0.525 over the 51
    # v10 candidates -- closing more circuits means smaller voids, and p95 is already too
    # small. Expect this objective to walk toward LOWER thin:grow; the four best foam_ge8
    # candidates in the v10 search are all `_rlo` (thinning cut 30 %) moves.
    #
    # The crossover stays OUT, as in dgo, so it remains an out-of-sample prediction.
    "dgof":  (dict(density=2.0, gap_p95=1.5, order=1.0, foam_ge8=1.0), 0.0, True, None),
    # Same terms, foam weighted like density. Justified by size: Xs_foam8 averages 1.62 over
    # the v10 candidates against 1.36 gap, 1.15 density, 0.77 order -- it is the largest
    # residual in the set, so weight 1.0 arguably under-serves it. Tuned alongside rather
    # than instead of, because "how much weight" is a choice and should be shown as one.
    "dgof2": (dict(density=2.0, gap_p95=1.5, order=1.0, foam_ge8=2.0), 0.0, True, None),
    "dgo":      (dict(density=2.0, gap_p95=1.5, order=1.0), 0.0, True, None),
    # v5 with 32 hpf dropped. At 32 hpf the field is the seeded array alone, so its density
    # is fixed by locked geometry; scoring the model on a point no rate can move spends
    # budget on a constant and lets it distort the ranking.
    "no32":   (dict(density=2.0, gap_p95=1.5, order=1.0), 0.5, False, np.array([False] + [True] * 5)),
}

# The referee. Declared before any tune was run, used by none of the candidates: equal weight
# on all five independent per-age statistics plus the crossover, with moderated SDs so no
# single 3-heart SD decides the comparison. It is still a choice, not a neutral fact -- but it
# is the same choice for every candidate.
REFEREE = (dict(density=1.0, gap_p95=1.0, order=1.0, count=1.0, frag=1.0), 1.0, True, None)


def score(arrays, T, spec):
    w, wxc, shrunk, use = spec
    sd = T.sd_shrunk if shrunk else T.sd
    terms, L = {}, 0.0
    for k, wk in w.items():
        terms[f"X_{k}"] = X(arrays[k], T.mean[k], sd[k], use)
        L += wk * terms[f"X_{k}"]
    xc = crossover(arrays["order"])
    xx = abs(xc - TRACED_XC) / XC_SD_H if np.isfinite(xc) else abs(HPF[0] - TRACED_XC) / XC_SD_H
    terms["X_crossover"] = xx
    L += wxc * xx
    terms["L"] = L
    terms["xc"] = xc
    return terms


def gates(arrays, T):
    """Hard gates. NOT part of any loss -- these reject a field outright.

    THE FOAM GATE CHANGED 23 Aug 2026, and the reason is in measure.foam2(). The gate exists
    to reject a field that has stopped being a fibre network and become a filled tangle. Its
    old form capped the RAW count of enclosed regions at the traced maximum (2.614), but that
    count is dominated in the model by 1-3 px skeletonisation loops that carry no biological
    meaning: 48-65 % of the model's enclosed regions are <= 3 px against 5-17 % of the
    traces'. So the old gate capped a quantity that was mostly rendering artefact.

    Two things follow. (1) It rejected on seed noise -- five of the ten best candidates in
    the v9 round were rejected, EACH on exactly one seed out of four, while passing on the
    other three. (2) It pointed the wrong way: once holes below 8 px are discarded, the model
    has roughly HALF the enclosed regions the traces do at every mature age, so the real
    discrepancy is a shortage of mesh cells, not an excess.

    The gate now caps foam_ge8 at the traced foam_ge8 maximum (2.446). It keeps its power:
    the three worst offenders in the search pool read 3.25 / 3.46 / 4.64 on foam_ge8 and are
    still rejected, against 0.46-1.14 for every candidate the old gate removed. Raw foam is
    still recorded and reported; it is simply no longer what decides.
    """
    fails = []
    key = "foam_ge8" if ("foam_ge8" in arrays and np.isfinite(T.foam8_ceiling)
                         and np.all(np.isfinite(arrays["foam_ge8"]))) else "foam"
    ceil = T.foam8_ceiling if key == "foam_ge8" else T.foam_ceiling
    if np.nanmax(arrays[key]) > ceil:
        fails.append(f"{key} {np.nanmax(arrays[key]):.2f}")
    if np.nanmax(arrays["gap_p95"]) > T.p95_ceiling:
        fails.append(f"p95 {np.nanmax(arrays['gap_p95']):.2f}")
    if np.nanmin(arrays["density"]) <= 0 or not np.all(np.isfinite(arrays["density"])):
        fails.append("extinction")
    return "; ".join(fails) or "ok"


def evaluate_all(arrays, T):
    """One row: L under every candidate loss, the referee, all X terms, and the gates."""
    row = {}
    for name, spec in LOSSES.items():
        row[f"L_{name}"] = score(arrays, T, spec)["L"]
    ref = score(arrays, T, REFEREE)
    row["L_ref"] = ref["L"]
    for k in ("density", "gap_p95", "order", "count", "frag"):
        row[f"X_{k}"] = X(arrays[k], T.mean[k], T.sd[k])
        row[f"Xs_{k}"] = X(arrays[k], T.mean[k], T.sd_shrunk[k])
    row["X_crossover"] = ref["X_crossover"]
    row["xc"] = ref["xc"]
    row["gate"] = gates(arrays, T)
    row["foam_max"] = float(np.nanmax(arrays["foam"]))
    row["p95_max"] = float(np.nanmax(arrays["gap_p95"]))
    return row


def frame(runs_csv, T=None, seeds=None):
    """Score a runs CSV: one row per case, averaged over seeds first."""
    T = T or Targets()
    d = pd.read_csv(runs_csv)
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    out = []
    for case, g in d.groupby("case", sort=False):
        arrays = {k: g.groupby("hpf")[k].mean().reindex(HPF).to_numpy() for k in METRICS}
        r = evaluate_all(arrays, T)
        r["case"] = case
        r["n_seeds"] = int(g.seed.nunique())
        out.append(r)
    cols = (["case", "n_seeds"] + [f"L_{k}" for k in LOSSES] + ["L_ref"]
            + [c for c in out[0] if c not in ("case", "n_seeds")
               and not c.startswith("L_")])
    return pd.DataFrame(out)[cols]
