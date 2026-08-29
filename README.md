# Atrial actin reorganisation — model and analysis code

Code accompanying the MRes thesis *(title)*, Department of Bioengineering,
Imperial College London.

A stochastic bundle-lattice model of the nematic-to-quartic actin reorganisation in
zebrafish atrial cardiomyocytes between 32 and 52 hpf, together with the image-analysis
and calibration code used to measure the hand-traced confocal images the model is
calibrated against.

This repository contains **only** the code behind results that appear in the thesis.
Superseded tuning rounds, abandoned analyses and exploratory figures are not included;
see [What is not here](#what-is-not-here).

---

## Layout

```
modelling/CARMA/carma_6_particle/   the model itself
    cell_particle.py                one cell: nucleation, growth, branching, thinning
    multicell_particle.py           the sheet: cell tiling, cadherin coupling, render
    parameters.py                   geometry, rates and derived constants
    order_params.py                 orientation helpers used inside the model

lib/                                shared layers every script imports
    measure.py                      the measurement layer: density, gap p95, order,
                                    count, frag, foam  (all loss evaluation goes here)
    band_dominance.py               oriented_profile, band_dominance, band_map,
                                    band_summary — the traced-only C2/C4 work
    carma_stats.py                  low-level image statistics (skeleton, FFT, windows)
    sim3.py                         run harness: configure, run_once, field, paste
    loss_lab.py                     the candidate objectives; L_dgo is density+gap+order
    thesisstyle.py                  figure style used by the print-size figures
    traced_dominance.py             per-image measurement of the hand-traced set
    bias_check.py                   crossing-angle measurement on traced masks
    _bias_check.py                  alias for bias_check (see file header)

analysis/
    traced_order/                   orientational order in the traced tissue
    spacing/                        fibre spacing and width from the confocal images
    calibration/                    the v13 tune, sensitivity, multi-start, aspect sweep
    seed_angle/                     the nucleation dead band and the three-arm test
    convergence/                    dt and n_sub convergence
    crossing_check/                 crossing-angle construction check

figures/                            one script per thesis figure
docs/figure-map.md                  thesis figure → script → inputs
```

---

## Running it

Nothing is installed as a package. Scripts import their siblings directly and the model
by its full path, so three directories go on `PYTHONPATH`:

```bash
pip install -r requirements.txt

# from the repository root
export PYTHONPATH="$PWD:$PWD/lib:$PWD/modelling/CARMA/carma_6_particle"
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\lib;$PWD\modelling\CARMA\carma_6_particle"
```

Scripts read and write their input and output files in the **current working directory**,
so run each one from the directory holding its inputs.

A short end-to-end check that the model runs:

```python
import sim3
out = sim3.run_once(seed=1, hpf=[32.0, 32.6], verbose=True)
```

Python 3.11. `statsmodels` is imported lazily by a few reporting paths in
`carma_stats.py` and is only needed for those.

---

## Data

Code only. The repository does not carry the confocal images, the hand-traced masks, or
the derived CSV/JSON files the figure scripts read. `docs/figure-map.md` lists, for every
thesis figure, the script that draws it and the input files that script expects, and which
analysis script produces each of them.

---

## What is not here

Deliberately excluded, so that what remains is what the thesis actually rests on:

- **Tuning rounds v5–v12.** Superseded by v13. The nucleation rule changed between v12 and
  v13, so sensitivity, identifiability and resolution figures measured under the older rule
  do not transfer and are not reproduced here.
- **The untraced-image dominance work** (acceptance scan, boundary coverage, variant sweep).
  Superseded by the traced measurements; none of it appears in the thesis.
- **Earlier versions of the traced figures** (F1–F5) and the earlier order figures
  (`c2c4_transition`, `c2c4_plane`), all folded into the two figures that remain.

One dead end *is* kept, because a live script depends on it: `analysis/traced_order/traced_c4.py`
computes C2/C4 from skeleton branch angles. That approach does not work — skeletonisation cuts
every fibre at its junctions, so a branch is short and nearly straight whatever the fibre does,
and C4 ends up measuring branch straightness. The spectral estimator in `fft_c4.py` is the one
used in the thesis. `traced_c4.py` remains because `curved_refs.py` imports its
`branch_angles_weighted` helper for the curvature control.

## Provenance note

`multicell_particle.py` and `order_params.py` are the 22 August 2026 versions. `cell_particle.py`
and `parameters.py` are the corrected v13 versions (the fixed `_seed_angle` rule, and the
re-measured cell geometry).

---

## License

MIT — see `LICENSE`.
