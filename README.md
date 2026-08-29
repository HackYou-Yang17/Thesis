# A Kinetic Model of Actin Network Organisation in the Zebrafish Atrium

Model and analysis code accompanying the MRes thesis of the same name, Department of
Bioengineering, Imperial College London.

A stochastic bundle-lattice model of the nematic-to-quartic actin reorganisation in
zebrafish atrial cardiomyocytes between 32 and 52 hpf, together with the image-analysis
and calibration code used to measure the hand-traced confocal images the model is
calibrated against.

This repository contains the code behind the results that appear in the thesis, and the
hand-traced images the model is calibrated against.
Superseded tuning rounds, abandoned analyses and exploratory figures are not included;
see [What is not here](#what-is-not-here).

---

## Layout

Three top-level directories, split by one rule: **`pipeline/` is everything that another
file imports; `figures/` and `analysis/` are the leaves that nothing imports.** So a module
appears exactly once, and moving or editing a figure can never fork the measurement code
underneath it.

### `pipeline/` — the shared measurement and model layer

```
bundle_model/                       the model itself
    cell_particle.py                one cell: nucleation, growth, branching, thinning
    multicell_particle.py           the sheet: cell tiling, cadherin coupling, render
    parameters.py                   geometry, rates and derived constants
    order_params.py                 orientation helpers used inside the model

measure.py            the measurement layer: density, gap p95, order, count, frag, foam.
                      Every loss evaluation goes through this file
band_dominance.py     oriented_profile, band_dominance, band_map, band_summary
fieldstats.py         low-level image statistics (skeleton, FFT, windowing)
sim3.py               run harness: configure, run_once, field, paste
loss_lab.py           the candidate objectives; L_dgo is density + gap + order
thesisstyle.py        figure style shared by the print-size figures
traced_dominance.py   per-image measurement of the hand-traced set
bias_check.py         crossing-angle measurement on traced masks
_bias_check.py        alias for bias_check (see file header)
spacing.py            the spacing measurement chain (its own normalise — see below)
nn.py                 nearest-neighbour fibre spacing
raw_detect.py         fibre detection on unannotated images
run_nn.py             per-image nearest-neighbour driver
width.py              fibre bundle width
reference_fields.py   reference fields of known structure at matched line density
curved_refs.py        the curvature control on C2
traced_c4.py          skeleton-branch C2/C4 — a dead end, kept as a dependency (see below)
datapaths.py          resolves where the traced images live (see Data below)
```

### `figures/` — one directory per thesis figure

Each directory holds the scripts unique to that figure, in dependency order.

```
fig2.2-spacing/            green.py -> make_box_fig.py -> spacing_box.png
                           check_box.py audits the type size
fig2.3-schematic/          make_Fschematic_print.py -> F_schematic_v13.png
fig3.1-order/              fft_c4.py -> make_order_fig.py -> order_transition.png
fig3.1-3.4-traced-curves/  make_F1_print.py -> traced_curves_v13.png
fig3.3-constraints/        make_F_constrain_print.py -> F_constrain_v13.png
fig3.3-aspect/             make_F_aspect_print.py -> F_aspect_v13.png
appendix-crossing-angle/   calib.py, full.py -> fig.py -> crossing_check.png
                           decisive.py is the supporting check
_tools/                    glyphscan.py, a type-size audit for any finished PNG
```

### `analysis/` — runs that produce numbers, not figures

```
spacing/       cellseg.py, handcells.py, raw_width.py, width_levels.py,
               simple_untraced.py
calibration/   v13_tune.py, multistart13.py, sens_v13.py, aspect_sweep13.py,
               v13_fields.py, v13_crossings.py
seed_angle/    the nucleation dead band and the three-arm test
convergence/   dt and n_sub convergence
```

`docs/figure-map.md` gives the figure → script → input-file chain in full.

---

## Running it

Nothing is installed as a package. Everything importable lives in `pipeline/`, so two
directories go on `PYTHONPATH` and no script needed editing to sit where it does:

```bash
pip install -r requirements.txt

# from the repository root
export PYTHONPATH="$PWD/pipeline:$PWD/pipeline/bundle_model"
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\pipeline;$PWD\pipeline\bundle_model"
```

The second entry exists because `curved_refs.py` and `traced_c4.py` import `order_params`
by its bare name rather than through the package path.

Scripts read and write their input and output files in the **current working directory**,
so run each one from the directory holding its inputs.

A short end-to-end check that the model runs:

```python
import sim3
out = sim3.run_once(seed=1, hpf=[32.0, 32.6], verbose=True)
```

Python 3.11. `statsmodels` is imported lazily by a few reporting paths in
`fieldstats.py` and is only needed for those.

---

## Data

`data/` carries the hand-traced images the model is calibrated against — 44 TIFFs, 1.5 MB.
Three hearts at each of 32, 36, 40, 44, 48 and 52 hpf, plus the two cell-size images:

```
data/
    32hpf/ … 52hpf/     per stage:
        <stage>-<n>.tif                 the confocal image
        <stage>-<n> - Copy.tif          the hand-traced fibre annotation
        <stage>-<n> - Copy (2).tif      the second annotation layer (32 and 52 hpf only,
                                        the two endpoints the spacing work uses)
    52hpf cell size.tif
    52hpf cell size - Copy.tif
```

Every script finds them through `pipeline/datapaths.py`, which resolves in this order:

1. the `TRACED_ROOT` environment variable, if set
2. `data/` at the repository root

So the pipeline runs from a clean clone with no configuration. Point `TRACED_ROOT` at
another directory to run it against a different set of traces.

**Not included:** the derived CSV and JSON files the figure scripts read — the tune
outputs, the per-image measurement tables, the reference-field results.
`docs/figure-map.md` names each one and the script that produces it, so they can be
regenerated from the images and the model.

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

One dead end *is* kept, because a live script depends on it: `pipeline/traced_c4.py`
computes C2/C4 from skeleton branch angles. That approach does not work — skeletonisation cuts
every fibre at its junctions, so a branch is short and nearly straight whatever the fibre does,
and C4 ends up measuring branch straightness. The spectral estimator in `figures/fig3.1-order/fft_c4.py` is the one
used in the thesis. `traced_c4.py` remains because `curved_refs.py` imports its
`branch_angles_weighted` helper for the curvature control.

## Repeated names that are not repeated code

Several function names appear in more than one module. They are **not** interchangeable,
and they are kept apart on purpose: each was the version actually used by the scripts
around it, and the numbers in the thesis came from that version. Unifying them now would
silently change which pipeline ran.

| name | where | difference that matters |
|---|---|---|
| `normalise` | `pipeline/measure.py` · `pipeline/spacing.py` | `measure` resamples with a linear resize at a 0.15 threshold, re-thins, then **dilates to a fixed 1.2 µm stroke**. `spacing` uses nearest-neighbour zoom, re-thins, **does not dilate**, and pads before cropping. The spacing work measures centreline geometry, so a stroke width would corrupt it. |
| `branch_angles` | `pipeline/measure.py` · `pipeline/bias_check.py` | Different junction tests (`degree < 3` on an 8-neighbour convolution against `neighbours >= 3` removed), and different minimum branch lengths. |
| `load_trace` | `pipeline/measure.py` · `pipeline/traced_dominance.py` | The same logic. `traced_dominance` inlines `measure.normalise` rather than calling it, so the two must be changed together if they are changed at all. |

`load`, `files`, `measure`, `draw`, `synth`, `paste`, `_one` and `use_style` also recur.
Those are local helpers scoped to their own script and share nothing but a name.

## Provenance note

`multicell_particle.py` and `order_params.py` are verified byte-identical to the working copies. `cell_particle.py`
and `parameters.py` are the corrected v13 versions (the fixed `_seed_angle` rule, and the
re-measured cell geometry).

---

## License

MIT — see `LICENSE`.
