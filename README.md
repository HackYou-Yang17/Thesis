# A Kinetic Model of Actin Network Organisation in the Zebrafish Atrium

Model and analysis code accompanying the MRes thesis of the same name, Department of
Bioengineering, Imperial College London.

A stochastic bundle-lattice model of the nematic-to-quartic actin reorganisation in
zebrafish atrial cardiomyocytes between 32 and 52 hpf, together with the image-analysis
and calibration code used to measure the hand-traced confocal images the model is
calibrated against.

**Scope rule (3 Sep 2026).** This repository holds the code that was used to produce the
figures, tables, numbers and methods of the thesis, plus the hand-traced images it was run
on — nothing else. Superseded routes, exploratory analyses and figure drafts have been removed; see
[What is not here](#what-is-not-here) and, more importantly,
[What is missing](#what-is-missing--three-scorers-never-saved).

---

## Layout

Three top-level directories, split by one rule: **`pipeline/` is everything that another
file imports; `figures/` and `analysis/` are the leaves that nothing imports** (one
exception: `figures/appendix-crossing-angle/full_tan.py` imports its sibling
`tangent_crossings.py`). So a module appears exactly once, and moving or editing a figure
can never fork the measurement code underneath it.

### `pipeline/` — the shared measurement and model layer

```
bundle_model/                       the model itself (thesis §2.3–2.5, Appendix A.6–A.9)
    cell_particle.py                one cell: nucleation, growth/thickening, branching,
                                    disassembly, the monomer field and the tau-leap step
    multicell_particle.py           the sheet: cell tiling, window placement, cadherin coupling
    parameters.py                   geometry, rates and derived constants (Tables 1 and 2)
    order_params.py                 orientation helpers used inside the model
    cell_render.py                  per-cell intensity maps for the rendered movie
    multicell_render.py             the simulated confocal pipeline (Appendix A.10): haze,
                                    junction boost, PSF blur, noise, gamma; movie and snapshots.
                                    RENDER ONLY — the measured statistics never pass through it

measure.py            the measurement layer both traces and model go through (§2.6, A.11):
                      normalise, density, band statistic, gap p95, mesh closure, and the
                      fibre-stitching mean fibre length (fibres(), 40° turn) — additive, 2 Sep
band_dominance.py     oriented_profile, band_dominance, band_map, band_summary (the null-corrected
                      angular power profile and the ±45° band statistic)
fieldstats.py         low-level image statistics (skeleton, FFT windowing, gap_percentiles).
                      Large; only the functions measure.py and band_dominance.py call are live
sim3.py               run harness: configure, run_once, field, paste. configure() defaults
                      to bundle_model/parameters.py as committed (the historical v5 ANCHOR
                      dict is kept under ANCHOR_V5 for provenance only)
loss_lab.py           the loss L = 2X_density + 1.5X_gap + X_order, moderated SDs, gates (§2.7)
thesisstyle.py        figure style shared by the print-size figures
traced_dominance.py   load_trace and per-image measurement of the hand-traced set
spacing.py            skeleton normalisation and the arc-local tangent (its own normalise — see below)
nn.py                 perpendicular nearest-neighbour fibre spacing, 5 µm cutoff (§2.2, A.5)
width.py              FWHM / EDT width helpers at traced centrelines; imported by raw_width.py
reference_fields.py   reference fields of known structure at matched line density (A.12)
curved_refs.py        the curvature control on C2 (A.12)
traced_c4.py          skeleton-branch C2/C4 — a dead end, kept as a dependency (see below)
datapaths.py          resolves where the traced images live (see Data below)
```

### `figures/` — one directory per thesis figure

Each directory holds the scripts unique to that figure, in dependency order. Some output
names were shortened by hand when placed in the thesis (`F_schematic_v13.png` →
`schematic_v13.png`, `F_constrain_v13.png` → `constrain_v13.png`, `F_aspect_v13.png` →
`aspect_v13.png`).

```
fig2.2-spacing/            green.py -> make_box_fig.py -> spacing_box.png            (Fig. 2, §2.2)
appendix-crossing-angle/   full_tan.py -> fig_tangent.py -> crossing_check_tan.png   (Fig. 3, §2.2)
                           tangent_crossings.py is the declared junction-tangent route (A.3)
fig2.3-schematic/          make_Fschematic_print.py -> F_schematic_v13.png           (Fig. 4, §2.6)
fig3.1-order/              fft_c4.py -> make_order_fig.py -> order_transition.png    (Fig. 5, §3.1)
fig3.1-3.4-traced-curves/  make_F1_print.py -> traced_curves_v13.png                (Fig. 6, §3.1/3.4)
fig3.3-constraints/        make_F_constrain_print.py -> F_constrain_v13.png         (Fig. 7, §3.2)
fig3.3-aspect/             make_F_aspect_print.py -> F_aspect_v13.png               (Fig. 9, §3.4)
```

### `analysis/` — runs that produce numbers, not figures

```
spacing/       handcells.py           cell length/width from the hand-traced 52 hpf outlines (§2.1, A.2)
               raw_width.py           bundle width w = 1.23 µm on the untraced images: tubeness ridges at
                                      the 70th percentile, mask area / skeleton length, pooled over 18 (Table 1, A.7)
calibration/   sweep.py               THE BATCH RUNNER: cases.json -> sim3.run_once per case × seed
                                      -> <name>_runs.csv / _summary.csv; every V13*/MS13* runs CSV came from it
               v13_tune.py            the two-round Latin-hypercube case lists (§2.7, A.13)
               multistart13.py        the six multi-start case lists and basin selection (§3.2, A.13)
               sens_v13.py            the ±40 % one-at-a-time sensitivity case list (§3.2, A.13)
               aspect_sweep13.py      cell shape at constant area — runs the model itself (§3.4, Fig. 9)
fibrelength/   run_fib_v13.py         model-side fibre length at the committed tune, seeds
                                      7301–7320 + 7501–7510 -> V13FIB_runs.csv (Fig. 6)
               traced_targets_fib.py  traced-side writer of traced_per_image_fib.csv; re-run
                                      3 Sep 2026 against the TIFFs: reproduces it exactly
convergence/   converge13.py          dt and n_sub convergence at fixed rates (§4.1, A.9)
               conv_probe.py          why polymer mass is not n_sub-invariant: the quantum gate (A.9)
               V13_TUNE.json          the calibrated tune converge13/conv_probe read
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

In VS Code none of this is needed: `.vscode/settings.json` is committed with the repo, so
Pylance resolves the imports and every new integrated terminal already has `PYTHONPATH`
set. Reload the window after cloning for it to take effect.

Scripts read and write their input and output files in the **current working directory**,
so run each one from the directory holding its inputs.

A short end-to-end check that the model runs and renders:

```python
import sim3
out = sim3.run_once(seed=1, hpf=[32.0, 32.6], verbose=True)

from bundle_model import multicell_particle as mc, multicell_render as mr
frame = mr.render_frame(mc.MultiCell(base_seed=1), 0)
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
        <stage>-<n> - Copy.tif          the hand-traced fibre annotation (red)
        <stage>-<n> - Copy (2).tif      the second annotation layer (yellow-green; 32 and 52 hpf
                                        only — the spacing and crossing-angle set)
    52hpf cell size.tif                 the membrane-marked image
    52hpf cell size - Copy.tif          its hand-traced cell boundaries (handcells.py reads this)
```

Every script finds them through `pipeline/datapaths.py`, which resolves in this order:

1. the `TRACED_ROOT` environment variable, if set
2. `data/` at the repository root

So the pipeline runs from a clean clone with no configuration. Point `TRACED_ROOT` at
another directory to run it against a different set of traces.

Reading these files prints `tifffile.imagej_metadata raised IndexError` on stderr. That is
`tifffile` failing to parse an ImageJ metadata tag the annotations carry; it does not touch
the pixel data and every measurement is unaffected. It can be ignored.

**Not included:** the derived CSV and JSON files the figure scripts read — the tune
outputs, the per-image measurement tables, the reference-field results.
`docs/figure-map.md` names each one and, where the producer is in this repository, the
script that produces it.

---

## What is missing — three scorers, never saved

The archive `OneDrive\analysis\` (A_tune_v13, A_fibrelength, A_order_refs, …) holds every
derived CSV/JSON the figure scripts read, so no number is lost. Three small scripts were
built inline in a session and never saved; their **outputs** are archived, but they cannot
be re-derived if anything upstream changes:

1. the scorer that wrote `SENSV13_scored.csv` (`param` → `maxdev`) from `SENSV13_runs.csv`;
2. the scorer that wrote `V13_tied_spread.csv` (`param` → `log2`) across the five tied finalists;
3. the script that computed `NOISE13.json` (`sd_pair` = 0.472) from `NOISE13_perseed.csv`.

The writer of `figures/fig3.1-order/order_traced.json` was also never saved (`fft_c4.py`
prints the same quantities but writes nothing); the JSON itself is archived and a copy is
included beside `make_order_fig.py`, so Figure 5 rebuilds.

## What is not here

Deliberately excluded, so that what remains is what the thesis actually rests on:

- **Tuning rounds v5–v12**, superseded by v13. The nucleation rule changed between v12 and
  v13, so figures measured under the older rule do not transfer.
- **The whole-branch PCA crossing-angle route** (`bias_check.py` and its alias, `calib.py`,
  `full.py`, `fig.py`, `decisive.py`, `v13_crossings.py`). The thesis declares the
  junction-tangent definition (Section 2.2, Appendix A.3); the PCA route measured a
  different quantity and is no longer reported.
- **The nucleation dead-band fix** (`analysis/seed_angle/`). The fix is in the model
  (`_seed_angle` fires along the long axis; 222 competent sites); the paired test that
  motivated it is not in the thesis.
- **Untraced-image analyses** (`raw_detect.py`, `run_nn.py`, `simple_untraced.py`,
  `width_levels.py`, watershed `cellseg.py`). The thesis uses the hand traces for every
  statistic; the one untraced measurement it keeps is the bundle width (`raw_width.py`), and
  the cell dimensions come from the hand-traced boundaries (`handcells.py`).
- **Figure drafts and type-size tools** (`fig_v2/v4/v5.py`, `check_box.py`, `glyphscan.py`).
- **Earlier versions of the traced figures** (F1–F5) and the earlier order figures,
  all folded into the figures that remain.

One dead end *is* kept, because a live script depends on it: `pipeline/traced_c4.py`
computes C2/C4 from skeleton branch angles. That approach does not work — skeletonisation cuts
every fibre at its junctions, so a branch is short and nearly straight whatever the fibre does,
and C4 ends up measuring branch straightness (Appendix A.11). The spectral estimator in
`figures/fig3.1-order/fft_c4.py` is the one used in the thesis. `traced_c4.py` remains because
`curved_refs.py` imports its `branch_angles_weighted` helper for the curvature control.

## Repeated names that are not repeated code

Several function names appear in more than one module. They are **not** interchangeable,
and they are kept apart on purpose: each was the version actually used by the scripts
around it, and the numbers in the thesis came from that version.

| name | where | difference that matters |
|---|---|---|
| `normalise` | `pipeline/measure.py` · `pipeline/spacing.py` | `measure` resamples with a linear resize at a 0.15 threshold, re-thins, then **dilates to a fixed 1.2 µm stroke**. `spacing` uses nearest-neighbour zoom, re-thins, **does not dilate**, and pads before cropping. The spacing work measures centreline geometry, so a stroke width would corrupt it. |
| `load_trace` | `pipeline/measure.py` · `pipeline/traced_dominance.py` | The same logic. `traced_dominance` inlines `measure.normalise` rather than calling it, so the two must be changed together if they are changed at all. |

`load`, `files`, `measure`, `draw`, `synth`, `paste`, `_one` and `use_style` also recur.
Those are local helpers scoped to their own script and share nothing but a name.

## Provenance note

`multicell_particle.py` and `order_params.py` are verified byte-identical to the working
copies. `cell_particle.py` and `parameters.py` are the corrected v13 versions (the fixed
`_seed_angle` rule, and the re-measured cell geometry). `cell_render.py` and
`multicell_render.py` had their imports repointed from the original development tree
(`modelling.CARMA.carma_6_particle.*`) to `bundle_model.*` on 3 Sep 2026; nothing else in
them changed.

---

## License

MIT — see `LICENSE`.
