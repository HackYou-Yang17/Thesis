# Thesis figure → script → inputs

Every figure in the thesis that this repository is responsible for. Figures not listed
(`zebrafish_embryo_heart`, `observation`, `chopra`, and the confocal snapshots) are
photographs or drawn illustrations, not script output.

Run each script from the directory that holds its inputs, with `PYTHONPATH` set as in the
README.

| § | figure | script | inputs | produced by |
|---|---|---|---|---|
| 2.2 | `spacing_box.png` | `figures/make_box_fig.py` | `green_dists.npy` | `analysis/spacing/green.py` |
| 2.3 | `F_schematic_v13.png` | `figures/make_Fschematic_print.py` | none — runs the model | — |
| 3.1 | `order_transition.png` | `figures/make_order_fig.py` | `order_traced.json`; optional `model_c2c4.json` | `analysis/traced_order/fft_c4.py`; model C2/C4 pass |
| 3.1 / 3.4 | `traced_curves_v13.png` | `figures/make_F1_print.py` | `V13F_runs.csv`, `V13H2_runs.csv`, `V13b_runs.csv` | `analysis/calibration/v13_tune.py` |
| 3.3 | `F_constrain_v13.png` | `figures/make_F_constrain_print.py` | `MS13_final_runs.csv`, `SENSV13_scored.csv`, `V13_tied_spread.csv`, `NOISE13.json`, `traced_per_image.csv` | `multistart13.py`, `sens_v13.py`, `v13_tune.py`, `lib/traced_dominance.py` |
| 3.3 | `F_aspect_v13.png` | `figures/make_F_aspect_print.py` | `ASPECT13_runs.csv` | `analysis/calibration/aspect_sweep13.py` |
| appendix | `crossing_check.png` | `analysis/crossing_check/fig.py` | `res.json` | `analysis/crossing_check/full.py` (after `calib.py`) |

## Notes on inputs

- **`traced_per_image.csv`** — `lib/traced_dominance.py` writes this table as
  `traced_dominance_per_image.csv`. `loss_lab.Targets()` and
  `analysis/seed_angle/seedangle_score.py` expect it under the shorter name; rename or
  symlink it.
- **`model_c2c4.json`** is optional. Supply `{stage: [[C2, C4], ...]}`, one entry per seed,
  beside `make_order_fig.py` and the model path is drawn into panel B of
  `order_transition.png`. Without it panel B shows the traced path and the reference
  corners only.
- **`green_dists.npy`** is a pickled dict, so `np.load(..., allow_pickle=True)`.

## Figure checks

Two helpers, not figures:

- `figures/check_box.py` re-runs `make_box_fig.py` and reports any text element below 11 pt.
- `figures/glyphscan.py` measures the smallest rendered glyph in a finished PNG, for
  figures whose build script is not to hand. Calibrate it on a figure of known authored
  size before trusting it elsewhere.

## Analysis scripts not tied to a figure

These produce numbers quoted in the text rather than a figure:

| script | what it reports |
|---|---|
| `analysis/traced_order/reference_fields.py` | reference fields of known structure at matched line density |
| `analysis/traced_order/curved_refs.py` | the curvature control on C2 |
| `analysis/spacing/nn.py`, `run_nn.py`, `spacing.py` | nearest-neighbour fibre spacing |
| `analysis/spacing/width.py`, `raw_width.py`, `width_levels.py` | fibre bundle width |
| `analysis/spacing/cellseg.py`, `handcells.py` | cell segmentation and the hand-traced cell check |
| `analysis/calibration/multistart13.py` | multi-start basins and identifiability |
| `analysis/calibration/sens_v13.py` | one-at-a-time parameter sensitivity |
| `analysis/calibration/v13_crossings.py` | crossover timing at the tuned rates |
| `analysis/seed_angle/seedangle_run.py`, `_score.py`, `_check.py` | the nucleation dead band and the three-arm test |
| `analysis/convergence/converge13.py`, `conv_probe.py` | dt and `n_sub` convergence |
