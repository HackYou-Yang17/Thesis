# Thesis figure → script → inputs

Every figure this repository is responsible for. Figures not listed (`observation.png`,
`zebrafish_embryo_heart.png`, `contact_junction_concept.png`, the traced-image panels of
Appendix A.2, and the confocal row of `real_vs_sim.png`) are photographs, hand annotations
or drawn illustrations, not script output.

Each figure has its own directory under `figures/`. Everything those scripts import lives
in `pipeline/`. Run each script from the directory holding its inputs, with `PYTHONPATH`
set as in the README. Output names in brackets were shortened by hand when the file was
placed in the thesis `figures/` folder.

| Fig. | § | thesis file | directory | chain | inputs | produced by |
|---|---|---|---|---|---|---|
| 2 | 2.2 | `spacing_box.png` | `figures/fig2.2-spacing/` | `green.py` → `make_box_fig.py` | `green_dists.npy` | `green.py` |
| 3 | 2.2 | `crossing_check_tan.png` | `figures/appendix-crossing-angle/` | `full_tan.py` → `fig_tangent.py` | `res_tan.json` | `full_tan.py` (uses `tangent_crossings.py`) |
| 4 | 2.6 | `schematic_v13.png` [`F_schematic_v13.png`] | `figures/fig2.3-schematic/` | `make_Fschematic_print.py` | none — runs the model | — |
| 5 | 3.1 | `order_transition.png` | `figures/fig3.1-order/` | `fft_c4.py` → `make_order_fig.py` | `order_traced.json` (included in the directory); optional `model_c2c4.json` | archived JSON; its writer was never saved (README, "What is missing") |
| 6 | 3.1 / 3.4 | `traced_curves_v13.png` | `figures/fig3.1-3.4-traced-curves/` | `make_F1_print.py` | `V13FIB_runs.csv`, `traced_per_image_fib.csv` | `analysis/fibrelength/run_fib_v13.py` and `traced_targets_fib.py` |
| 7 | 3.2 | `constrain_v13.png` [`F_constrain_v13.png`] | `figures/fig3.3-constraints/` | `make_F_constrain_print.py` | `MS13_final_runs.csv`, `SENSV13_scored.csv`, `V13_tied_spread.csv`, `NOISE13.json`, `traced_per_image.csv` | case lists from `multistart13.py`, `sens_v13.py`, `v13_tune.py`, run by `analysis/calibration/sweep.py`; `traced_per_image.csv` from `pipeline/traced_dominance.py`; the three scorers were never saved (README, "What is missing") — their outputs are archived |
| 8 | 3.3 | `real_vs_sim.png` | — | model row: `bundle_model/multicell_render.save_snapshots` | rendered frames at the v13 tune (the movie in the repository) | assembled by hand |
| 9 | 3.4 | `aspect_v13.png` [`F_aspect_v13.png`] | `figures/fig3.3-aspect/` | `aspect_sweep13.py` → `make_F_aspect_print.py` | `ASPECT13_runs.csv` | `analysis/calibration/aspect_sweep13.py` |

## Notes on inputs

- **`traced_per_image.csv`** — `pipeline/traced_dominance.py` writes this table as
  `traced_dominance_per_image.csv`. `loss_lab.Targets()` expects it under the shorter name;
  rename or symlink it. `traced_per_image_fib.csv` is the same table with the fib columns
  added, written by `analysis/fibrelength/traced_targets_fib.py`.
- **`model_c2c4.json`** is optional. Supply `{stage: [[C2, C4], ...]}`, one entry per seed,
  beside `make_order_fig.py` and the model path is drawn into panel B of
  `order_transition.png`. Without it panel B shows the traced path and the reference
  corners only.
- **`green_dists.npy`** is a pickled dict, so `np.load(..., allow_pickle=True)`.
- **`res_tan.json`** holds the tangent-definition ladder (0/30/45/60/75/90°, five seeds per
  rung) and the per-heart traced fractions above 60°; the numbers in Appendix A.4 are read
  from it.

## Analysis that produces numbers rather than figures

| script | what it reports | where it appears |
|---|---|---|
| `analysis/spacing/handcells.py` | cell length and width from the hand-traced 52 hpf outlines | Table 1; §2.1; A.2 |
| `analysis/spacing/raw_width.py` | bundle width on the untraced images: `area_over_len_um` pooled = 1.23 µm at percentile 70; FWHM 1.00 ± 0.09 µm alongside | Table 1; A.7 |
| `analysis/calibration/v13_tune.py` | the two-round Latin-hypercube case lists (±40 %, then ±25 %) | §2.7; A.13, Table 8 |
| `analysis/calibration/multistart13.py` | the six multi-start case lists and basin selection | §3.2; A.13 |
| `analysis/calibration/sens_v13.py` | the ±40 % one-at-a-time sensitivity case list | §3.2; A.13 |
| `analysis/calibration/aspect_sweep13.py` | cell shape at constant cell area (runs the model) | §3.4; Fig. 9 |
| `analysis/convergence/converge13.py` | dt and `n_sub` convergence at fixed rates | §4.1; A.9 |
| `analysis/convergence/conv_probe.py` | attempts, gate failures and acceptances against `n_sub` | A.9 |

Library modules in `pipeline/` that are also worth running directly:
`reference_fields.py` (reference fields of known structure at matched line density, Table 6),
`curved_refs.py` (the curvature control on C2, A.12),
and `nn.py` / `spacing.py` (the spacing measurement, A.5).
