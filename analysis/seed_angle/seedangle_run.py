"""Does removing the nucleation dead band move the fit? A paired null test at v12.

WHAT CHANGED IN THE MODEL.  _seed_angle used to snap the site's firing direction to whichever
of the four lattice directions pointed most nearly at the CELL CENTROID. On the flat top and
bottom membranes the short axis wins that contest wherever |col - cx| < |row - cy| = hex_half_h,
so those sites were assigned a short-axis direction, failed the is_primary filter, and never
nucleated: a dead band exactly one CELL HEIGHT wide, produced by the centroid construction and
not by anything local to the membrane. The rule is now "along the long axis, into the cell",
which is the assumption the old rule was trying to express. 159 -> 222 sites, and all 159 of
the old sites keep exactly the direction they had (seedangle_check.py).

WHY NOT THE LOCAL MEMBRANE NORMAL, which was the obvious fix. Measured, and it fails: the
hexagon's taper e = hex_end_frac * hex_half_w = 18.9 lu is LONGER than hex_half_h = 17.3 lu, so
every membrane segment -- flat and slanted alike -- has an inward normal pointing more across
the short axis than along the long one. Under a normal rule all 222 sites are non-primary and
the model nucleates nothing at all. In an elongated cell the membrane normal is the wrong
quantity: cortical fibres run ALONG the long axis, not perpendicular to the membrane they
start from.

THE CONFOUND, and the arm that removes it. rate_nucleate is per-SITE, so 159 -> 222 raises
total nucleation pressure by 39.6 % on its own. Arm C rescales rate_nucleate by 159/222 to hold
events-per-step fixed, which isolates WHERE the sites are from HOW MANY there are.

PREDICTION, RECORDED BEFORE THE RUN.
  (1) Arm C (matched pressure) will not differ from arm A by more than the six-seed resolution
      of L_dgo, 0.46. Reason: the band was measured not to be empty -- mesh occupancy inside it
      was 0.139 +- 0.107 against 0.071 +- 0.040 outside (8 single-cell runs, paired +0.068,
      p = 0.076), i.e. if anything ENRICHED, because fibres nucleated elsewhere grow through it.
      Filling the band redistributes seeding, it does not add field.
  (2) Arm B (unmatched) will also not differ detectably, though it is the riskier call: the v12
      sensitivity sweep moved L_dgo by 0.182 across a +-40 % rate_nucleate span, so a +39.6 %
      step is worth about 0.09 -- a fifth of the resolution.
  (3) If either arm moves by more than 0.46, the dead band was load-bearing and v12's rates were
      partly compensating for it, which would mean the tune has to be repeated on the fixed code.

usage: python3 seedangle_run.py
"""
from __future__ import annotations

import json, shutil, subprocess, sys

PARTICLE = "bundle_model/cell_particle.py"
ORIG = "cell_particle_ORIG_backup.py"      # the centroid rule, byte-for-byte
FIXED = "cell_particle_FIXED.py"           # the long-axis rule
SEEDS = "6001 6002 6003 6004 6005 6006"

V12 = dict(rate_grow=0.0082531, rate_nematic_depoly=0.0068080, rate_nematic_poly=0.0015298,
           rate_branch=0.0019257, rate_nucleate=0.0788939, nematic_thresh=0.35,
           angle_noise=0.6950862, axis_spread=0.3882413, cadherin_nucleation_prob=0.3486433,
           rate_thin=0.0019649, n_sub=4, phi_max=None)

SCALE = 159.0 / 222.0

if __name__ == "__main__":
    shutil.copy(PARTICLE, FIXED)

    json.dump({"A_centroid_159": V12}, open("SA_cases.json", "w"), indent=1)
    json.dump({"B_longaxis_222": V12,
               "C_longaxis_222_matched": {**V12, "rate_nucleate": V12["rate_nucleate"] * SCALE}},
              open("SBC_cases.json", "w"), indent=1)

    try:
        shutil.copy(ORIG, PARTICLE)
        print("=== ARM A: centroid rule, 159 sites ===", flush=True)
        subprocess.run([sys.executable, "sweep.py", "SA", "SA_cases.json",
                        "--seeds"] + SEEDS.split() + ["--procs", "3"], check=True)

        shutil.copy(FIXED, PARTICLE)
        print("=== ARMS B/C: long-axis rule, 222 sites ===", flush=True)
        subprocess.run([sys.executable, "sweep.py", "SBC", "SBC_cases.json",
                        "--seeds"] + SEEDS.split() + ["--procs", "3"], check=True)
    finally:
        shutil.copy(FIXED, PARTICLE)
        print("cell_particle.py left at the FIXED (long-axis) rule")
