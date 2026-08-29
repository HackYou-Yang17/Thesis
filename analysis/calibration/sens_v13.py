"""Per-parameter sensitivity at the v13 tune, on the SAME axes and the SAME span as the
multi-start probe (+-40 %), so the two can be read against each other in one figure.

WHY +-40 % AND NOT THE USUAL x0.5-x2. The multi-start probed a +-40 % hypercube; matching the
span means panel B (how far does the fit move when this parameter moves alone) and panel C
(do two tied configurations agree on this parameter) are describing displacements of the same
size. A sensitivity ranking is a property of the point AND the step size it was measured at.

nematic_thresh is a LEVEL, not a continuum, so it is moved to its two other swept values
(0.20 and 0.60) rather than scaled.

Control = the v13 tune. It is run here TOO (case S_control) rather than borrowed from the
tuning round, so every point in the sweep shares one seed set, 7601-7606.
"""
import json
V12 = {"rate_grow": 0.004903568768146313, "rate_nematic_depoly": 0.0063040708394408885, "rate_nematic_poly": 0.0016555393492007537, "rate_branch": 0.0021422015778192914, "rate_nucleate": 0.05903938775253556, "nematic_thresh": 0.35, "angle_noise": 0.7131579655826681, "axis_spread": 0.35946663576674076, "cadherin_nucleation_prob": 0.2659229960288876, "rate_thin": 0.0013049314623728191, "n_sub": 4, "phi_max": None}
SPAN = 0.40
SCALED = ["rate_grow", "rate_nematic_depoly", "rate_nematic_poly", "rate_branch",
          "rate_nucleate", "angle_noise", "axis_spread", "cadherin_nucleation_prob"]
LIM = dict(angle_noise=0.785398, axis_spread=0.785398, cadherin_nucleation_prob=1.0)
cases = {"S_control": dict(V12)}
for k in SCALED:
    for d, tag in ((1 + SPAN, "hi"), (1 - SPAN, "lo")):
        c = dict(V12); c[k] = float(V12[k] * d)
        if k in LIM: c[k] = min(c[k], LIM[k])
        if k == "rate_grow": c["rate_thin"] = float(V12["rate_thin"] * d)   # hold thin:grow
        cases[f"S_{k}_{tag}"] = c
# thin:grow moved on its own, at fixed growth
for d, tag in ((1 + SPAN, "hi"), (1 - SPAN, "lo")):
    c = dict(V12); c["rate_thin"] = float(V12["rate_thin"] * d)
    cases[f"S_thin_grow_{tag}"] = c
for v, tag in ((0.60, "hi"), (0.20, "lo")):
    c = dict(V12); c["nematic_thresh"] = v
    cases[f"S_nematic_thresh_{tag}"] = c
json.dump(cases, open("SENSV13_cases.json", "w"), indent=1)
print(len(cases), "configs -> SENSV13_cases.json")
