"""Keep the rendered fields at the v13 tune, for the crossing check and the schematic."""
import numpy as np, time, json
from multiprocessing import Pool
V13 = {"rate_grow": 0.004903568768146313, "rate_nematic_depoly": 0.0063040708394408885, "rate_nematic_poly": 0.0016555393492007537, "rate_branch": 0.0021422015778192914, "rate_nucleate": 0.05903938775253556, "nematic_thresh": 0.35, "angle_noise": 0.7131579655826681, "axis_spread": 0.35946663576674076, "cadherin_nucleation_prob": 0.2659229960288876, "rate_thin": 0.0013049314623728191, "n_sub": 4, "phi_max": None}
def one(seed):
    import sim3
    t0 = time.time(); out, fields = sim3.run_once(V13, seed=seed, keep_fields=True)
    np.savez_compressed(f"v13fields_s{seed}.npz", **{f"h{int(h)}": m for h, m in fields.items()})
    print(f"  seed {seed} {time.time()-t0:.0f}s", flush=True); return seed
if __name__ == "__main__":
    with Pool(2) as p: p.map(one, [7601, 7602, 7603, 7604, 7605, 7606], chunksize=1)
