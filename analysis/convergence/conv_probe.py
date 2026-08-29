"""Direct evidence for WHY total polymer mass is not n_sub-invariant.

polymerise() has TWO monomer conditions, and only one of them is n_sub-free:
    hard gate      avail >= monomer_quantum = monomers_per_seg / n_sub     <- scales with 1/n_sub
    soft rejection accept with probability avail / monomers_per_point      <- n_sub-free
The attempted flux is invariant (n_sub times as many events, each 1/n_sub of a segment), but the
ACCEPTED flux is not: raising n_sub lowers the hard gate, so more attempts survive it once the
local pool is depleted. This probe counts attempts, gate failures and acceptances directly on a
single cell, so the mechanism is measured rather than argued.
"""
import json, numpy as np
import sim3
from modelling.CARMA.carma_6_particle import cell_particle as CP
from modelling.CARMA.carma_6_particle.cell_particle import P

V13 = json.load(open("V13_TUNE.json"))["V13"]
STEPS = 4000

for n in (1, 4, 8, 16):
    sim3.configure({**V13, "n_sub": n})
    np.random.seed(11)
    stats = dict(attempt=0, gate_fail=0, reject=0, ok=0, mass=0.0)
    orig = CP.Simulation.polymerise

    def probe(self, f, _s=stats):
        if not f.frontier:
            return
        col, row = tuple(f.frontier)[np.random.randint(len(f.frontier))]
        avail = max(self.field.pool[row, col], 0.0)
        _s["attempt"] += 1
        if avail < P.monomer_quantum:
            _s["gate_fail"] += 1
            return
        if np.random.random() >= avail / P.monomers_per_point:
            _s["reject"] += 1
            return
        took = f.associate(col, row, avail)
        if took > 0.0:
            self.field.remove([(col, row)], took)
            _s["ok"] += 1
            _s["mass"] += took

    CP.Simulation.polymerise = probe
    s = CP.Simulation()
    for _ in range(STEPS):
        s.kmc_steps()
    CP.Simulation.polymerise = orig
    a = stats["attempt"]
    print(f"n_sub {n:2d}  attempts {a:7d}  gate-fail {stats['gate_fail']/max(a,1):6.1%}  "
          f"reject {stats['reject']/max(a,1):6.1%}  accept {stats['ok']/max(a,1):6.1%}  "
          f"mass moved {stats['mass']/P.monomers_per_seg:8.2f} segments", flush=True)
