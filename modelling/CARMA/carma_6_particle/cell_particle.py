import numpy as np # uses [row, col]
from modelling.CARMA.carma_6_particle.parameters import Params
from modelling.CARMA.carma_6_particle.order_params import analyse

P = Params()

def is_primary(angle): return min((angle - P.mesh_axis) % np.pi, np.pi - (angle - P.mesh_axis) % np.pi) < np.pi / 4


class MonomerField():
    def __init__(self, mask):
        self.mask_coords = np.column_stack(np.nonzero(mask))
        self.pool = np.where(mask, P.monomers_per_point, 0.0).astype(float)
        self.total_init = self.pool.sum()

    def add(self, amount): # fast-diffusion approx
        row, col = self.mask_coords[np.random.randint(0, len(self.mask_coords))]
        self.pool[row, col] += amount

    def remove(self, pts, amount=None): # amount=None -> a full segment per point (bulk-seeded fibres)
        taken = 0.0
        for col, row in pts:
            want = P.monomers_per_seg if amount is None else amount
            got = min(want, max(self.pool[row, col], 0.0)) # clamp: the pool cannot go negative
            self.pool[row, col] -= got
            taken += got
        return taken


class Fibre():
    def __init__(self, base, angle, centre_pts): # angles are in radians
        self.base = base # nucleation point
        self.angle = angle # in radians
        self.centre_pts: list[tuple[int, int]] = centre_pts # the central line
        self.lattice_pts: set[tuple[int, int]] = set() # all potential points
        self.occupied_pts: dict[tuple[int, int], float] = {} # occupied point -> monomers held, in (0, monomers_per_seg]
        self.frontier: set[tuple[int, int]] = set() # cache the neighbouring unoccupied points
        self._primary = is_primary(angle)
    _N4 = ((1, 0), (-1, 0), (0, 1), (0, -1)) # 4 adjacent neighbour pts

    @staticmethod
    def _bresenham(x0, y0, x1, y1): # responsible for the angled fibres. line rasterisation
        pts = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            pts.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    @staticmethod
    def _expand_width(angle, width, cpts): # expand a centre line
        span = range(-(width // 2), width // 2 + (1 if width % 2 else 0))
        offs = [(0, width) for width in span] if is_primary(angle) else [(width, 0) for width in span] 
        return {(cx + dc, cy + dr) for (cx, cy) in cpts for (dc, dr) in offs}

    @property
    def mass(self): return sum(self.occupied_pts.values())

    def associate(self, col, row, avail): # returns the amount actually taken (capped at saturation and at avail)
        held = self.occupied_pts.get((col, row), 0.0)
        take = min(P.monomer_quantum, P.monomers_per_seg - held, max(avail, 0.0))
        if take <= 0.0:
            return 0.0
        self.occupied_pts[(col, row)] = held + take
        self._refresh((col, row))
        return take

    def dissociate(self, col, row): # returns the amount actually released
        held = self.occupied_pts.get((col, row), 0.0)
        give = min(P.monomer_quantum, held)
        if held - give <= 1e-12:
            self.occupied_pts.pop((col, row), None)
        else:
            self.occupied_pts[(col, row)] = held - give
        self._refresh((col, row))
        return give

    def _is_frontier(self, pt): # if unsaturated then stay and add unsaturated new neighbours. if saturated then remove
        col, row = pt
        if pt not in self.lattice_pts or self.occupied_pts.get(pt, 0.0) >= P.monomers_per_seg:
            return False
        return (pt in self.occupied_pts
                or any((col + dc, row + dr) in self.occupied_pts for dc, dr in self._N4))

    def _refresh(self, pt): # only p and its 4 neighbours can change status
        col, row = pt
        for q in ((col, row), (col + 1, row), (col - 1, row), (col, row + 1), (col, row - 1)):
            if self._is_frontier(q):
                self.frontier.add(q)
            else:
                self.frontier.discard(q)


class Simulation():
    def __init__(self):
        self._cx, self._cy = P.W / 2.0, P.H / 2.0 # cache halves
        self.mask = self._make_hex_mask()
        self.field = MonomerField(self.mask)
        self.mesh_axis = P.mesh_axis
        c0, s0 = np.cos(self.mesh_axis), np.sin(self.mesh_axis)
        c9, s9 = np.cos(self.mesh_axis + np.pi / 2), np.sin(self.mesh_axis + np.pi / 2)
        self._qc = {True: (c9, s9, c0, s0), False: (c0, s0, c9, s9)}
        self.time = 0
        self.fibres = {'nematic': [], 'cortex': [], 'mesh': [], 'dead': []} # three families of fibres (+1 for convenience)
        self._lines_cache = {} # PERF: memoised _lines(), invalidated on any mesh mutation
        self.phi = {True: np.random.uniform(0, P.exclusion_len), False: np.random.uniform(0, P.exclusion_len)} # template offset

        self.seed_nematic()
        self.seed_cortex()
        self.seed_sites = self._seed_sites() # fixed nucleation base pool
        self._seed_angle_cache = {s: self._seed_angle(*s) for s in self.seed_sites}
        self.nuc_sites = [s for s in self.seed_sites if is_primary(self._seed_angle_cache[s] + self.mesh_axis)]
        self._nuc_cache = {}
        for _s in self.nuc_sites:
            _a = self._seed_angle_cache[_s]
            _p = is_primary(_a)
            _sc, _sr = self.crystallise(_s[0], _s[1], _p)
            self._nuc_cache[_s] = (_a, _p, _sc, _sr, self._quantise_coords(_sc, _sr, _p)[0])

        self.frames, self.timestamp = [], []
        self.op_time, self.op_nematic, self.op_quartic, self.op_nemfrac = [], [], [], []

    def inside(self, pt): return 0 <= pt[0] < P.W and 0 <= pt[1] < P.H and self.mask[pt[1], pt[0]]

    def _make_hex_mask(self): # hexagonal mask
        e = P.hex_end_frac * P.hex_half_w
        yy, xx = np.ogrid[0:P.H, 0:P.W]
        inset = e * np.abs(yy - self._cy) / P.hex_half_h
        return (np.abs(yy - self._cy) <= P.hex_half_h) & (np.abs(xx - self._cx) <= P.hex_half_w - inset)

    def _seed_sites(self): # fixed pool of nucleation sites, one point inside the cortex centre line
        sites = []
        for f in self.fibres['cortex']:
            ncol, nrow = -np.sin(f.angle), np.cos(f.angle)  # inward normal (vertices wound consistently)
            for (col, row) in f.centre_pts:
                pt = (round(col + ncol), round(row + nrow))
                sites.append(pt)
        return sites

    # Cortical nucleation lays fibres down ALONG THE CELL'S LONG AXIS, growing inward from the
    # membrane. Only the sense (+ or -) is free, and the cell interior fixes it. Snapping over
    # all four lattice directions instead -- the earlier rule -- let the SHORT axis win wherever
    # |row - cy| > |col - cx|, which is the whole middle |col - cx| < hex_half_h of the flat
    # membranes; those sites then failed the is_primary filter and never nucleated. That dead
    # band was one cell-height wide and had no biological content: it came from measuring the
    # direction to the centroid rather than from anything local to the membrane. 222 sites, not
    # 159, and every site the old rule kept fires in exactly the same direction as before.
    def _seed_angle(self, col, row): return 0.0 if col <= self._cx else np.pi

    def _quantise_coords(self, col, row, primary): # gets the quantised line (k) number and position (s) from the column and row
        nc, ns, fc, fs = self._qc[primary] # precomputed cos/sin of the line normals
        dx, dy = col - self._cx, row - self._cy
        perp, along = dx * nc + dy * ns, dx * fc + dy * fs
        return round((perp - self.phi[primary]) / P.exclusion_len), along

    def _lines(self, primary):
        cached = self._lines_cache.get(primary)
        if cached is None:
            cached = {f._k: f for f in self.fibres['mesh'] if f._primary == primary}
            self._lines_cache[primary] = cached
        return cached

    def _spawn_fibre(self, base, angle, width, family, end=None):
        if end is not None: # cortex: explicit segment
            centre = Fibre._bresenham(round(base[0]), round(base[1]), round(end[0]), round(end[1]))
        else: # mesh/nematic: cast inward, clip at the far wall
            reach = 4 * max(P.W, P.H)
            far = (round(base[0] + reach * np.cos(angle)), round(base[1] + reach * np.sin(angle)))
            centre = [pt for pt in Fibre._bresenham(round(base[0]), round(base[1]), *far) if self.inside(pt)]

        f = Fibre(base, angle, centre)
        f.lattice_pts = {pt for pt in Fibre._expand_width(angle, width, centre) if self.inside(pt)}
        if family == 'mesh':
            f._k = self._quantise_coords(base[0], base[1], f._primary)[0]   # PERF: fixed for life
            self._lines_cache = {}
        self.fibres[family].append(f)
        return f

    def _is_occupied(self, lines, k, gap=0):
        if gap == 0:
            return False
        return any(abs(k - taken) <= gap for taken in lines if taken != k)

    def seed_nematic(self): # one horizontal nematic array at randomised gaps. In need to future monomer check
        ys = np.nonzero(self.mask)[0]
        row = ys.min() + np.random.randint(P.nematic_gap_min // 2, P.nematic_gap_min)

        while row < ys.max() - P.nematic_gap_min // 2:
            cols = np.nonzero(self.mask[row])[0]
            f = self._spawn_fibre(base=(cols.min(), row), angle=0.0 + np.random.uniform(-P.angle_noise, P.angle_noise), width=P.fibre_width, family="nematic")
            f.occupied_pts = {pt: got for pt in f.lattice_pts if (got := self.field.remove([pt])) > 0.0} # nematic fibre are full
            for pt in f.occupied_pts:
                f._refresh(pt)
            row += np.random.randint(P.nematic_gap_min, P.nematic_gap_max + 1) # +1: numpy upper bound is EXCLUSIVE
        self.total_nem = max(len(self.fibres['nematic']), 1)

    def seed_cortex(self): # cortex around the cell wall. Depolymerises to width = 1. Need monomer check
        a, b = P.hex_half_w, P.hex_half_h
        e = P.hex_end_frac * a
        verts = [(self._cx - a, self._cy), (self._cx - a + e, self._cy - b), (self._cx + a - e, self._cy - b),
                 (self._cx + a, self._cy), (self._cx + a - e, self._cy + b), (self._cx - a + e, self._cy + b)]

        for i in range(len(verts)):
            (bx, by), (ex, ey) = verts[i], verts[(i + 1) % len(verts)]
            f = self._spawn_fibre(base=(bx, by), angle=np.arctan2(ey - by, ex - bx), width=P.fibre_width * 2, family="cortex", end=(ex, ey))
            f.occupied_pts = {pt: got for pt in f.lattice_pts if (got := self.field.remove([pt])) > 0.0}
            for pt in f.occupied_pts:
                f._refresh(pt)

    def polymerise(self, f):
        if f.frontier:
            col, row = tuple(f.frontier)[np.random.randint(len(f.frontier))]
            avail = max(self.field.pool[row, col], 0.0)
            if avail >= P.monomer_quantum and np.random.random() < avail / P.monomers_per_point: # hard availability gate
                took = f.associate(col, row, avail)
                if took > 0.0:
                    self.field.remove([(col, row)], took) # remove takes a list of tuples

    def depolymerise(self, f):
        if not f.occupied_pts:
            self.fibres['dead'].append(f)
            return
        col, row = tuple(f.occupied_pts)[np.random.randint(len(f.occupied_pts))] # pick random col, row from the occupied fibre
        self.field.add(f.dissociate(col, row))
        if not f.occupied_pts: # emptied here, not on some later event: it has no frontier and cannot regrow
            self.fibres['dead'].append(f)

    def nucleate(self, site=None): # nucleation at the boundaries
        col, row = site if site is not None else self.seed_sites[np.random.randint(len(self.seed_sites))]
        seed = self._nuc_cache.get((col, row)) # PERF: memoised angle/primary/snap/k
        if seed is not None:
            angle, primary, scol, srow, k = seed
        else:
            angle = self._seed_angle_cache.get((col, row))
            if angle is None:
                angle = self._seed_angle(col, row)
            primary = is_primary(angle)
            scol, srow = self.crystallise(col, row, primary)
            k, _ = self._quantise_coords(scol, srow, primary)
        lines = self._lines(primary)
        owner = lines.get(k)

        if owner is not None: # second fibre on an existing line
            if (round(scol), round(srow)) in owner.lattice_pts and\
                (round(scol), round(srow)) not in owner.occupied_pts and\
                    self.field.pool[srow, scol] >= P.monomer_quantum:
                self.field.remove([(scol, srow)], owner.associate(scol, srow, self.field.pool[srow, scol])) # a new growth front
        elif self.inside((scol, srow)) and not self._is_occupied(lines, k, gap=P.seed_gap) and\
            self.field.pool[srow, scol] >= P.monomer_quantum: # regular nucleation
            f = self._spawn_fibre(base=(scol, srow), angle=angle + self.mesh_axis, width=P.fibre_width, family="mesh")
            self.field.remove([(scol, srow)], f.associate(scol, srow, self.field.pool[srow, scol]))

    def cadherin(self, col, row, angle, mesh_noise): # cadherin cross-cell nucleation (superfibre)
        if not self.inside((col, row)) or self.field.pool[row, col] < P.monomer_quantum:
            return
        if any((col, row) in f.occupied_pts for f in self.fibres['mesh']): # do not stack on an occupied pixel
            return
        prev = P.mesh_axis
        P.mesh_axis = mesh_noise # rasterise/classify the superfibre in the neighbour's frame
        f = self._spawn_fibre(base=(col, row), angle=angle, width=P.fibre_width, family="mesh")
        self.field.remove([(col, row)], f.associate(col, row, self.field.pool[row, col]))
        P.mesh_axis = prev

    def branch(self, m): # daughter pair at +-90 deg with enforced spacing, exactly as in minimal
        primary = not is_primary(m.angle)
        lines = self._lines(primary)
        coords = []
        for col, row in m.centre_pts:
            if (col, row) not in m.occupied_pts: # the one change: no branching off bare lattice
                continue
            scol, srow = self.crystallise(col, row, primary)
            k, _ = self._quantise_coords(scol, srow, primary)
            if self.inside((scol, srow)) and not (k in lines or self._is_occupied(lines, k, gap=P.seed_gap)):
                coords.append((scol, srow)) # no _viable(): carma_particle has no L0 length gate
        if not coords:
            return
        scol, srow = coords[np.random.randint(len(coords))]
        if self.field.pool[srow, scol] >= P.monomer_quantum * 2:
            for angle in (m.angle - np.pi / 2, m.angle + np.pi / 2):
                f = self._spawn_fibre(base=(scol, srow), angle=angle, width=P.fibre_width, family="mesh")
                self.field.remove([(scol, srow)], f.associate(scol, srow, self.field.pool[srow, scol]))

    def crystallise(self, col, row, primary): # snap to template
        normal = self.mesh_axis + (np.pi / 2 if primary else 0.0)
        nc, ns = np.cos(normal), np.sin(normal)
        k, _ = self._quantise_coords(col, row, primary)
        d = (self.phi[primary] + k * P.exclusion_len) - ((col - (P.W / 2.0)) * nc + (row - (P.H / 2.0)) * ns)
        return round(col + d * nc), round(row + d * ns)

    def kmc_steps(self):
        events = []
        def add(fibres, k, action):
            if not fibres or k <= 0:
                return
            counts = np.random.poisson(k, size=len(fibres))
            events.extend((f, action) for f, n in zip(fibres, counts) for _ in range(n))

        sites = self.nuc_sites # PERF: cached in __init__, identical content
        add(sites, P.k_nucleate, self.nucleate) # UNGATED: replacements assemble WHILE the seeded array clears
        mesh, nem, cortex = self.fibres['mesh'], self.fibres['nematic'], self.fibres['cortex']
        if len(nem) / self.total_nem < P.nematic_thresh:
            add(mesh, P.k_branch, self.branch)
        add(mesh, P.k_decay, self.depolymerise)
        add(mesh, P.k_grow, self.polymerise)
        add(nem, P.k_nematic_depoly, self.depolymerise)
        add(nem, P.k_nematic_poly, self.polymerise)
        add(cortex, P.k_nematic_depoly * 0.9, self.depolymerise)

        np.random.shuffle(events)
        dead, n_seen = set(), 0
        for f, action in events:
            if f not in dead:
                action(f)
                if len(self.fibres['dead']) != n_seen: # only rebuild when something died
                    dead, n_seen = set(self.fibres['dead']), len(self.fibres['dead'])

        self.time += 1
        dead = set(self.fibres['dead'])
        if dead:
            self.fibres = {k: [f for f in v if f not in dead] for k, v in self.fibres.items()}
            self._lines_cache = {} # PERF: only rebuild when the mesh list changed
        self.fibres['dead'] = []

    def check_conservation(self): # final monomer check
        stored = sum(f.mass for family in self.fibres.values() for f in family)
        total = self.field.pool.sum() + stored
        drift = abs(total - self.field.total_init) / self.field.total_init * 100
        print(f"Conservation: total: {total:.1f}  init: {self.field.total_init:.0f}  drift: {drift:.2f}%")
    
    def order_parameters(self):
            ang, wts = [], []
            for f in self.fibres['mesh'] + self.fibres['nematic']:
                if f.occupied_pts:
                    ang.append(f.angle)
                    wts.append(f.mass)
            return analyse(np.array(ang), np.array(wts))