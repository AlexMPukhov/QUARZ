"""Section 28: beam import from openPMD files of other codes (synthetic files in their conventions).
write DIR       : DIR/foreign_si_%06T.h5 (iteration 7) and DIR/foreign_norm_%06T.h5 (iteration 3), plus the
                  expected beams in code units (DIR/expected_*.npy), computed here independently:
  si   : SI units, co-moving z with an origin moved into positionOffset (position z = xi - 1 mm,
         positionOffset z = 1 mm), momentum per macro-particle (macroWeighted = 1, weightingPower = 1),
         charge and mass as constant records, one component in float32; deck n0 = 1e17 cm^-3,
         openpmd_xi_shift = -0.5;
  norm : normalised units (unitSI = 1) referring to n0 = 4e17 cm^-3 (deck: 1e17): lengths and weights
         rescaled, momentum in m_e c per physical particle; species 'pbar' (mass 1836.15267343, charge -1).
compare DUMP EXPECTED : max relative difference per column of the imported beam (written again at step 0)
charge DUMP_A DUMP_B  : total weight of A / total weight of B"""
import os
import sys

import numpy as np

C, E, ME, EPS0 = 299792458.0, 1.602176634e-19, 9.1093837015e-31, 8.8541878128e-12


def skin(n0_cm3):
    return C / np.sqrt(n0_cm3 * 1e6 * E * E / (EPS0 * ME))


def write(d):
    import openpmd_api as io
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 3000
    # ---- SI file of another code (n0 = 1e17 cm^-3 in the deck)
    nd = 1e17
    L = skin(nd)
    xi = rng.uniform(-3.0, -1.0, n) * L                     # true co-moving position, m
    x, y = rng.normal(0, 0.05 * L, n), rng.normal(0, 0.05 * L, n)
    u = np.stack([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(400, 4, n)])   # p / (m_e c)
    W = rng.uniform(1e4, 5e4, n)                            # physical electrons per macro-particle
    puy32 = (u[1] * ME * C * W).astype(np.float32)          # p_y per macro-particle, float32
    S = io.Series(f"{d}/foreign_si_%06T.h5", io.Access.create)
    it = S.iterations[7]
    it.time = 1.0e-9
    sp = it.particles["beam"]
    ds = io.Dataset(np.dtype("float64"), [n])
    ds32 = io.Dataset(np.dtype("float32"), [n])

    def comp(rc, v, unit, d32=False):
        rc.reset_dataset(ds32 if d32 else ds)
        rc.unit_SI = unit
        rc.store_chunk(np.ascontiguousarray(v))
    pos = sp["position"]
    comp(pos["x"], x, 1.0); comp(pos["y"], y, 1.0); comp(pos["z"], xi - 1e-3, 1.0)
    off = sp["positionOffset"]
    for c, v in (("x", 0.0), ("y", 0.0), ("z", 1e-3)):
        off[c].reset_dataset(ds); off[c].unit_SI = 1.0; off[c].make_constant(v)
    mom = sp["momentum"]
    mom.set_attribute("macroWeighted", np.uint32(1)); mom.set_attribute("weightingPower", 1.0)
    comp(mom["x"], u[0] * ME * C * W, 1.0); comp(mom["y"], puy32, 1.0, True); comp(mom["z"], u[2] * ME * C * W, 1.0)
    w = sp["weighting"][io.Record_Component.SCALAR]
    comp(w, W, 1.0)
    for rec, v in (("charge", -E), ("mass", ME)):
        r = sp[rec][io.Record_Component.SCALAR]; r.reset_dataset(ds); r.unit_SI = 1.0; r.make_constant(v)
    S.flush(); S.close()
    shift = -0.5
    uy = puy32.astype(np.float64) / (ME * C) / W
    np.save(f"{d}/expected_si.npy", np.stack([x / L, y / L, u[0], uy, u[2], xi / L + shift, W / (nd * 1e6 * L ** 3)], 1))
    # ---- normalised file, normalised to n_file = 4e17 cm^-3; antiprotons
    nf, m = 4e17, 1836.15267343
    Lf = skin(nf)
    xn, yn = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)    # c/omega_p of n_file
    zn = rng.uniform(-6.0, -2.0, n)
    pn = np.stack([rng.normal(0, 50, n), rng.normal(0, 50, n), rng.normal(5e5, 1e3, n)])   # m_e c per particle
    wn = rng.uniform(0.5, 2.0, n) * 1e-6                     # n_file (c/omega_p,file)^3
    S = io.Series(f"{d}/foreign_norm_%06T.h5", io.Access.create)
    sp = S.iterations[3].particles["pbar"]
    pos = sp["position"]
    comp(pos["x"], xn, 1.0); comp(pos["y"], yn, 1.0); comp(pos["z"], zn, 1.0)
    mom = sp["momentum"]
    comp(mom["x"], pn[0], 1.0); comp(mom["y"], pn[1], 1.0); comp(mom["z"], pn[2], 1.0)
    comp(sp["weighting"][io.Record_Component.SCALAR], wn, 1.0)
    for rec, v in (("charge", -1.0), ("mass", m)):
        r = sp[rec][io.Record_Component.SCALAR]; r.reset_dataset(ds); r.unit_SI = 1.0; r.make_constant(v)
    S.flush(); S.close()
    f = Lf / L
    np.save(f"{d}/expected_norm.npy", np.stack([xn * f, yn * f, pn[0] / m, pn[1] / m, pn[2] / m, zn * f,
                                                wn * nf * Lf ** 3 / (nd * L ** 3)], 1))


def dump(fn):
    with open(fn, "rb") as f:
        n = int(np.frombuffer(f.read(4), np.int32)[0])
        a = np.frombuffer(f.read(), np.float64)
    return a.reshape(n, -1)


if sys.argv[1] == "write":
    write(sys.argv[2])
elif sys.argv[1] == "compare":
    a, e = dump(sys.argv[2])[:, :7], np.load(sys.argv[3])
    ia, ie = np.lexsort(a.T[::-1]), np.lexsort(e.T[::-1])   # the import keeps the order; sorted to be safe
    a, e = a[ia], e[ie]
    names = ["x", "y", "px", "py", "pz", "xi", "w"]
    d = [np.abs(a[:, c] - e[:, c]).max() / np.abs(e[:, c]).max() for c in range(7)] if a.shape == e.shape else None
    print(f"  {os.path.basename(sys.argv[3])[9:-4]}: {len(a)} of {len(e)} particles; max rel. difference: " +
          (", ".join(f"{k} {v:.1e}" for k, v in zip(names, d)) if d else "SHAPE MISMATCH"))
else:
    a, b = dump(sys.argv[2]), dump(sys.argv[3])
    print(f"{a[:, 6].sum() / b[:, 6].sum():.4f}")
