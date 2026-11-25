"""Sections 27 and 29: output files of QUARZ v1.0.0 (xi = t - z) read by this version. The v1 files are written
here, independently of the readers, from the files of a current run (the v1 formats as QUARZ v1.0.0 wrote them):
  fields V3 V2      : fields_*.bin version 3 -> version 2 (version 2, xi = t - z, the head of the box first)
  dump V2 V1        : beam dump with QUARZDMP header -> v1 dump (int32 n, then the records; xi = t - z)
  opmd DUMP T N0 V1 : v1 openPMD file (lab frame z = t - xi_v1 = xi + t, SI with n0 in cm^-3, iteration 5 at
                      the time T in 1/omega_p) holding the particles of the dump as species 'witness'
  cmpfields V3 V2   : quarz_read and the GUI's reader: version 2 vs version 3 (expected identical)
  cmpdump V2 V1     : quarz_read of the v1 dump vs the v2 dump (expected identical, with a warning)
  cmpbeam A B       : max relative difference of two dumps (columns 0-6, relative to the column maximum)"""
import importlib.util
import os
import sys
import warnings

import numpy as np

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
sys.path.insert(0, TOOLS)
from quarz_read import read_beam, read_dump, read_fields  # noqa: E402

C, E, ME, EPS0 = 299792458.0, 1.602176634e-19, 9.1093837015e-31, 8.8541878128e-12


def raw_dump(fn):
    """independent reader of a dump with header: (array, columns)"""
    with open(fn, "rb") as f:
        b = f.read()
    assert b[:8] == b"QUARZDMP"
    nc, n = int(np.frombuffer(b[12:16], np.int32)[0]), int(np.frombuffer(b[16:24], np.int64)[0])
    return np.frombuffer(b[24:], np.float64).reshape(n, nc)


mode = sys.argv[1]
if mode == "fields":
    with open(sys.argv[2], "rb") as f:
        b = f.read()
    ver, M, K, m1 = np.frombuffer(b[:16], np.int32)
    assert ver == 3
    t = b[16:24]
    r = b[24:24 + 8 * M]
    xi = np.frombuffer(b[24 + 8 * M:24 + 8 * M + 8 * K], np.float64)
    pos = 24 + 8 * M + 8 * K
    nc = int(np.frombuffer(b[pos:pos + 4], np.int32)[0])
    out = np.array([2, M, K, m1], np.int32).tobytes() + t + r + (0.0 - xi[::-1]).tobytes() + b[pos:pos + 4]
    pos += 4
    for _ in range(nc):
        name, data = b[pos:pos + 16], np.frombuffer(b[pos + 16:pos + 16 + 8 * M * K], np.float64).reshape(K, M)
        out += name + data[::-1].tobytes()
        pos += 16 + 8 * M * K
    open(sys.argv[3], "wb").write(out)
elif mode == "dump":
    a = raw_dump(sys.argv[2]).copy()
    a[:, 5] = 0.0 - a[:, 5]
    open(sys.argv[3], "wb").write(np.int32(len(a)).tobytes() + a.tobytes())
elif mode == "opmd":
    import openpmd_api as io
    a, T, n0 = raw_dump(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    wp = np.sqrt(n0 * 1e6 * E * E / (EPS0 * ME))
    L = C / wp
    S = io.Series(sys.argv[5], io.Access.create)
    S.set_attribute("quarz_units", "SI")
    S.set_attribute("quarz_coordinates", "z = t - xi (lab frame); xi = t - z is the code's co-moving coordinate")
    it = S.iterations[5]
    it.time = T
    it.time_unit_SI = 1.0 / wp
    sp = it.particles["witness"]
    n = len(a)
    ds = io.Dataset(np.dtype("float64"), [n])

    def put(rc, v, unit):
        rc.reset_dataset(ds)
        rc.unit_SI = unit
        rc.store_chunk(np.ascontiguousarray(v))
    pos = sp["position"]
    put(pos["x"], a[:, 0], L); put(pos["y"], a[:, 1], L); put(pos["z"], T + a[:, 5], L)   # z = t - xi_v1 = t + xi
    off = sp["positionOffset"]
    for c in "xyz":
        off[c].reset_dataset(ds); off[c].unit_SI = L; off[c].make_constant(0.0)
    mom, W = sp["momentum"], a[:, 6] * n0 * 1e6 * L ** 3      # physical particles
    mom.set_attribute("macroWeighted", np.uint32(0)); mom.set_attribute("weightingPower", 0.0)
    for c, k in zip("xyz", (2, 3, 4)):
        put(mom[c], a[:, k], ME * C)                           # m_e c per physical particle (electrons)
    put(sp["weighting"][io.Record_Component.SCALAR], W, 1.0)
    for rec, v in (("charge", -E), ("mass", ME)):
        r = sp[rec][io.Record_Component.SCALAR]; r.reset_dataset(ds); r.unit_SI = 1.0; r.make_constant(v)
    S.flush(); S.close()
elif mode == "cmpfields":
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a2 = read_fields(sys.argv[3])
    a3 = read_fields(sys.argv[2])
    same = a2["t"] == a3["t"] and np.array_equal(a2["xi"], a3["xi"]) and all(
        np.array_equal(a2[k], a3[k]) for k in a3 if isinstance(a3[k], np.ndarray))
    spec = importlib.util.spec_from_file_location("quarz_gui", os.path.join(TOOLS, "quarz_gui.py"))
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    g2, g3 = gui.FieldCache().get(sys.argv[3]), gui.FieldCache().get(sys.argv[2])
    gsame = np.array_equal(g2["xi"], g3["xi"]) and all(np.array_equal(g2["comps"][k], g3["comps"][k]) for k in g3["comps"])
    print(f"  field file version 2 vs version 3: quarz_read {'identical' if same else 'DIFFERENT'} "
          f"(warnings: {len(w)}), GUI {'identical' if gsame else 'DIFFERENT'}")
elif mode == "cmpdump":
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a1 = read_beam(sys.argv[3])
    a2 = read_beam(sys.argv[2])
    same = all(np.array_equal(a1[k], a2[k]) for k in a2) and read_dump(sys.argv[3])[1] and not read_dump(sys.argv[2])[1]
    print(f"{'identical' if same else 'DIFFERENT'} (warnings: {len(w)})")
elif mode == "cmpbeam":
    a, b = raw_dump(sys.argv[2])[:, :7], raw_dump(sys.argv[3])[:, :7]
    d = max(np.abs(a[:, c] - b[:, c]).max() / np.abs(a[:, c]).max() for c in range(7)) if a.shape == b.shape else np.inf
    print(f"{len(b)} of {len(a)} particles, max rel. difference {d:.1e}")
