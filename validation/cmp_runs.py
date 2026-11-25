"""Compare all outputs of two QUARZ runs (e.g. serial vs MPI):
fields_*.bin, axis_*.txt, beams.txt, slices_*.txt, beam_*.bin (particles sorted).
Prints the maximum relative difference per file type; exit code 1 if above tol.
Differences are relative to the field scale (a field and its cos/sin mode parts share
one scale) and, for text columns, to max(|column|, 1e-9): components that vanish by
symmetry contain only round-off and are not compared relative to themselves.
Particle dumps are compared column by column after sorting (independent of the order).
usage: python3 cmp_runs.py dirA dirB [tol] [--exact kinds] [--roundoff kinds]
--exact axis,dumps,fields: print one line instead of the values: those kinds must be bit-identical (0), the others
    within tol (OK/FAIL). For checks whose round-off level depends on the platform (e.g. the beam-log moments of
    MPI vs serial: the sums over the ranks are added in another order), so that the reference text does not.
--roundoff fields: (with --exact) those kinds must agree to round-off, within 1e-12 of their scale (for fields the
    maximum of the field), e.g. the fields in the rank-boundary slice of MPI vs serial."""
import glob
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Input line")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_fields  # noqa: E402

args = sys.argv[1:]
exact = None
if "--exact" in args:
    i = args.index("--exact")
    exact = [k for k in args[i + 1].split(",") if k]
    del args[i:i + 2]
roundoff = []
if "--roundoff" in args:
    i = args.index("--roundoff")
    roundoff = [k for k in args[i + 1].split(",") if k]
    del args[i:i + 2]
ROUNDOFF = 1e-12
a_dir, b_dir = args[0], args[1]
tol = float(args[2]) if len(args) > 2 else 1e-10


def rel(x, y, scale=None, floor=0.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        return np.inf
    if not x.size:
        return 0.0
    den = max(np.abs(x).max() if scale is None else scale, floor)
    return float(np.abs(x - y).max() / den) if den > 0 else float(np.abs(x - y).max())


worst = {}


def note(kind, v, what):
    if v > worst.get(kind, (-1, ""))[0]:
        worst[kind] = (v, what)


for fa in sorted(glob.glob(os.path.join(a_dir, "*"))):
    name = os.path.basename(fa)
    fb = os.path.join(b_dir, name)
    if name.endswith(".log") or name == "grid.txt":
        continue
    if not os.path.exists(fb):
        note("missing", np.inf, name)
        continue
    if name.startswith("fields_"):
        A, B = read_fields(fa), read_fields(fb)
        fam = {}
        for k in A:
            if isinstance(A[k], np.ndarray) and A[k].ndim == 2:
                base = k[:-2] if k.endswith(("_c", "_s")) else k
                fam[base] = max(fam.get(base, 0.0), float(np.abs(A[k]).max()))
        for k in A:
            if not (isinstance(A[k], np.ndarray) and A[k].ndim == 2):
                continue
            base = k[:-2] if k.endswith(("_c", "_s")) else k
            note("fields", rel(A[k], B[k], fam[base]), name + ":" + k)
    elif name.startswith("beam_") and name.endswith(".bin"):
        A, B = read_beam(fa), read_beam(fb)
        for k in A:
            note("dumps", rel(np.sort(A[k]), np.sort(B[k]), floor=1e-9), name + ":" + k)
    else:   # text: axis, beams.txt, slices
        A = np.loadtxt(fa, comments="#", ndmin=2, usecols=None, dtype=str)
        B = np.loadtxt(fb, comments="#", ndmin=2, usecols=None, dtype=str)
        if A.shape != B.shape:
            note(name.split("_")[0], np.inf, name + " (shape)")
            continue
        num = [j for j in range(A.shape[1]) if all(c.replace(".", "").replace("e", "").replace("-", "").replace("+", "").isdigit() or c in ("nan", "inf") for c in A[:, j])]
        kind = "axis" if name.startswith("axis") else ("slices" if name.startswith("slices") else "beamlog")
        for j in num:
            scale = None
            if kind == "axis" and j >= 6:   # Wx(0), Wy(0) of mode 1 share one scale
                scale = max(np.abs(A[:, c].astype(float)).max() for c in num if c >= 6)
            note(kind, rel(A[:, j].astype(float), B[:, j].astype(float), scale, floor=1e-9), "%s col %d" % (name, j))
        if any((A[:, j] != B[:, j]).any() for j in range(A.shape[1]) if j not in num):
            note(kind, np.inf, name + " (text columns differ)")

if exact is not None:
    def group(kinds, lim):
        bad = [f"{k} {worst[k][0]:.1e} ({worst[k][1]})" for k in kinds if worst[k][0] > lim]
        return "OK" if not bad else "FAIL " + ", ".join(bad), bool(bad)
    ex = [k for k in sorted(worst) if k in exact]
    ro = [k for k in sorted(worst) if k in roundoff]
    rest = [k for k in sorted(worst) if k not in exact and k not in roundoff]
    missing = [k for k in exact + roundoff if k not in worst]
    t_ex, b_ex = group(ex, 0.0)
    if missing:
        t_ex, b_ex = ("FAIL " if t_ex == "OK" else t_ex + ", ") + ", ".join("no " + m for m in missing), True
    t_ro, b_ro = group(ro, ROUNDOFF)
    t_rest, b_rest = group(rest, tol)
    print("  bit-identical (%s): %s;  " % (" ".join(ex), t_ex)
          + ("round-off, within %g of the maximum (%s): %s;  " % (ROUNDOFF, " ".join(ro), t_ro) if ro else "")
          + "within %g (%s): %s" % (tol, " ".join(rest) or "-", t_rest))
    sys.exit(1 if (b_ex or b_ro or b_rest) else 0)

bad = False
for k, (v, what) in sorted(worst.items()):
    flag = "" if v <= tol else "   <-- above tolerance"
    bad |= v > tol
    print("  %-8s max rel diff %.2e  (%s)%s" % (k, v, what, flag))
sys.exit(1 if bad else 0)
