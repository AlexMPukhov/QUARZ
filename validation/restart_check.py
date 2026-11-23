"""Section 15: compare a restarted run B with the uninterrupted run A.
Every output file of B (fields, axis, beam dumps, in all diagnostic directories) must agree with
A's file of the same step; log files (beams.txt, laser.txt, ionization.txt): the lines of B's steps.
Beam particles are compared after sorting (their order depends on the number of ranks).
usage: restart_check.py A B [tol [verdict]]      (tol = 0: bit-identical)
With "verdict" the line ends with [verdict] and omits the file of the maximum: the reference comparison
(compare_results.py) then checks OK / FAIL against tol, not the number (used where the tolerance depends on the
build, e.g. round-off of a GPU restart)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_fields  # noqa: E402

A, B = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
vmode = len(sys.argv) > 4 and sys.argv[4] == "verdict"


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    s = np.max(np.abs(a))
    return 0.0 if s == 0 and np.all(a == b) else float(np.max(np.abs(a - b)) / (s if s > 0 else 1))


worst, n, where = 0.0, 0, ""
for root, _, files in os.walk(B):
    if "checkpoints" in root:
        continue
    for f in sorted(files):
        pb = os.path.join(root, f)
        pa = os.path.join(A, os.path.relpath(pb, B))
        if f.startswith("fields_") and f.endswith(".bin"):
            a, b = read_fields(pa), read_fields(pb)
            d = max([rel(a[k], b[k]) for k in a if isinstance(a[k], np.ndarray)] + [rel(a["t"], b["t"])])
        elif f.startswith("beam_") and f.endswith(".bin"):
            a, b = read_beam(pa), read_beam(pb)
            ka = np.array([a[k] for k in a]).T   # x y px py pz xi w (+ sx sy sz)
            kb = np.array([b[k] for k in a]).T
            if len(ka) and len(ka) == len(kb):
                ka, kb = ka[np.lexsort(ka.T[::-1])], kb[np.lexsort(kb.T[::-1])]
                # sorting can pair different particles at round-off: compare column-wise sorted values
                if tol > 0:
                    ka, kb = np.sort(ka, axis=0), np.sort(kb, axis=0)
            d = rel(ka, kb)
        elif f.startswith("axis_"):
            d = rel(np.loadtxt(pa, comments="#"), np.loadtxt(pb, comments="#"))
        elif f in ("beams.txt", "laser.txt", "ionization.txt", "timestep.txt", "spin.txt"):
            c = 2 if f in ("beams.txt", "spin.txt") else 1
            la = {tuple(l.split()[:c]): l.split()[c:] for l in open(pa) if l.strip() and l[0] != "#"}
            lb = [l.split() for l in open(pb) if l.strip() and l[0] != "#"]
            d = max([rel([float(x) for x in la[tuple(l[:c])]], [float(x) for x in l[c:]]) for l in lb] + [0.0])
        else:
            continue
        n += 1
        if d > worst:
            worst, where = d, os.path.relpath(pb, B)
ok = n > 0 and worst <= tol
print(f"  {n} output files of the restarted run vs the uninterrupted run: max rel. difference {worst:.2e}"
      f"{' (' + where + ')' if where and not vmode else ''}  {'OK' if ok else 'FAIL'}{' [verdict]' if vmode else ''}")
sys.exit(0 if ok else 1)
