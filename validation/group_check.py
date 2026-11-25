"""Section 15: a diagnostic group must contain exactly the selected subset of the main output.
usage: group_check.py main_dir group_dir fields rmax xi_stride particle_stride beams
(fields, beams: comma-separated).  Fields: every component of the group file equals the main
file restricted to r <= rmax (plus the first node beyond) and to every xi_stride-th slice; beams:
the group dump equals every particle_stride-th particle of the main dump (serial run)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_fields  # noqa: E402

M, G, fields, rmax, xs, ps, beams = sys.argv[1], sys.argv[2], sys.argv[3].split(","), float(sys.argv[4]), \
    int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split(",")
bad, nf, nb, size_m, size_g = 0, 0, 0, 0, 0
for fg in sorted(glob.glob(G + "/fields_*.bin")):
    fm = M + "/" + os.path.basename(fg)
    a, b = read_fields(fm), read_fields(fg)
    size_m += os.path.getsize(fm); size_g += os.path.getsize(fg)
    jm = int(np.argmax(a["r"] >= rmax * (1 - 1e-12))) + 1 if a["r"][-1] >= rmax else len(a["r"])
    names = [k for k in b if isinstance(b[k], np.ndarray) and b[k].ndim == 2]
    base = sorted({k[:-2] if k.endswith(("_c", "_s")) else k for k in names})
    if base != sorted(fields):
        print("  components", base, "expected", sorted(fields)); bad += 1
    ks = np.arange(len(a["xi"]) - 1, -1, -xs)[::-1]   # every xs-th slice counted from the head (the last row)
    if not np.array_equal(b["r"], a["r"][:jm]) or not np.array_equal(b["xi"], a["xi"][ks]):
        print("  grid of", fg, "wrong"); bad += 1
    for k in names:
        if not np.array_equal(b[k], a[k][ks, :jm]):
            print("  field", k, "of", fg, "differs"); bad += 1
    nf += 1
for bm in beams:
    for fg in sorted(glob.glob(G + f"/beam_{bm}_*.bin")):
        fm = M + "/" + os.path.basename(fg)
        a, b = read_beam(fm), read_beam(fg)
        size_m += os.path.getsize(fm); size_g += os.path.getsize(fg)
        for k in a:
            if not np.array_equal(b[k], a[k][::ps]):
                print("  beam", fg, k, "differs"); bad += 1; break
        nb += 1
others = [f for f in os.listdir(G) if f.startswith("beam_") and not any(f.startswith(f"beam_{x}_") for x in beams)]
if others:
    print("  unselected beams written:", others); bad += 1
print(f"  group '{os.path.basename(G)}': {nf} field files, {nb} beam files = exact subsets of the main output; "
      f"size {size_g / max(size_m, 1) * 100:.1f} % of the corresponding main files  {'OK' if bad == 0 and nf + nb > 0 else 'FAIL'}")
sys.exit(0 if bad == 0 and nf + nb > 0 else 1)
