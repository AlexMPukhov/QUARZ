"""compare two field files: max relative differences of selected fields (optionally restricted to xi > ximin)"""
import sys
import numpy as np
sys.path.insert(0, "../tools")
from quarz_read import read_fields
a = read_fields(sys.argv[1] + "/fields_000000.bin"); b = read_fields(sys.argv[2] + "/fields_000000.bin")
label = sys.argv[3] if len(sys.argv) > 3 else ""
ximin = float(sys.argv[4]) if len(sys.argv) > 4 else -1e9
m = a["xi"] > ximin
out = []
for n in ["psi", "ez", "er", "bth", "ne", "rhob"] + [k for k in ("ez_c", "er_c", "ne_c", "rhob_c") if k in a and k in b]:
    den = np.abs(a[n][m]).max()
    out.append("%s %.1e" % (n, np.abs(a[n][m] - b[n][m]).max() / den if den > 0 else 0.0))
print("%-44s" % label, "  ".join(out))
