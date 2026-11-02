"""m = 1 validation by translation invariance.

A driver displaced by d in a uniform plasma produces the displaced axisymmetric
solution F(x - d, y). To first order in d the m = 1 amplitudes are
    scalars:  f_c = -d f0'(r)
    vectors:  V_r,c = -d V_r0',  V_th,s = +d V_r0 / r,  V_th,c = -d V_th0',  V_r,s = -d V_th0 / r
usage: python check_shift.py ref_dir(m=0, centred) shift_dir(m=1, x0 = d) d [sym_dir]
"""
import sys
import numpy as np
sys.path.insert(0, "../tools")
from quarz_read import read_fields

ref = read_fields(sys.argv[1] + "/fields_000000.bin")
sh = read_fields(sys.argv[2] + "/fields_000000.bin")
d = float(sys.argv[3])
r, xi = ref["r"], ref["xi"]
dr = lambda f: np.gradient(f, r, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    over_r = lambda f: np.where(r > 0, f / r, 0.0)

pred = {
    "psi_c": -d * dr(ref["psi"]), "ez_c": -d * dr(ref["ez"]), "ne_c": -d * dr(ref["ne"]),
    "er_c": -d * dr(ref["er"]), "eth_s": d * over_r(ref["er"]),
    "bth_c": -d * dr(ref["bth"]), "br_s": -d * over_r(ref["bth"]),
}
# compare inside the first bucket, away from the sheath / closure singularities
K = (xi > -8.0) & (xi < -0.5)
Rm = (r > 0.02) & (r < 1.0)
print(f"shifted driver, d = {d}: m=1 amplitudes vs -d * d/dr(axisymmetric solution), region xi in (-8,-0.5), r in (0.02,1)")
for name, p in pred.items():
    s = sh[name][np.ix_(K, Rm)]
    q = p[np.ix_(K, Rm)]
    err = np.abs(s - q).max() / np.abs(q).max()
    print(f"  {name:6s}  max|pred| = {np.abs(q).max():.3e}   rel. error = {err:.3e}")
# the sin/cos parts that must vanish, and m=0 unchanged
for name in ["psi_s", "ez_s", "er_s", "bth_s", "bz_c", "bz_s"]:
    print(f"  {name:6s}  max = {np.abs(sh[name][K]).max():.2e}  (should be ~0)")
for name in ["psi", "ez", "er", "bth"]:
    print(f"  m=0 {name:4s} change vs centred run: {np.abs(sh[name][K] - ref[name][K]).max() / np.abs(ref[name][K]).max():.2e}  (O(d^2))")
if len(sys.argv) > 4:
    sym = read_fields(sys.argv[4] + "/fields_000000.bin")
    print("centred driver with modes = 1:")
    for name in ["psi", "ez", "er", "bth", "ne"]:
        print(f"  m=0 {name:4s} vs modes=0 run: {np.abs(sym[name][K] - ref[name][K]).max() / np.abs(ref[name][K]).max():.2e}"
              f"   |{name}_c| max {np.abs(sym[name + '_c'][K]).max():.2e}")
