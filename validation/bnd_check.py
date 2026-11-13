"""Section 19: two-stage runs through the rear boundary.
fields REF STAGE2 STEP_REF STEP_S2 : max relative difference of psi, E_z, E_r, B_theta, n_e between
    the stage-2 box and the same xi range of a single run (REF), at the given steps
witness REF RUN [RUN...] : final <gamma>, r_rms, eps_n of the witness relative to REF"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beamlog, read_fields  # noqa: E402

if sys.argv[1] == "fields":
    a = read_fields(f"{sys.argv[2]}/fields_{int(sys.argv[4]):06d}.bin")
    b = read_fields(f"{sys.argv[3]}/fields_{int(sys.argv[5]):06d}.bin")
    # the stage-2 box lies behind stage 1: the first rows (smallest xi) of the single run
    k1 = int(np.argmin(np.abs(a["xi"] - b["xi"][-1]))) + 1
    assert k1 == len(b["xi"]) and abs(a["t"] - b["t"]) < 1e-9
    w = max(float(np.max(np.abs(a[c][:k1] - b[c])) / max(np.max(np.abs(a[c][:k1])), 1e-300))
            for c in ("psi", "ez", "er", "bth", "ne"))
    print(f"  fields of the stage-2 box (xi <= {b['xi'][-1]:g}) vs the single run, t = {b['t']:g}: max rel. difference {w:.1e}")
else:
    R = read_beamlog(f"{sys.argv[2]}/beams.txt")["witness"]
    for o in sys.argv[3:]:
        B = read_beamlog(f"{o}/beams.txt")["witness"]
        d = [abs(B[k][-1] / R[k][-1] - 1) for k in ("gamma_mean", "r_rms", "emit_nx")]
        print(f"  {o:10s} t = {B['t'][-1]:g}: witness <gamma> {B['gamma_mean'][-1]:.4f} (rel. diff {d[0]:.1e}), "
              f"r_rms {d[1]:.1e}, eps_n {d[2]:.1e}")
