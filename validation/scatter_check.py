"""Section 18: multiple Coulomb scattering of beam particles on the plasma.
free OUT STEP T LI LE Z ZETA FACTOR : weak zero-emittance witness in a uniform plasma (n_e = n_i = 1):
    <p_x^2> = <p_y^2> = D t, D = k_p r_e factor (Z^2 n_i L_i + n_e L_e + (Z - zeta) n_i L_e); |p| conserved.
channel OUT GAMMA LI FACTOR : matched witness in a pure ion channel (n_i = 1, Z = 1):
    d eps_n/dt = D / sqrt(2 gamma), D = k_p r_e factor L_i (Kirby et al., PAC 2007). Needs the particle dumps of the
    first and last step: the statistical error comes from the spread over the particles of the growth of the betatron
    action J = (gamma k_b x^2 + p_x^2 / (gamma k_b)) / 2, k_b^2 = 1 / (2 gamma), whose mean is eps_n (conservative for
    the slope fitted to the beam log, which also averages over time).
The random kicks hash the particle state, so other compilers or machines draw other samples: every line ends with a
verdict (within 3 statistical errors of the theory) and the marker [verdict]; compare_results.py compares the
verdict, not the numbers."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_beamlog  # noqa: E402


def verdict(ok):
    return f"  {'OK' if ok else 'FAIL'} [verdict]"


N0 = 1e17
KAPPA = 5.64146e4 * np.sqrt(N0) / 2.99792458e10 * 2.8179403262e-13
mode = sys.argv[1]
if mode == "free":
    out, step, T, Li, Le, Z, zeta, fac = sys.argv[2], int(sys.argv[3]), *map(float, sys.argv[4:10])
    b = read_beam(f"{out}/beam_witness_{step:06d}.bin")
    w = b["w"] > 0
    D = KAPPA * fac * (Z * Z * Li + Le + (Z - zeta) * Le)
    vx, vy = np.var(b["px"][w]) / (D * T), np.var(b["py"][w]) / (D * T)
    g = np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)[w]
    err, dg = np.sqrt(2 / w.sum()), np.max(np.abs(g / g.mean() - 1))
    print(f"  Z = {Z:g}, ion charge {zeta:g}: <p_x^2>/(D t) = {vx:.4f}, <p_y^2>/(D t) = {vy:.4f}"
          f" ({w.sum()} particles, statistical error {err:.4f}); gamma conserved to {dg:.0e}"
          + verdict(abs(vx - 1) <= 3 * err and abs(vy - 1) <= 3 * err and dg < 1e-10))
elif mode == "channel":
    out, gam, Li, fac = sys.argv[2], *map(float, sys.argv[3:6])
    b = read_beamlog(f"{out}/beams.txt")["witness"]
    t, e = b["t"], 0.5 * (b["emit_nx"] + b["emit_ny"])
    slope = np.polyfit(t, e, 1)[0]
    th = KAPPA * fac * Li / np.sqrt(2 * gam)
    dumps = sorted(glob.glob(f"{out}/beam_witness_*.bin"))
    b0, b1 = read_beam(dumps[0]), read_beam(dumps[-1])
    kb = 1 / np.sqrt(2 * gam)

    def action(b, c, p):
        g = np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)
        return 0.5 * (g * kb * b[c] ** 2 + b[p] ** 2 / (g * kb))
    dJ = 0.5 * (action(b1, "x", "px") - action(b0, "x", "px") + action(b1, "y", "py") - action(b0, "y", "py"))
    err = dJ.std() / np.sqrt(dJ.size) / (t[-1] - t[0]) / th
    print(f"  matched witness, gamma {gam:g}: d eps_n/dt = {slope:.4e}, theory D/sqrt(2 gamma) = {th:.4e},"
          f" ratio {slope / th:.4f}, statistical error {err:.4f}  (eps_n {e[0]:.5f} -> {e[-1]:.5f})"
          + verdict(abs(slope / th - 1) <= 3 * err))
