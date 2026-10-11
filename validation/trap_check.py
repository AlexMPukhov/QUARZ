#!/usr/bin/env python3
"""Trapping of plasma electrons into a beam (section 21).
h OUT N: every trapped electron was born at rest (Delta = gamma - p_z = 1) at the potential psi_b, so
  H = gamma - p_z - psi(r, xi) = 1 - psi_b. In the stationary wake H is conserved, in the quasi-static
  plasma push (as Delta - psi) and, after the conversion, in the beam push. Checks: min H at the
  conversion (step 0) = 1 - max psi (the electrons born at the top of the potential, on the axis
  where the laser is strongest), and H of the electrons converted at step 0 is unchanged at step N.
rate OUT_A N_A DT_A OUT_B N_B DT_B: trapped charge per unit time (weight = plasma weight x dt, the
  conversion at step n covers t_n - dt/2 .. t_n + dt/2: Q / (T + dt)) must not depend on dt.
thr OUT... N: the trapped charge at step N for several pusher.max_qsa_factor (all of these electrons
  are trapped, so the same electrons are converted, only at a different point of their path)."""
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_fields, read_beamlog

NAME = "hel_trapped"


def H(out, step):
    b = read_beam(f"{out}/beam_{NAME}_{step:06d}.bin")
    F = read_fields(f"{out}/fields_{step:06d}.bin")
    r = np.hypot(b["x"], b["y"])
    g = np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)
    rg, xg = F["r"], F["xi"]
    j = np.clip(np.searchsorted(rg, r) - 1, 0, len(rg) - 2)
    tr = (r - rg[j]) / (rg[j + 1] - rg[j])
    dx = xg[1] - xg[0]
    k = np.clip(((b["xi"] - xg[0]) / dx).astype(int), 0, len(xg) - 2)
    tk = (b["xi"] - xg[k]) / dx
    f = F["psi"]
    psi = (1 - tk) * ((1 - tr) * f[k, j] + tr * f[k, j + 1]) + tk * ((1 - tr) * f[k + 1, j] + tr * f[k + 1, j + 1])
    return g - b["pz"] - psi, g, F


mode = sys.argv[1]
if mode == "h":
    out, N = sys.argv[2], int(sys.argv[3])
    h0, g0, F0 = H(out, 0)
    h1, g1, _ = H(out, N)
    n = len(h0)
    d = h1[:n] - h0   # memory order: the electrons converted at step 0 come first (none has left the box)
    print(f" {n} electrons converted per step; at the conversion gamma {np.median(g0):.2f}, H = gamma - p_z - psi in"
          f" [{h0.min():.5f}, {h0.max():.5f}]; 1 - max psi = {1 - F0['psi'].max():.5f}"
          f" (difference {abs(h0.min() - (1 - F0['psi'].max())):.1e})")
    print(f" step {N}: these electrons have gamma {np.median(g1[:n]):.2f}; H changed by rms {np.sqrt(np.mean(d ** 2)):.1e},"
          f" max {np.abs(d).max():.1e}; all {len(h1)} trapped: H in [{h1.min():.5f}, {h1.max():.5f}]")
elif mode == "rate":
    q = []
    for out, N, dt in [(sys.argv[2], int(sys.argv[3]), float(sys.argv[4])), (sys.argv[5], int(sys.argv[6]), float(sys.argv[7]))]:
        L = read_beamlog(f"{out}/beams.txt")[NAME]
        i = list(L["step"]).index(N)
        q.append(L["charge"][i] / (L["t"][i] + dt))
        print(f" dt = {dt:g}: {int(L['alive'][i])} macro-particles, Q/(T + dt) = {q[-1]:.6e}")
    print(f" relative difference {abs(q[0] / q[1] - 1):.1e}")
else:
    N = int(sys.argv[-1])
    for out in sys.argv[2:-1]:
        L = read_beamlog(f"{out}/beams.txt")[NAME]
        i = list(L["step"]).index(N)
        print(f" {out}: Q = {L['charge'][i]:.10e}  ({int(L['alive'][i])} macro-particles)")
