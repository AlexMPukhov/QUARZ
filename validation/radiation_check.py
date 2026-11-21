"""Section 26: stochastic radiation reaction of a matched electron beam in a pure ion channel (n_i = 1).
usage: radiation_check.py N0_CM3 STEP CLASSICAL STOCHASTIC GAUSSIAN   (output directories)

Independent reference: RK4 (1000 steps, omega_beta h = 0.016) of every particle's orbit with the Landau-Lifshitz drag in the channel field
E_r = r/2 (w = r^2/4 for v_r << 1), K = hbar omega_p / (m c^2):
   mean loss        d<gamma>/dt       = (2/3) alpha K gamma^2 r^2/4                        (Landau-Lifshitz)
   quantum spread   d Var(gamma)/dt   = (55 / (24 sqrt 3)) alpha chi^3 gamma / K,  chi = K gamma r/2   (Sokolov-Ternov)
Checks: the classical run and the stochastic run lose the predicted mean energy; the stochastic minus the classical
energy of each particle has the predicted variance (the orbits are the same, the particles keep their order);
H = gamma + r^2/4 (conserved without radiation) never increases with the stochastic recoil, but does for the
Gaussian (Fokker-Planck) model."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam  # noqa: E402

ALPHA = 7.2973525693e-3
n0, step = float(sys.argv[1]), int(sys.argv[2])
dirs = sys.argv[3:6]
K = 5.64146e4 * np.sqrt(n0) / 2.99792458e10 * 3.8615926796e-11
b0 = read_beam(f"{dirs[0]}/beam_witness_{0:06d}.bin")
T = None
for line in open(f"{dirs[0]}/beams.txt"):
    if not line.startswith("#"):
        T = float(line.split()[2])    # t of the last line
live = b0["w"] > 0
g0 = np.sqrt(1 + b0["px"] ** 2 + b0["py"] ** 2 + b0["pz"] ** 2)
w = 1 / np.sqrt(2 * g0)
# RK4 of the Landau-Lifshitz orbits (electron, E = (x, y)/2, B = 0): du/dt = -E - nu u, nu = rr gamma w, rr = (2/3) alpha K
rr = 2 / 3 * ALPHA * K
X = np.array([b0["x"], b0["y"], b0["px"], b0["py"], b0["pz"], np.zeros_like(g0), np.zeros_like(g0), np.zeros_like(g0)])


def rhs(s):
    x, y, ux, uy, uz = s[:5]
    g = np.sqrt(1 + ux * ux + uy * uy + uz * uz)
    ex, ey = x / 2, y / 2
    ve = (ux * ex + uy * ey) / g
    w = np.maximum(ex * ex + ey * ey - ve * ve, 0)
    nu = rr * g * w
    chi = K * g * np.sqrt(w)
    return np.array([ux / g, uy / g, -ex - nu * ux, -ey - nu * uy, -nu * uz,
                     55 / (24 * np.sqrt(3)) * ALPHA * chi ** 3 * g / K,     # quantum variance
                     5 / (2 * np.sqrt(3)) * ALPHA * chi / (g * K),          # photon number
                     chi])


nst = 1000
h = T / nst
chimax = 0.0
for _ in range(nst):
    k1 = rhs(X); k2 = rhs(X + h / 2 * k1); k3 = rhs(X + h / 2 * k2); k4 = rhs(X + h * k3)
    X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    chimax = max(chimax, (K * np.sqrt(1 + X[2] ** 2 + X[3] ** 2 + X[4] ** 2) * np.hypot(X[0], X[1]) / 2).max())
var, nph, chisum = X[5], X[6], X[7].sum() / T
gT = np.sqrt(1 + X[2] ** 2 + X[3] ** 2 + X[4] ** 2)
H0 = g0 + (b0["x"] ** 2 + b0["y"] ** 2) / 4
loss = H0 - (gT + (X[0] ** 2 + X[1] ** 2) / 4)
print(f"  gamma0 = {g0.mean():.4g}, T = {T:g}: chi max {chimax:.2e}, mean {chisum / len(g0):.2e};"
      f" photons per particle {nph.mean():.2f}")
res = {}
for name, d in zip(["classical", "stochastic", "gaussian"], dirs):
    b = read_beam(f"{d}/beam_witness_{step:06d}.bin")
    ok = live & (b["w"] > 0)
    g = np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)
    H = g + (b["x"] ** 2 + b["y"] ** 2) / 4
    res[name] = (g, H, ok)
mpred, err = loss[live].mean(), None
for name in ["classical", "stochastic", "gaussian"]:
    g, H, ok = res[name]
    dl = (H0 - H)[ok]
    se = np.sqrt(var[ok].mean() / ok.sum()) if name != "classical" else 0
    gain = np.mean(H[ok] > H0[ok] + 1e-6 * H0[ok])
    print(f"  {name:10s}: mean loss {dl.mean():8.2f}  (LL orbits {mpred:8.2f}, ratio {dl.mean() / mpred:.4f}"
          + (f", statistical error {se / mpred:.4f})" if se else ")") + f"; particles that gained energy: {gain:.4f}")
gc, _, okc = res["classical"]
for name in ["stochastic", "gaussian"]:
    g, _, ok = res[name]
    m = ok & okc
    d = g[m] - gc[m]
    v = d.var()
    # statistical error of a variance: sqrt((mu4 - v^2)/N)
    ev = np.sqrt(max(np.mean((d - d.mean()) ** 4) - v * v, 0) / m.sum())
    print(f"  {name:10s}: Var(gamma - gamma_classical) = {v:9.4g}  (Sokolov-Ternov along the LL orbits {var[m].mean():9.4g},"
          f" ratio {v / var[m].mean():.4f} +- {ev / var[m].mean():.4f})")
