"""Section 26: stochastic radiation reaction of a matched electron beam in a pure ion channel (n_i = 1).
usage: radiation_check.py N0_CM3 STEP CLASSICAL STOCHASTIC GAUSSIAN   (output directories)

Independent reference: RK4 (1000 steps, omega_beta h = 0.016) of every particle's orbit with the Landau-Lifshitz drag in the channel field
E_r = r/2 (w = r^2/4 for v_r << 1), K = hbar omega_p / (m c^2):
   mean loss        d<gamma>/dt       = (2/3) alpha K gamma^2 r^2/4                        (Landau-Lifshitz)
   quantum spread   D = d Var(gamma)/dt = (55 / (24 sqrt 3)) alpha chi^3 gamma / K,  chi = K gamma r/2   (Sokolov-Ternov)
The spread is damped: a particle that lost more energy than the mean radiates less afterwards (P ~ gamma^2 r^2) and
its betatron motion changes. The expected variance therefore follows the linearized motion about each LL orbit:
the covariance C of (x, y, u) obeys dC/dt = A C + C A^T + D n n^T (A = Jacobian of the LL equations, n = du/dgamma =
gamma u / u^2: the recoil is along u), integrated with the orbit; Var(gamma) = g^T C g, g = dgamma/du = u / gamma.
For this deck damping lowers it by about 2 % below the plain integral of D (an independent Fokker-Planck Monte Carlo
along the same orbits gives 0.981 +- 0.004 of the plain integral).
Checks: the classical run and the stochastic run lose the predicted mean energy; the stochastic minus the classical
energy of each particle has the predicted (damped) variance (the orbits are the same, the particles keep their order);
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


def jac(s):
    """Jacobian A = d(dx/dt, dy/dt, du/dt) / d(x, y, ux, uy, uz) of the LL equations (shape 5 x 5 x N)."""
    x, y, ux, uy, uz = s[:5]
    u = (ux, uy, uz)
    g = np.sqrt(1 + ux * ux + uy * uy + uz * uz)
    ex, ey = x / 2, y / 2
    ve = (ux * ex + uy * ey) / g
    w0 = ex * ex + ey * ey - ve * ve
    on = w0 > 0
    w = np.where(on, w0, 0)
    nu = rr * g * w
    z = np.zeros_like(g)
    dg = [z, z, ux / g, uy / g, uz / g]
    dve = [ux / (2 * g), uy / (2 * g), ex / g - ve * ux / g ** 2, ey / g - ve * uy / g ** 2, -ve * uz / g ** 2]
    dw = [np.where(on, ex - 2 * ve * dve[0], 0), np.where(on, ey - 2 * ve * dve[1], 0)] + \
         [np.where(on, -2 * ve * dve[k], 0) for k in (2, 3, 4)]
    dnu = [rr * (dg[j] * w + g * dw[j]) for j in range(5)]
    A = np.zeros((5, 5, len(g)))
    for i in range(2):                 # dx_i/dt = u_i / g
        for k in range(3):
            A[i, 2 + k] = ((i == k) - u[i] * u[k] / g ** 2) / g
    for i in range(3):                 # du_i/dt = -E_i - nu u_i
        for j in range(5):
            A[2 + i, j] = -u[i] * dnu[j] - nu * (j == 2 + i)
        if i < 2:
            A[2 + i, i] -= 0.5
    return A


def crhs(s, Cs, D):
    """dC/dt = A C + C A^T + D n n^T, n = gamma u / u^2 (the energy recoil is along u)."""
    AC = np.einsum("ijn,jkn->ikn", jac(s), Cs)
    u = s[2:5]
    u2 = (u * u).sum(0)
    n = np.concatenate([np.zeros((2, len(u2))), np.sqrt(1 + u2) * u / u2])
    return AC + AC.transpose(1, 0, 2) + D * n[:, None] * n[None, :]


C = np.zeros((5, 5, len(g0)))         # covariance of the deviations (x, y, ux, uy, uz) from the LL orbit
nst = 1000
h = T / nst
chimax = 0.0
for _ in range(nst):
    k1 = rhs(X); c1 = crhs(X, C, k1[5])
    X2 = X + h / 2 * k1; k2 = rhs(X2); c2 = crhs(X2, C + h / 2 * c1, k2[5])
    X3 = X + h / 2 * k2; k3 = rhs(X3); c3 = crhs(X3, C + h / 2 * c2, k3[5])
    X4 = X + h * k3; k4 = rhs(X4); c4 = crhs(X4, C + h * c3, k4[5])
    X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    C = C + h / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
    chimax = max(chimax, (K * np.sqrt(1 + X[2] ** 2 + X[3] ** 2 + X[4] ** 2) * np.hypot(X[0], X[1]) / 2).max())
var, nph, chisum = X[5], X[6], X[7].sum() / T
gT = np.sqrt(1 + X[2] ** 2 + X[3] ** 2 + X[4] ** 2)
gv = X[2:5] / gT                                        # d gamma / d u at t = T
vard = np.einsum("in,ijn,jn->n", gv, C[2:5, 2:5], gv)   # damped variance of gamma
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
    se = np.sqrt(vard[ok].mean() / ok.sum()) if name != "classical" else 0
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
          f" with damping {vard[m].mean():9.4g}; ratio to the damped {v / vard[m].mean():.4f} +- {ev / vard[m].mean():.4f})")
