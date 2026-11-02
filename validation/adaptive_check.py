"""Section 16: adaptive time step.
ramp OUT N GAMMA XI0 XIMIN XIMAX TEND : ion channel with n(z) = 1 + 3 min(max(z-500,0)/1000, 1);
   (1) every dt equals (2 pi/N) sqrt(2 gamma / n_max), n_max over the box-head positions of the
       step (independent re-computation), (2) r_rms(t)/r_rms(0) follows |f(t)| with
       f'' = -n(t + xi_max) f / (2 gamma), f(0) = 1, f'(0) = 0 (scipy, rtol 1e-10; in the
       quasi-static model the whole box has the density at its head),
   (3) the run ends exactly at time.t_end.
safe OUT N LAG : dt / ((2 pi/N) sqrt(2 gamma_true(t_{n+1}) / n)), gamma_true from gamma_records.txt:
   the lagged, extrapolated gamma should not give a step longer than the true one (decelerating
   beam, n = 1); during the first LAG steps no rate is known yet.
compare A B : max relative difference of the beam log (gamma_mean, r_rms) at equal times
   (interpolated), e.g. adaptive vs fixed small dt."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beamlog  # noqa: E402


def dens(z):
    return 1 + 3 * np.clip((np.asarray(z) - 500) / 1000, 0, 1)


def steps(out):
    d = np.loadtxt(out + "/timestep.txt", comments="#", ndmin=2)
    return d[:, 0].astype(int), d[:, 1], d[:, 2]


mode = sys.argv[1]
if mode == "ramp":
    out, N, gam, xi0, ximin, ximax = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), *map(float, sys.argv[5:8])
    n, t, dt = steps(out)
    # the whole box has the density at its head, z = t + xi_max; n_max over the head positions of
    # the step, first with dt_{n-1}, then with the dt found (monotone ramp: max at an end)
    zh, c = t + ximax, 2 * np.pi / N
    dprev = np.concatenate([[0.0], dt[:-1]])
    n1 = np.maximum(dens(zh), dens(zh + dprev))
    d1 = c * np.sqrt(2 * gam / n1)
    n2 = np.maximum(dens(zh), dens(zh + d1))
    dt_ref = np.where(n2 > n1, np.minimum(d1, c * np.sqrt(2 * gam / n2)), d1)
    tend = float(sys.argv[8])
    ok = t + 2 * dt_ref < tend                  # the last steps land on t_end
    e1 = np.max(np.abs(dt[ok] / dt_ref[ok] - 1))
    from scipy.integrate import solve_ivp
    b = read_beamlog(out + "/beams.txt")["witness"]
    T = b["t"]
    sol = solve_ivp(lambda s, y: [y[1], -dens(s + ximax) * y[0] / (2 * gam)], (0, T[-1]), [1.0, 0.0],
                    t_eval=T, rtol=1e-10, atol=1e-12, max_step=5.0)
    e2 = np.max(np.abs(b["r_rms"] / b["r_rms"][0] - np.abs(sol.y[0])))
    print(f"  {len(dt)} steps, dt {dt.min():.3f} ... {dt.max():.3f}: max |dt/dt_formula - 1| = {e1:.1e}"
          f" (all but the last {np.sum(~ok)}, which land on t_end)")
    print(f"  r_rms/r0 vs |f(t)| of the ODE: max deviation {e2:.2e};  run ends at t = {T[-1]:.6f} (t_end {tend:g})")
elif mode == "safe":
    out, N, lag = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    n, t, dt = steps(out)
    g = np.loadtxt(out + "/gamma_records.txt", comments="#", ndmin=2)
    G = dict(zip(g[:, 0].astype(int), g[:, 2]))      # true gamma_eff after the push of step m
    dt_true = np.array([2 * np.pi / N * np.sqrt(2 * G[k]) for k in n])
    r = dt / dt_true
    print(f"  lag {lag}: {len(dt)} steps, gamma_eff {min(G.values()):.2f} ... {max(G.values()):.2f}: max dt / dt(true gamma"
          f" at t_n+1) = {r.max():.4f}, after the first {lag} steps {r[n > lag].max():.4f}")
elif mode == "compare":
    a = read_beamlog(sys.argv[2] + "/beams.txt")["witness"]
    b = read_beamlog(sys.argv[3] + "/beams.txt")["witness"]
    worst = []
    for k in ("gamma_mean", "r_rms"):
        bi = np.interp(a["t"], b["t"], b[k])
        worst.append(np.max(np.abs(a[k] - bi)) / np.max(np.abs(b[k])))
    print(f"  gamma_mean max rel. diff {worst[0]:.2e}, r_rms {worst[1]:.2e}  ({len(a['t'])} vs {len(b['t'])} steps)")
