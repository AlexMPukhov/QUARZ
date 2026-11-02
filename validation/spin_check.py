#!/usr/bin/env python3
"""Spin precession check (section 20).
channel OUT STEP: cold beams in a pure ion channel, spin initially along z. Each particle moves in its
  own r-z plane, the spin turns about the same axis kappa times faster than the momentum:
  s_r = sin(kappa theta), s_z = cos(kappa theta), theta = atan(p_r / p_z), kappa = g beta^2 (a + 1/(g+1)).
  Prints, per beam, the error relative to the largest s_r, the out-of-plane spin and max ||s| - 1|.
wake OUT STEP0 STEP1 DT [A]: low-energy witness in a stationary linear wake, vs an RK4 integration (Python) of
  the Lorentz force and the T-BMT equation in the fields of the QUARZ field file.
free OUT STEP A: scattering only (no wake): small-angle kicks about varying axes commute to first order,
  s_perp = kappa theta_perp + O(theta^2); prints the fitted slope of s_x vs theta_x over kappa."""
import sys
import numpy as np
sys.path.insert(0, "../tools")
from quarz_read import read_beam, read_spinlog

A = {"electrons": 0.00115965218128, "muons": 0.00116592061, "pbar": 1.79284734463, "witness": 0.00115965218128}


def kappa(b, a):
    p2 = b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2
    g = np.sqrt(1 + p2)
    return p2 / g * (a + 1 / (g + 1))


mode, out, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
if mode == "channel":
    for name in ["electrons", "muons", "pbar"]:
        b = read_beam(f"{out}/beam_{name}_{step:06d}.bin")
        r = np.hypot(b["x"], b["y"])
        c, s = b["x"] / r, b["y"] / r
        pr = b["px"] * c + b["py"] * s
        sr = b["sx"] * c + b["sy"] * s
        st = -b["sx"] * s + b["sy"] * c                       # out of the r-z plane: should stay 0
        th = np.arctan2(pr, b["pz"])
        k = kappa(b, A[name])
        ref = np.sin(k * th)
        smax = np.max(np.abs(ref))
        err = np.max(np.abs(sr - ref)) / smax
        err_z = np.max(np.abs(b["sz"] - np.cos(k * th)))
        norm = np.max(np.abs(np.sqrt(b["sx"] ** 2 + b["sy"] ** 2 + b["sz"] ** 2) - 1))
        print(f" {name:9s} kappa {np.mean(k):8.4f}  max|s_r| {smax:.3e}  rel. error of s_r = sin(kappa theta) {err:.2e}"
              f"  s_z {err_z:.1e}  max|s_theta| {np.max(np.abs(st)):.1e}  max||s|-1| {norm:.1e}")
    sl = read_spinlog(f"{out}/spin.txt")
    print(f" spin.txt: final polarization P = |<s>|: " +
          "  ".join(f"{k} {v['P'][-1]:.6f}" for k, v in sl.items()))
elif mode == "wake":
    # RK4 of du/dt = (q/m)(E + v x B), ds/dt = (q/m) s x W, dxi/dt = v_z - 1 with the stored (stationary) fields
    from quarz_read import read_fields
    nend = int(sys.argv[4])
    F = read_fields(f"{out}/fields_{step:06d}.bin")
    r, xg = F["r"], F["xi"]
    dr, dx = r[1] - r[0], xg[1] - xg[0]
    a = float(sys.argv[6]) if len(sys.argv) > 6 else A["witness"]
    b0 = read_beam(f"{out}/beam_witness_{step:06d}.bin")
    b1 = read_beam(f"{out}/beam_witness_{nend:06d}.bin")
    dt_code = float(sys.argv[5])
    T = (nend - step) * dt_code - 0.5 * dt_code   # momenta and spins of the dump are at t_n - dt/2

    def gather(name, R, XI):
        j = np.clip(((R - r[0]) / dr).astype(int), 0, len(r) - 2)
        k = np.clip(((XI - xg[0]) / dx).astype(int), 0, len(xg) - 2)
        tr = (R - r[j]) / dr
        tk = (XI - xg[k]) / dx
        f = F[name]
        return (1 - tk) * ((1 - tr) * f[k, j] + tr * f[k, j + 1]) + tk * ((1 - tr) * f[k + 1, j] + tr * f[k + 1, j + 1])

    def rhs(y):
        X, Y, XI, u, sp = y[0], y[1], y[2], y[3:6], y[6:9]
        R = np.hypot(X, Y)
        c = np.where(R > 0, X / np.maximum(R, 1e-300), 1.0)
        s = np.where(R > 0, Y / np.maximum(R, 1e-300), 0.0)
        er, et, br, bt = gather("er", R, XI), gather("eth", R, XI), gather("br", R, XI), gather("bth", R, XI)
        E = np.array([er * c - et * s, er * s + et * c, gather("ez", R, XI)])
        B = np.array([br * c - bt * s, br * s + bt * c, gather("bz", R, XI)])
        g = np.sqrt(1 + np.sum(u * u, axis=0))
        be = u / g
        du = -(E + np.cross(be, B, axis=0))
        W = (a + 1 / g) * B - (a * g / (g + 1) * np.sum(be * B, axis=0)) * be - (a + 1 / (g + 1)) * np.cross(be, E, axis=0)
        ds = -np.cross(sp, W, axis=0)
        return np.concatenate([be[0:1], be[1:2], be[2:3] - 1, du, ds])

    y = np.array([b0["x"], b0["y"], b0["xi"], b0["px"], b0["py"], b0["pz"], b0["sx"], b0["sy"], b0["sz"]])
    nsub = int(np.ceil(T / 0.01))
    h = T / nsub
    for _ in range(nsub):
        k1 = rhs(y); k2 = rhs(y + 0.5 * h * k1); k3 = rhs(y + 0.5 * h * k2); k4 = rhs(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ds = np.sqrt((b1["sx"] - y[6]) ** 2 + (b1["sy"] - y[7]) ** 2 + (b1["sz"] - y[8]) ** 2)
    turn = np.arccos(np.clip(y[6] * b0["sx"] + y[7] * b0["sy"] + y[8] * b0["sz"], -1, 1))
    du = np.sqrt((b1["px"] - y[3]) ** 2 + (b1["py"] - y[4]) ** 2 + (b1["pz"] - y[5]) ** 2) / np.sqrt(1 + y[3] ** 2 + y[4] ** 2 + y[5] ** 2)
    g1 = np.sqrt(1 + b1["px"] ** 2 + b1["py"] ** 2 + b1["pz"] ** 2)
    print(f" wake, a = {a:g}, gamma {np.mean(np.sqrt(1 + b0['pz'] ** 2)):.2f} -> {np.mean(g1):.2f}: rms spin turn {np.sqrt(np.mean(turn ** 2)):.3f} rad,"
          f"  max |s - s_ref| {np.max(ds):.1e}  (rms {np.sqrt(np.mean(ds ** 2)):.1e}),  max |u - u_ref|/gamma {np.max(du):.1e}")
else:
    a = float(sys.argv[4])
    b = read_beam(f"{out}/beam_witness_{step:06d}.bin")
    k = kappa(b, a)
    thx, thy = b["px"] / b["pz"], b["py"] / b["pz"]
    sl = (np.sum(b["sx"] * thx * k) / np.sum((k * thx) ** 2), np.sum(b["sy"] * thy * k) / np.sum((k * thy) ** 2))
    res = np.std(b["sx"] - k * thx) / np.std(k * thx)
    norm = np.max(np.abs(np.sqrt(b["sx"] ** 2 + b["sy"] ** 2 + b["sz"] ** 2) - 1))
    print(f" scattering, a = {a:g}: kappa {np.mean(k):.4f}  slope s_x/(kappa theta_x) {sl[0]:.6f}"
          f"  s_y/(kappa theta_y) {sl[1]:.6f}  residual rms {res:.1e}  rms theta {np.std(thx):.1e}  max||s|-1| {norm:.1e}")
