"""Energy balance of the laser ionization losses (laser.ionization_losses), validation section 12.
usage: ion_loss_check.py DIR_ON DIR_OFF DIR_PLASMA Ip_eV [tol]
  DIR_ON:     ionizing gas, losses on      -> dW = -dt sum w (Ip + p_d^2/2)
  DIR_OFF:    ionizing gas, losses off     -> dW = +dt sum w <a^2>/2   (photon number conserved)
  DIR_PLASMA: pre-ionized, no ionization   -> dW = 0, photon number conserved
The laser energy and photon number come from the field files (envelope a^, linear polarisation, xi = z - t):
  E^ = i k0 a^ + da^/dxi,  W = sum |E^|^2/2 V dxi,  N = sum Re[conj(a^) (k0 a^ - i da^/dxi)]/2 V dxi  (per unit omega_p);
the predictions come from the electrons the code wrote at birth (born_dump: slice, x, y, p_x, p_y, Delta, w) and the
envelope of the same step at the birth position (<a^2> = |a^|^2/2, linear interpolation in r), with the code's cell
volumes V (grid.txt).  Independent of the implementation of the loss term."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402

ME_EV = 510998.95


def grid_volumes(d):
    g = np.loadtxt(os.path.join(d, "grid.txt"), comments="#")
    return g[:, 1], g[:, 3]


def energy(fn, V, k0):
    f = read_fields(fn)
    A = f["a_re"] + 1j * f["a_im"]               # [K][M]
    xi = f["xi"]
    dA = np.gradient(A, xi, axis=0)
    E = 1j * k0 * A + dA
    dxi = abs(xi[1] - xi[0])
    W = np.sum(0.5 * np.abs(E) ** 2 * V[None, :]) * dxi
    N = np.sum(0.5 * np.real(np.conj(A) * (k0 * A - 1j * dA)) * V[None, :]) * dxi
    return W, N, f


def run(d, k0, dt, ip):
    r, V = grid_volumes(d)
    files = sorted(glob.glob(os.path.join(d, "fields_*.bin")))
    W0, N0, _ = energy(files[0], V, k0)
    W1, N1, _ = energy(files[-1], V, k0)
    nsteps = len(files) - 1
    s_ip = s_q = s_d = 0.0
    nb = 0
    for n in range(nsteps):
        bf = os.path.join(d, f"born_gas_{n:06d}.txt")
        if not os.path.exists(bf):
            continue
        b = np.loadtxt(bf, comments="#", ndmin=2)
        if b.size == 0:
            continue
        f = read_fields(files[n])
        A2 = f["a_re"] ** 2 + f["a_im"] ** 2
        xi = f["xi"]
        # global slice k counts from the head of the box (xi = xi_max - k dxi); find its row in the file
        dxi = abs(xi[1] - xi[0])
        rows = np.rint((xi.max() - b[:, 0] * dxi - xi[0]) / (xi[1] - xi[0])).astype(int)
        if rows.min() < 0 or rows.max() >= len(xi):
            raise SystemExit("birth slices outside the field file")
        rr = np.hypot(b[:, 1], b[:, 2])
        aa = np.array([0.5 * np.interp(rb, f["r"], A2[k]) for rb, k in zip(rr, rows)])
        w = b[:, 6]
        s_ip += np.sum(w) * ip / ME_EV
        s_q += np.sum(w * 0.5 * aa)
        s_d += np.sum(w * 0.5 * (b[:, 3] ** 2 + b[:, 4] ** 2))
        nb += len(w)
    return dict(W0=W0, dW=W1 - W0, dN=N1 - N0, N0=N0, ip=dt * s_ip, q=dt * s_q, d=dt * s_d, nb=nb)


if __name__ == "__main__":
    d_on, d_off, d_pl, ip = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
    d_noip = sys.argv[6] if len(sys.argv) > 6 else None   # laser.ionization_potential_loss = 0
    k0 = float(os.environ.get("K0", "20.0"))
    dt = float(os.environ.get("DT", "2.0"))
    ok = True
    on, off, pl = run(d_on, k0, dt, ip), run(d_off, k0, dt, ip), run(d_pl, k0, dt, ip)
    pred_on = -(on["ip"] + on["d"])
    pred_off = off["q"]
    e_on = on["dW"] / pred_on - 1
    e_off = off["dW"] / pred_off - 1
    print(f"  losses on:  dW/W = {on['dW'] / on['W0']:+.4e}, predicted -dt sum w (Ip + p_d^2/2) = {pred_on / on['W0']:+.4e} "
          f"(Ip {on['ip'] / on['W0']:.3e}, drift {on['d'] / on['W0']:.3e}; {on['nb']} births): ratio - 1 = {e_on:+.2e}")
    print(f"  losses off: dW/W = {off['dW'] / off['W0']:+.4e}, predicted +dt sum w <a^2>/2 = {pred_off / off['W0']:+.4e}: "
          f"ratio - 1 = {e_off:+.2e}; photon number dN/N = {off['dN'] / off['N0']:+.2e}")
    print(f"  pre-ionized plasma: dW/W = {pl['dW'] / pl['W0']:+.2e}, dN/N = {pl['dN'] / pl['N0']:+.2e}")
    if d_noip:
        ni = run(d_noip, k0, dt, ip)
        pred = -ni["d"]
        # small: compare the difference to the size of the full loss of the 'on' run
        e_ni = (ni["dW"] - pred) / abs(pred_on)
        print(f"  losses without Ip: dW/W = {ni['dW'] / ni['W0']:+.4e}, predicted -dt sum w p_d^2/2 = {pred / ni['W0']:+.4e}: "
              f"difference / full loss = {e_ni:+.2e}")
        if abs(e_ni) > tol: ok = False; print(f"FAILED: losses without Ip: off by {e_ni:+.2e} of the full loss (tol {tol})")
    # the size of the effect: the losses remove (on) the quiver energy the cycle-averaged model gains (off), plus Ip
    if abs(e_on) > tol: ok = False; print(f"FAILED: losses on: energy balance off by {e_on:+.2e} (tol {tol})")
    if abs(e_off) > tol: ok = False; print(f"FAILED: losses off: quiver gain off by {e_off:+.2e} (tol {tol})")
    if abs(pl["dW"] / pl["W0"]) > tol * abs(pred_on / on["W0"]):
        ok = False; print("FAILED: the pre-ionized plasma does not conserve the laser energy")
    print("ionization losses: ok" if ok else "ionization losses test FAILED")
    sys.exit(0 if ok else 1)
