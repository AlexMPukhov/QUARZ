"""Checks of electron-impact ionization by plasma electrons (<species>.ionization.electron_impact) against an
independent implementation of Lotz's formula (Z. Phys. 206, 205 (1967), Table 1; ions: a = 4.5e-14 cm^2 eV^2,
b = c = 0, outermost subshell), in physical units.

usage:
  eimpact_check.py rate DIR n0_cm3 u density_a density_b GAS:density:z0[:file]
      two counter-streaming electron species with u_z = +-u (stream densities density_a, density_b): the lab
      density n (gamma/Delta) of each vs the code's electron density, and the ionized fraction of the dilute
      gas along xi vs 1 - exp(-R xi), R = sum_s n_s,lab sigma(E) v k_p^-1/c.  With a file: sigma(E) table.
  eimpact_check.py loss DIR n0_cm3 u GAS:density:z0  SPECIES...
      dense gas, dilute projectiles: the energy of the projectiles at the end of the box vs the integral of
      dE/dxi = -n sigma(E) (Ip + <W>(E)) |u|/Delta k_p^-1 (RK4), <W> of dsigma/dW ~ 1/(J^2+W^2) on
      [0, (E - Ip)/2]; and the energy balance: energy lost by all electrons vs sum over the electron-impact
      births of (Ip + W) (expected: equal within the Monte Carlo noise).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402

ME = 510998.95
IP = {"Xe": [12.1298, 20.975, 32.1230], "Ar": [15.7596, 27.62966, 40.74], "He": [24.58739, 54.41776]}
J_OPB = {"Xe": 8.7, "Ar": 10.0, "He": 15.8}
# Lotz 1967, Table 1: (P_i, q_i), a, b, c
LOTZ = {("Xe", 0): ([12.1, 23.4, 67], [6, 2, 10], 4.0, 0.54, 0.64),
        ("Ar", 0): ([15.8, 29.2], [6, 2], 4.0, 0.62, 0.40),
        ("He", 0): ([24.6], [2], 4.0, 0.75, 0.46),
        ("He", 1): ([54.4], [1], 4.4, 0.38, 0.60)}
OUTER = {("Xe", 1): 5, ("Xe", 2): 4, ("Ar", 1): 5, ("Ar", 2): 4}   # 5p^5, 5p^4; 3p^5, 3p^4


def sigma(gas, z, E, table=None):
    """cm^2, E in eV"""
    Ip = IP[gas][z]
    E = np.atleast_1d(np.asarray(E, dtype=float))
    if table is not None:
        t = np.loadtxt(table)
        s = np.interp(E, t[:, 0], t[:, 1], left=0.0)
        hi = E > t[-1, 0]
        s[hi] = t[-1, 1] * (np.log(E[hi] / Ip) / E[hi]) / (np.log(t[-1, 0] / Ip) / t[-1, 0])
        return np.where(E > Ip, s, 0.0)
    if (gas, z) in LOTZ:
        P, q, a, b, c = LOTZ[(gas, z)]
        P = [Ip] + P[1:]
    else:
        P, q, a, b, c = [Ip], [OUTER[(gas, z)]], 4.5, 0.0, 0.0
    s = sum(qi * np.log(np.maximum(E / Pi, 1)) / (E * Pi) for Pi, qi in zip(P, q))
    return np.where(E > Ip, 1e-14 * a * (1 - b * np.exp(-c * (E / Ip - 1))) * s, 0.0)


def mean_w(gas, z, E):
    Ip, J = IP[gas][z], J_OPB[gas] * IP[gas][z] / IP[gas][0]
    wm = np.maximum(0.5 * (E - Ip), 1e-300)
    return J * np.log1p((wm / J) ** 2) / (2 * np.arctan(wm / J))


def units(n0):
    wp = 5.64146e4 * np.sqrt(n0)
    return 2.99792458e10 / wp   # k_p^-1 in cm


what = sys.argv[1]
if what == "rate":
    d, n0, u = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    da, db = float(sys.argv[5]), float(sys.argv[6])
    g = sys.argv[7].split(":")
    gas, dens, z0 = g[0], float(g[1]), int(g[2])
    table = g[3] if len(g) > 3 else None
    kpi = units(n0)
    gam = np.sqrt(1 + u * u)
    E = (gam - 1) * ME
    beta = u / gam
    nlab = n0 * (da * gam / (gam - u) + db * gam / (gam + u))   # cm^-3
    sg = float(sigma(gas, z0, E, table)[0])
    R = nlab * sg * beta * kpi                                    # per k_p^-1 of xi
    f = read_fields(d + "/fields_000000.bin")
    r, xi = f["r"], f["xi"]
    sel = r < 0.8 * r.max()
    ne = f["ne"][len(xi) // 2, sel].mean() * n0
    print(f"  E = {E:.2f} eV, sigma({gas}{'+' * z0 if z0 else ''} -> +1) = {sg:.4e} cm^2"
          f"{' (table)' if table else ''};  electron density (lab) code {ne:.5e}, expected {nlab:.5e} cm^-3")
    out = []
    worst = 0.0
    for frac in (0.25, 0.5, 1.0):
        k = int(frac * (len(xi) - 1))
        code = (f["nz_xe"][k, sel].mean() / dens) - z0
        ref = -np.expm1(-R * (xi[k] + (xi[1] - xi[0])))
        worst = max(worst, abs(code / ref - 1))
        out.append(f"xi = {xi[k]:.0f}: {code:.4e} / {ref:.4e}")
    print("  ionized fraction code / expected:  " + "   ".join(out) + f"   (max rel. diff {worst:.1e})")
elif what == "loss":
    d, n0, u = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    g = sys.argv[5].split(":")
    gas, dens, z0 = g[0], float(g[1]), int(g[2])
    proj = sys.argv[6:]
    kpi = units(n0)
    f = read_fields(d + "/fields_000000.bin")
    xi = f["xi"]
    L = xi[-1] - xi[0]
    n = dens * n0                    # cm^-3 (the gas stays almost neutral)

    def rhs(E, sgn):
        g_ = 1 + E / ME
        uu = np.sqrt(g_ * g_ - 1)
        D = g_ - sgn * uu
        return -n * sigma(gas, z0, E) * (IP[gas][z0] + mean_w(gas, z0, E)) * uu / D * kpi

    E0 = (np.sqrt(1 + u * u) - 1) * ME
    for sp, sgn in zip(proj, (1, -1)):
        E, h = E0, L / 4000
        for _ in range(4000):
            k1 = rhs(E, sgn); k2 = rhs(E + h * k1 / 2, sgn); k3 = rhs(E + h * k2 / 2, sgn); k4 = rhs(E + h * k3, sgn)
            E += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = np.loadtxt(d + f"/tail_{sp}_000000.txt")
        p2 = t[:, 2] ** 2 + t[:, 3] ** 2
        gm = (1 + p2 + t[:, 4] ** 2) / (2 * t[:, 4])
        Ec = np.average((gm - 1) * ME, weights=t[:, 5])
        print(f"  {sp} (u_z = {'+' if sgn > 0 else '-'}{u}): energy {E0:.2f} eV -> code {Ec:.4f} eV, RK4 {float(np.squeeze(E)):.4f} eV"
              f"  (loss: code {E0 - Ec:.4f}, RK4 {E0 - float(np.squeeze(E)):.4f}, rel. diff {(E0 - Ec) / (E0 - float(np.squeeze(E))) - 1:+.1e})")
    # energy balance over all electron species
    lost = 0.0
    for fn in glob.glob(d + "/tail_*_000000.txt"):
        sp = os.path.basename(fn)[5:-11]
        t = np.loadtxt(fn, ndmin=2)
        if t.size == 0: continue
        p2 = t[:, 2] ** 2 + t[:, 3] ** 2
        gm = (1 + p2 + t[:, 4] ** 2) / (2 * t[:, 4])
        e_end = np.sum(t[:, 5] * (gm - 1) * ME)
        if sp in proj: lost += np.sum(t[:, 5]) * E0 - e_end
        else: lost -= e_end            # secondaries: start at their birth energy (counted below)
    b = np.vstack([np.loadtxt(fn, ndmin=2) for fn in glob.glob(d + f"/born_xe_*.txt")])
    sel = b[:, 7] == 2
    p2 = b[sel, 3] ** 2 + b[sel, 4] ** 2
    gm = (1 + p2 + b[sel, 5] ** 2) / (2 * b[sel, 5])
    Wb = np.sum(b[sel, 6] * (gm - 1) * ME)
    Ib = np.sum(b[sel, 6]) * IP[gas][z0]
    lost += Wb
    rel = np.sqrt(sel.sum())
    print(f"  energy balance: lost by the electrons {lost:.5e}, given to ionization (Ip + W) {Ib + Wb:.5e}"
          f"  (ratio {lost / (Ib + Wb):.4f}; {sel.sum()} births, Monte Carlo error ~{1 / rel:.1e})")
