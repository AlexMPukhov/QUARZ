"""Checks of the ionization module against independent reference calculations.

usage:
  ion_check.py adk DIR n0_cm3 ppc dr  NAME:ELEMENT:density [NAME:ELEMENT:density ...]
      field ionization by the (stored) plasma + beam fields: the expected charge state of every
      loaded ring is computed from the field files with an independent ADK implementation (SI units),
      as the exact Markov chain of the per-slice probabilities used by the code and as the continuous
      rate equations; compared with the simulated ion charge density at the end of the box.
  ion_check.py laser DIR n0_cm3 k0 a0 L0 xi0 ELEMENT density [z_max]
      laser ionization in 1D (wide spot): the code's ionization (nz on the axis) vs the exact
      cycle-resolved ADK rate integrated over the true oscillating field, and the mean p_perp^2
      of the born electrons vs a test-particle calculation (drift p = -a(t_birth)).
  ion_check.py impact DIR n0_cm3 sigma_cm2_or_M2:C beam_n0 sigma_r sigma_xi xi0 gamma density
      beam impact ionization of a neutral gas vs P = 1 - exp(-sigma n0 k_p^-1 beta int n_b dxi).
  ion_check.py impact_gases DIR n0_cm3 beam_n0 sigma_r sigma_xi xi0 gamma NAME:density:sigma_ref:nP_ref ...
      built-in Bethe data per gas: the cross-section that reproduces the simulated ionized charge at the
      end of the box vs a measured reference sigma_ref (1e-20 cm^2; Rieke & Prepejchal minimum primary
      cross-sections as tabulated by Blum, Riegler & Rolandi, Table 1.2), and the primary ion pairs per
      cm at NTP (20 C, 1 atm) vs nP_ref (measured, PDG / Blum Table 1.1; information only).
"""
import os
import sys

import numpy as np
from math import gamma as Gamma

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402

IP = {"H": [13.59844], "He": [24.58739, 54.41776], "Li": [5.39172, 75.6402, 122.4543],
      "N": [14.5341, 29.6013, 47.44924, 77.4735, 97.8902, 552.0718, 667.046],
      "Ar": [15.7596, 27.62966, 40.74, 59.81, 75.02, 91.009, 124.323, 143.460]}
EA, TA, HA = 5.14220674763e11, 2.4188843265857e-17, 27.211386245988


def units(n0):
    wp = 5.64146e4 * np.sqrt(n0)
    return wp, 9.1093837015e-31 * 2.99792458e8 * wp / 1.602176634e-19


def adk_si(E, ip_eV, Z):
    """static ADK rate (1/s) for field E (V/m), l = m = 0, l* = n* - 1"""
    Ip = ip_eV / HA
    ns = Z / np.sqrt(2 * Ip)
    C2 = 2 ** (2 * ns) / (ns * Gamma(2 * ns))
    F = np.maximum(np.abs(E), 1e-300) / EA
    e32 = (2 * Ip) ** 1.5
    with np.errstate(over="ignore", under="ignore"):
        W = Ip * C2 * (2 * e32 / F) ** (2 * ns - 1) * np.exp(-2 * e32 / (3 * F)) / TA
    return np.where(np.abs(E) > 0, W, 0.0)


def rings(r_nodes, ppc, dens):
    out_r, out_w = [], []
    for j in range(len(r_nodes) - 1):
        h = r_nodes[j + 1] - r_nodes[j]
        for m in range(ppc):
            ra, rb = r_nodes[j] + m * h / ppc, r_nodes[j] + (m + 1) * h / ppc
            out_r.append(2 / 3 * (rb ** 3 - ra ** 3) / (rb ** 2 - ra ** 2))
            out_w.append(dens * np.pi * (rb ** 2 - ra ** 2))
    return np.array(out_r), np.array(out_w)


what = sys.argv[1]
if what == "adk":
    d, n0, ppc, dr = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    wp, E0 = units(n0)
    f = read_fields(d + "/fields_000000.bin")
    r, xi = f["r"], f["xi"]
    dxi = xi[1] - xi[0]
    Emag = np.sqrt(f["er"] ** 2 + f["eth"] ** 2 + f["ez"] ** 2) * E0          # V/m, [K, M]
    for spec in sys.argv[6:]:
        name, el, dens = spec.split(":")
        dens = float(dens)
        ips = IP[el]
        L = len(ips)
        rr, ww = rings(r, ppc, dens)
        # field at the rings (linear in r, as the code)
        Er = np.array([np.interp(rr, r, Emag[k]) for k in range(len(xi))])  # [K, nring]
        W = np.stack([adk_si(Er, ips[l], l + 1) / wp for l in range(L)], axis=-1)   # [K, nring, L]
        # Markov chain of the code (all slices but the last are followed by an ionization step;
        # the last slice also ionizes): per slice P_l = 1 - exp(-W_l dxi), cascade within the slice
        f_m = np.zeros((len(rr), L + 1)); f_m[:, 0] = 1
        f_c = f_m.copy()
        for k in range(len(xi)):
            P = -np.expm1(-W[k] * dxi)          # [nring, L]
            new = np.zeros_like(f_m)
            for z in range(L + 1):
                stay = f_m[:, z].copy()
                acc = 1.0
                for l in range(z, L + 1):
                    if l == L:
                        new[:, l] += stay * acc
                        break
                    new[:, l] += stay * acc * (1 - P[:, l])
                    acc = acc * P[:, l]
            f_m = new
            # continuous rate equations (exact for piecewise-constant W over dxi), substeps
            for _ in range(8):
                flux = -np.expm1(-W[k] * dxi / 8) * f_c[:, :L]
                f_c[:, :L] -= flux
                f_c[:, 1:] += flux
        zm = (f_m * np.arange(L + 1)).sum(1)
        zc = (f_c * np.arange(L + 1)).sum(1)
        nz = f["nz_" + name][-1]
        Vn = np.zeros(len(r))   # hat-function volumes: int nz dV = sum w z
        for k in range(len(r) - 1):
            a, b = r[k], r[k + 1]; h = b - a
            Vn[k] += 2 * np.pi * (a * h / 2 + h * h / 6); Vn[k + 1] += 2 * np.pi * (b * h / 2 - h * h / 6)
        sim_tot = (nz * Vn).sum()
        ref_tot = (ww * zm).sum()
        # statistical error of the all-or-nothing Monte Carlo (variance of z per ring)
        varz = (f_m * np.arange(L + 1) ** 2).sum(1) - zm ** 2
        sig = np.sqrt((ww ** 2 * varz).sum())
        print(f"  {name} ({el}): total ionized charge  code {sim_tot:.5e}   expected {ref_tot:.5e} "
              f"(+- {sig:.1e} stat.)   continuous rate eq. {(ww * zc).sum():.5e}   -> "
              f"{(sim_tot - ref_tot) / sig:+.2f} sigma")
        # radial bands
        bands = np.linspace(0, r[-1] * 0.75, 7)
        line = []
        for a, b in zip(bands[:-1], bands[1:]):
            sel = (rr >= a) & (rr < b)
            ref = (ww[sel] * zm[sel]).sum() / ww[sel].sum()
            nsel = (r >= a) & (r < b)
            simv = (nz[nsel] * Vn[nsel]).sum() / (dens * Vn[nsel]).sum()
            e = np.sqrt((ww[sel] ** 2 * varz[sel]).sum()) / ww[sel].sum()
            line.append(f"[{a:.2f},{b:.2f}) {simv:.4f}/{ref:.4f}+-{e:.4f}")
        print("    <z> in r-bands, code/expected: " + "  ".join(line))

elif what == "laser":
    d, n0, k0, a0, L0, x0, el, dens = (sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
                                        float(sys.argv[6]), float(sys.argv[7]), sys.argv[8], float(sys.argv[9]))
    wp, E0 = units(n0)
    f = read_fields(d + "/fields_000000.bin")
    r, xi = f["r"], f["xi"]
    nz = f["nz_gas"][:, r < 1.0].mean(axis=1) / dens       # <z> near the axis (the spot is wide)
    ips = IP[el][:int(sys.argv[10])] if len(sys.argv) > 10 else IP[el]
    L = len(ips)
    # exact: E(xi) = k0 a_env sin(k0 xi) (a = a_env cos(k0 xi)), resolved with 400 points per period
    h = 2 * np.pi / k0 / 400
    xf = np.arange(xi[0], xi[-1] + h, h)
    aenv = a0 * np.exp(-(xf - x0) ** 2 / L0 ** 2)
    ph = k0 * xf
    E = k0 * aenv * np.abs(np.sin(ph)) * E0
    W = np.stack([adk_si(E, ips[l], l + 1) / wp for l in range(L)])     # [L, nf]
    drift2 = (aenv * np.cos(ph)) ** 2
    fz = np.zeros(L + 1); fz[0] = 1
    zt = np.zeros(len(xf))
    born, bornp2 = 0.0, 0.0
    for n in range(len(xf)):
        P = -np.expm1(-W[:, n] * h)
        flux = P * fz[:L]
        fz[:L] -= flux
        fz[1:] += flux
        born += flux.sum(); bornp2 += flux.sum() * drift2[n]
        zt[n] = (fz * np.arange(L + 1)).sum()
    # the ionization step of slice k (rate at xi_k, duration dxi) represents [xi_k - dxi/2, xi_k + dxi/2]
    zref = np.interp(xi + 0.5 * (xi[1] - xi[0]), xf, zt)
    print(f"  <z> on axis vs xi: max |code - exact cycle-resolved| = {np.abs(nz - zref).max():.3f} at xi - xi0 = "
          f"{xi[np.argmax(np.abs(nz - zref))] - x0:+.2f}  (final {nz[-1]:.4f} vs {zref[-1]:.4f})")
    for xx in np.arange(x0 - 2 * L0, x0 + 0.51 * L0, 0.5 * L0):
        k = np.argmin(abs(xi - xx))
        print(f"    xi - xi0 = {xi[k] - x0:+5.2f}:  code {nz[k]:.3f}   exact {zref[k]:.3f}")
    a = np.loadtxt(d + "/ionization.txt")
    print(f"  mean p_perp^2 of the born electrons: code {a[4]:.4e}   test particles (p = -a(t_birth)) {bornp2 / born:.4e}"
          f"   (ratio {a[4] / (bornp2 / born):.3f}; for comparison <a^2> at the pulse peak {a0 ** 2 / 2:.1e})")

elif what == "impact":
    d, n0, sp = sys.argv[2], float(sys.argv[3]), sys.argv[4]
    nb, sr, sxi, x0, gam, dens = [float(v) for v in sys.argv[5:11]]
    wp, E0 = units(n0)
    kpi = 2.99792458e10 / wp
    b2 = 1 - 1 / gam ** 2
    if ":" in sp:
        M2, C = [float(v) for v in sp.split(":")]
        sigma = 1.87386e-20 / b2 * (M2 * (np.log(b2 * gam ** 2) - b2) + C)
    else:
        sigma = float(sp)
    f = read_fields(d + "/fields_000000.bin")
    r, xi = f["r"], f["xi"]
    nz = f["nz_gas"] / dens
    s = sigma * n0 * kpi * np.sqrt(b2)
    print(f"  sigma = {sigma:.4e} cm^2,  dP/dxi = {s:.4e} n_b")
    # code value at slice k includes [xi_k, xi_k + dxi]
    Ixi = lambda x: nb * np.sqrt(np.pi / 2) * sxi * (1 + np.vectorize(__import__('math').erf)((x - x0) / (np.sqrt(2) * sxi)))
    dx = xi[1] - xi[0]
    Vn = np.zeros(len(r))
    for k in range(len(r) - 1):
        a, b = r[k], r[k + 1]; h = b - a
        Vn[k] += 2 * np.pi * (a * h / 2 + h * h / 6); Vn[k + 1] += 2 * np.pi * (b * h / 2 - h * h / 6)
    kk = len(xi) - 1
    ref_all = (-np.expm1(-s * Ixi(xi[kk] + dx) * np.exp(-r ** 2 / (2 * sr ** 2))) * Vn).sum()
    print(f"  total ionized charge at the end of the box: code {(nz[kk] * Vn).sum():.5e}   analytic {ref_all:.5e}"
          f"   (rel. diff {(nz[kk] * Vn).sum() / ref_all - 1:+.1e})")
    for xx in (x0 - sxi, x0, x0 + sxi, xi[-1]):
        k = np.argmin(abs(xi - xx))
        out = []
        for a, b in ((0, 0.5 * sr), (0.5 * sr, sr), (sr, 2 * sr), (2 * sr, 3 * sr)):
            sel = (r >= a) & (r < b)
            ref = (-np.expm1(-s * Ixi(xi[k] + dx) * np.exp(-r[sel] ** 2 / (2 * sr ** 2))) * Vn[sel]).sum() / Vn[sel].sum()
            code = (nz[k, sel] * Vn[sel]).sum() / Vn[sel].sum()
            out.append(f"r/sr in [{a / sr:.1f},{b / sr:.1f}): {code:.4e} / {ref:.4e}")
        print(f"    xi - xi0 = {xi[k] - x0:+6.1f}: ionized fraction code / analytic  " + "  ".join(out))

elif what == "impact_gases":
    from scipy.optimize import brentq
    d, n0 = sys.argv[2], float(sys.argv[3])
    nb, sr, sxi, x0, gam = [float(v) for v in sys.argv[4:9]]
    wp, E0 = units(n0)
    kpi = 2.99792458e10 / wp
    beta = np.sqrt(1 - 1 / gam ** 2)
    f = read_fields(d + "/fields_000000.bin")
    r, xi = f["r"], f["xi"]
    dx = xi[1] - xi[0]
    Vn = np.zeros(len(r))
    for k in range(len(r) - 1):
        a, b = r[k], r[k + 1]; h = b - a
        Vn[k] += 2 * np.pi * (a * h / 2 + h * h / 6); Vn[k + 1] += 2 * np.pi * (b * h / 2 - h * h / 6)
    kk = len(xi) - 1
    from math import erf
    I_end = nb * np.sqrt(np.pi / 2) * sxi * (1 + erf((xi[kk] + dx - x0) / (np.sqrt(2) * sxi)))
    prof = np.exp(-r ** 2 / (2 * sr ** 2))
    N_NTP = 101325 / (1.380649e-23 * 293.15) * 1e-6     # cm^-3
    worst = 0.0
    for spec in sys.argv[9:]:
        name, dens, sref, nref = spec.split(":")
        dens = float(dens)
        code = (f["nz_" + name][kk] / dens * Vn).sum()
        tot = lambda s20: (-np.expm1(-s20 * 1e-20 * n0 * kpi * beta * I_end * prof) * Vn).sum() - code
        sig = brentq(tot, 1e-2, 1e5, xtol=1e-10, rtol=1e-12) * 1e-20
        if sref == "-":
            cmp = "measured: not tabulated"
        else:
            sref = float(sref) * 1e-20
            worst = max(worst, abs(sig / sref - 1))
            cmp = f"measured {sref:.3e}  (rel. diff {sig / sref - 1:+.1e})"
        print(f"  {name:3s}: sigma code {sig:.4e} cm^2  {cmp};"
              f"  primary ion pairs/cm at NTP: {sig * N_NTP:5.1f}  (measured ~{nref})")
    print(f"  largest |rel. diff| of sigma: {worst:.1e}")
