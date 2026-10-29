"""Checks of the delta electrons of beam impact ionization (<species>.impact.secondary_energy) against an
independent implementation of the model (Ionization.hpp): energy spectrum dsigma/dW ~ 1/(J^2 + W^2) on
[0, W_max], polar angle of a binary collision cos(theta) = kappa sqrt(W/(W + 2)), or isotropic emission.
The parents are atoms at rest, so the birth momentum is the delta-electron momentum.

usage:
  delta_check.py spectrum DIR SPECIES J_eV beam_mass beam_charge beam_gamma  kinematic|isotropic
      born_<SPECIES>_*.txt (impact-born electrons): Kolmogorov-Smirnov test of W against the CDF
      atan(W/J)/atan(W_max/J), W <= W_max, and the angle: |cos(theta) - kappa sqrt(W/(W+2))| (kinematic)
      or a KS test of cos(theta) against the uniform distribution (isotropic).
  delta_check.py trapped DIR SPECIES max_qsa
      fast delta electrons (gamma/Delta > max_qsa at birth) -> beam particles of <product>_trapped: the
      number of the born electrons beyond the limit vs the trapped macro-particles in beams.txt (the
      conversion tests the state after the first push, so births within 0.1 % of the limit may differ).
  delta_check.py same DIR_ON DIR_OFF SPECIES
      the ion charge density nz_<SPECIES> with and without delta electrons (expected: bit-identical; the
      ionization rate does not depend on the electron energy).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402

ME_EV = 510998.95


def born(d, sp):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"born_{sp}_*.txt*"))):
        a = np.loadtxt(f, ndmin=2)
        if a.size: rows.append(a)
    a = np.vstack(rows)
    px, py, dl, w, ib = a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    p2 = px ** 2 + py ** 2
    gam = (1 + p2 + dl ** 2) / (2 * dl)
    uz = (1 + p2 - dl ** 2) / (2 * dl)
    return gam, uz, p2, dl, w, ib


def ks(x, cdf):
    """Kolmogorov-Smirnov distance and the asymptotic p-value"""
    x = np.sort(x)
    n = len(x)
    F = cdf(x)
    D = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))
    return D, p


what = sys.argv[1]
if what == "spectrum":
    d, sp, J = sys.argv[2], sys.argv[3], float(sys.argv[4]) / ME_EV
    M, q, g = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    mode = sys.argv[8]
    u2 = g * g - 1
    if M == 1: Wm = 0.5 * (g - 1) if q < 0 else g - 1
    else: Wm = 2 * u2 / (1 + 2 * g / M + 1 / M ** 2)
    kap = (g + 1 / M) / np.sqrt(u2)
    gam, uz, p2, dl, w, ib = born(d, sp)
    sel = ib == 1
    W, u = gam[sel] - 1, np.sqrt(gam[sel] ** 2 - 1)
    c = uz[sel] / np.maximum(u, 1e-300)
    print(f"  {sel.sum()} impact-born electrons; W_max = {Wm * ME_EV / 1e6:.4g} MeV, kappa = {kap:.6f}, J = {J * ME_EV:g} eV")
    D, p = ks(W, lambda x: np.arctan(x / J) / np.arctan(Wm / J))
    print(f"  energy: KS distance {D:.2e}, p-value {p:.3f};  max W / W_max = {W.max() / Wm:.4f};"
          f"  median W {np.median(W) * ME_EV:.2f} eV (model {J * np.tan(0.5 * np.arctan(Wm / J)) * ME_EV:.2f} eV)")
    m = W > 1e-6   # below, round-off of gamma - 1 dominates the angle
    if mode == "kinematic":
        dev = np.abs(c[m] - np.minimum(kap * np.sqrt(W[m] / (W[m] + 2)), 1))
        print(f"  angle: max |cos(theta) - kappa sqrt(W/(W+2))| = {dev.max():.1e}  (W > 0.5 eV, {m.sum()} electrons)")
    else:
        D, p = ks(c[m], lambda x: 0.5 * (x + 1))
        print(f"  angle: isotropic, KS distance of cos(theta) {D:.2e}, p-value {p:.3f}; <cos(theta)> = {c[m].mean():+.4f}")
elif what == "trapped":
    d, sp, qsa = sys.argv[2], sys.argv[3], float(sys.argv[4])
    gam, uz, p2, dl, w, ib = born(d, sp)
    fast = gam / dl > qsa
    nb = 0
    for line in open(os.path.join(d, "beams.txt")):
        f = line.split()
        if f[0].endswith("_trapped"): nb = int(f[3])
    edge = ((gam / dl > qsa / 1.001) & (gam / dl < qsa * 1.001)).sum()
    ok = abs(int(fast.sum()) - nb) <= edge
    print(f"  {len(gam)} electrons born, {fast.sum()} beyond gamma/Delta = {qsa:g} at birth;"
          f" trapped beam particles at the end: {nb}")
    print(f"  (expected: equal up to the {edge} born within 0.1 % of the threshold: the conversion tests the state"
          f" after the first push) {'OK' if ok else 'MISMATCH'}")
elif what == "same":
    a, b, sp = sys.argv[2], sys.argv[3], sys.argv[4]
    worst = 0.0
    for f in sorted(glob.glob(os.path.join(a, "fields_*.bin"))):
        x = read_fields(f)["nz_" + sp]
        y = read_fields(os.path.join(b, os.path.basename(f)))["nz_" + sp]
        worst = max(worst, float(np.max(np.abs(x - y))))
    print(f"  ion charge density with / without delta electrons: max |difference| = {worst:.1e}  (expected: 0)")
