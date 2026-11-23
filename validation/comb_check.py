"""checks of the comb laser input (laser.comb_a) against an independent evaluation of its definition
usage: comb_check.py init DIR       initial envelope vs the analytic sum; line energies (radial LG_p projections) vs the
                                     intensity I = eps0 c E0^2 / 2 = 1.368e18 a^2 / lambda^2 W/cm^2 (drive energy of case lo6: 29.1 mJ;
                                     the deck generator ps_deck.py rounds the coefficient to 1.38e18 and prints 29.4 mJ)
       comb_check.py drift DIR k0 w dt dxi scheme dw0 p0 dw1 p1
                                     two weak lines: phase rates and their difference (beat slip) vs the discrete
                                     dispersion relation of the envelope scheme (Crank-Nicolson in t, scheme in xi)"""
import cmath
import math
import os
import re
import sys

import numpy as np
from scipy.special import eval_laguerre

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import head_first, read_fields  # noqa: E402



def lg(p, r, w):
    return eval_laguerre(p, 2 * r ** 2 / w ** 2) * np.exp(-r ** 2 / w ** 2)


def project(A, r, p, w):   # amplitude of LG_p in A(r) (weights r dr; LG_p orthogonal in the plane)
    m = lg(p, r, w); dv = r * np.gradient(r)
    return (A * (m * dv)[None, :]).sum(axis=1) / (m * m * dv).sum()


what = sys.argv[1]
if what == "init":
    d = sys.argv[2]
    f = head_first(read_fields(os.path.join(d, "fields_000000.bin"))); xi, r = f["s"], f["r"]   # s = -xi (laser_comb.in: xi0 = -55)
    A = f["a_re"] + 1j * f["a_im"]
    # the deck of the validation (laser_comb.in), evaluated independently
    n0, lam0 = 6e18, 1.03
    a = [0.02, 0.04, 0.04, 0.04, 0.04]; lam = [1.213347, 1.114181, 1.03, 0.957646, 0.89479]; p = [0, 1, 2, 3, 4]
    w, fwhm, sg, xi0, chi0 = 9.2582, 69.08, 8.0, 55.0, 1.0
    e, me, eps0, c = 1.602176634e-19, 9.1093837015e-31, 8.8541878128e-12, 299792458.0
    wp = math.sqrt(n0 * 1e6 * e * e / (eps0 * me)); kp_um = wp / c * 1e-6
    k0 = 2 * math.pi / lam0 / kp_um
    om = [2 * math.pi / l / kp_um for l in lam]
    K = [chi0 + 4 * (2 * q + 1) / w ** 2 for q in p]
    kap = [0.5 * (o + math.sqrt(o * o - 2 * k)) - k0 for o, k in zip(om, K)]
    L = fwhm / 2 / (math.log(2) / 2) ** (1 / sg)
    env = np.exp(-np.abs((xi - xi0) / L) ** sg)
    th = sum(ai * env[:, None] * lg(q, r, w)[None, :] * np.exp(-1j * kk * (xi - xi0))[:, None] for ai, q, kk in zip(a, p, kap))
    th[:, -1] = 0
    err = np.abs(A - th).max() / np.abs(th).max()
    ax = np.abs(np.abs(A[:, 0]) ** 2 - np.abs(th[:, 0]) ** 2).max() / (np.abs(th[:, 0]) ** 2).max()
    # line energies: LG_p projection (separates the lines), int |c_q|^2 dxi * pi w^2 / 2 * (omega_q m c / e)^2 eps0 / 2
    dxi = xi[1] - xi[0]
    Ucode, Uth = [], []
    for ai, q, o, l in zip(a, p, om, lam):
        cq = project(A, r, q, w)
        Ucode.append(0.5 * eps0 * (me * c * wp * o / e) ** 2 * (np.abs(cq) ** 2).sum() * dxi * math.pi * w ** 2 / 2 * (c / wp) ** 3)
        I = 0.5 * eps0 * c * (me * c * 2 * math.pi * c / (l * 1e-6) * ai / e) ** 2   # W/m^2 = 1.368e18 a^2/lambda^2 W/cm^2 (linear)
        flu = (env ** 2).sum() * dxi / kp_um * 1e-6 / c           # s
        Uth.append(I * math.pi * (w / kp_um * 1e-6) ** 2 / 2 * flu)
    rel = max(abs(u / v - 1) for u, v in zip(Ucode, Uth))
    print(f"  envelope vs analytic sum: max error {err:.1e} (on axis |a|^2: {ax:.1e}); line energies "
          + " ".join(f"{u * 1e3:.3f}" for u in Ucode) + f" mJ, total {sum(Ucode) * 1e3:.2f} mJ (intensity formula: "
          f"{sum(Uth) * 1e3:.2f} mJ, lines within {rel:.1e})")
elif what == "drift":
    d = sys.argv[2]; k0, w, dt, dxi = [float(v) for v in sys.argv[3:7]]; scheme = sys.argv[7]
    lines = [(float(sys.argv[8]), int(sys.argv[9])), (float(sys.argv[10]), int(sys.argv[11]))]
    fs = sorted(x for x in os.listdir(d) if x.startswith("fields_"))
    t, C = [], []
    for fn in fs:
        f = head_first(read_fields(os.path.join(d, fn))); r = f["r"]; k = len(f["xi"]) // 2   # slices from the head
        A = (f["a_re"] + 1j * f["a_im"])[k:k + 1]
        t.append(f["t"]); C.append([project(A, r, q, w)[0] for _, q in lines])
    t, C = np.array(t), np.array(C)
    nu = [np.polyfit(t, np.unwrap(np.angle(C[:, i])), 1)[0] for i in range(2)]
    log = open(d + ".log").read()
    kap = [float(v) for v in re.findall(r"wavenumber k0 ([-+.0-9e]+)", log)]
    pred = []
    for (dw, q), kk in zip(lines, kap):
        z = cmath.exp(1j * kk * dxi)
        D = {"bdf2": (3 - 4 * z + z * z) / (2 * dxi), "trapezoid": 2 / dxi * (1 - z) / (1 + z), "euler": (1 - z) / dxi}[scheme]
        Kq = 1 + 4 * (2 * q + 1) / w ** 2
        cc = Kq * dt / (4 * (1j * k0 - D)); g = (1 + cc) / (1 - cc)
        pred.append(cmath.phase(g) / dt)
    print(f"  {scheme}: phase rates " + " ".join(f"{a:+.6f}" for a in nu) + "  (theory " + " ".join(f"{b:+.6f}" for b in pred)
          + f"); beat slip {nu[1] - nu[0]:+.3e} vs {pred[1] - pred[0]:+.3e} per 1/omega_p "
          f"(difference {abs((nu[1] - nu[0]) - (pred[1] - pred[0])):.1e})")
