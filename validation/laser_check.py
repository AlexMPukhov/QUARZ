"""checks of the laser envelope solver against analytic results
usage: laser_check.py vacuum DIR k0 w0 focus a0 | vortex DIR k0 w0 focus a0 l | linear DIR a0 w0 L0 xi0 | channel DIR k0 w0 | selffocus DIR a0 [DIR a0 ...]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402


def ctrap(y, x):
    return np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


what = sys.argv[1]
if what == "vacuum":   # Gaussian beam: w(t) = w0 sqrt(1 + ((t - zf)/zR)^2), a = a0 / sqrt(...)
    d = sys.argv[2]; k0, w0, zf, a0 = [float(v) for v in sys.argv[3:7]]
    a = np.loadtxt(d + "/laser.txt"); t = a[:, 1]
    zR = k0 * w0 ** 2 / 2; s = (t - zf) / zR
    w_th = w0 * np.sqrt(1 + s ** 2); a_th = a0 / np.sqrt(1 + s ** 2)
    slip = a[-1, 4] - a[0, 4]
    print(f"  spot size {np.abs(a[:, 3] / w_th - 1).max():.1e}, peak amplitude {np.abs(a[:, 2] / a_th - 1).max():.1e}, "
          f"int |a|^2 conserved to {np.abs(a[:, 6] / a[0, 6] - 1).max():.1e}; centroid slip {slip:.4f} "
          f"(group velocity 1 - 1/(k0 w0)^2: {t[-1] / (k0 * w0) ** 2:.4f})")
elif what == "vortex":   # LG_0l through the focus: w_rms = w(t) sqrt(l + 1), max|a| = a0 / sqrt(1 + s^2), profile
    d = sys.argv[2]; k0, w0, zf, a0 = [float(v) for v in sys.argv[3:7]]; l = int(sys.argv[7])
    a = np.loadtxt(d + "/laser.txt"); t = a[:, 1]
    zR = k0 * w0 ** 2 / 2; s = (t - zf) / zR
    w_th = w0 * np.sqrt(1 + s ** 2) * np.sqrt(l + 1); a_th = a0 / np.sqrt(1 + s ** 2)
    slip = a[-1, 4] - a[0, 4]
    # complex profile of the last field file at the pulse centre vs a0 N_l (sqrt2 r/w0)^l q^-(l+1) exp(-r^2/(w0^2 q))
    fn = sorted(f for f in os.listdir(d) if f.startswith("fields_"))[-1]
    f = read_fields(os.path.join(d, fn)); r = f["r"]
    A = f["a_re"] + 1j * f["a_im"]
    k = np.argmax(np.abs(A).max(axis=1))
    q = 1 + 1j * (f["t"] - zf) / zR
    th = a0 * (np.e / l) ** (l / 2) * (np.sqrt(2) * r / w0) ** l * q ** (-(l + 1)) * np.exp(-r ** 2 / (w0 ** 2 * q))
    c = np.vdot(th, A[k]) / np.vdot(th, th)   # best complex factor (the longitudinal envelope at this slice)
    err = np.linalg.norm(A[k] - c * th) / np.linalg.norm(A[k])
    print(f"  l = {l}: spot size w_rms/(w sqrt(l+1)) {np.abs(a[:, 3] / w_th - 1).max():.1e}, peak amplitude "
          f"{np.abs(a[:, 2] / a_th - 1).max():.1e}, int |a|^2 conserved to {np.abs(a[:, 6] / a[0, 6] - 1).max():.1e}; "
          f"profile at t = {f['t']:g}: rel. error {err:.1e}, phase {np.angle(c):+.1e} (Gouy phase (l+1) atan s included); "
          f"centroid slip {slip:.4f} (group velocity: (l+1) t/(k0 w0)^2 = {(l + 1) * t[-1] / (k0 * w0) ** 2:.4f})")
elif what == "linear":   # (d^2/dxi^2 + 1) psi = <a^2>/2
    d = sys.argv[2]; a0, w0, L0, x0 = [float(v) for v in sys.argv[3:7]]
    f = read_fields(d + "/fields_000000.bin"); xi = f["xi"]; r = f["r"]
    out = []
    for rr in (0.0, 1.5, 3.0):
        j = np.argmin(abs(r - rr))
        src = 0.25 * (a0 * np.exp(-r[j] ** 2 / w0 ** 2 - (xi - x0) ** 2 / L0 ** 2)) ** 2
        C = ctrap(np.cos(xi) * src, xi); S = ctrap(np.sin(xi) * src, xi)
        psi = np.sin(xi) * C - np.cos(xi) * S
        ez = np.cos(xi) * C + np.sin(xi) * S
        out.append(f"r={r[j]:.1f}: psi {np.abs(f['psi'][:, j] - psi).max() / np.abs(psi).max():.1e}, "
                   f"Ez {np.abs(f['ez'][:, j] - ez).max() / np.abs(ez).max():.1e}")
    print("  " + ";  ".join(out))
elif what == "channel":   # matched parabolic channel: w = w0; slip (1 + 4/w0^2) t / (2 k0^2)
    d = sys.argv[2]; k0, w0 = [float(v) for v in sys.argv[3:5]]
    a = np.loadtxt(d + "/laser.txt"); t = a[-1, 1]
    print(f"  spot size within {np.abs(a[:, 3] / w0 - 1).max():.1e} of w0 over {t / (k0 * w0 ** 2 / 2):.1f} Rayleigh lengths; "
          f"slip {a[-1, 4] - a[0, 4]:.4f} (theory {(1 + 4 / w0 ** 2) * t / (2 * k0 ** 2):.4f})")
elif what == "selffocus":
    for d, a0 in zip(sys.argv[2::2], sys.argv[3::2]):
        res = []
        for n in (0, 10, 20, 30, 40):
            f = read_fields(f"{d}/fields_{n:06d}.bin"); k = np.argmin(abs(f["xi"] - 30))
            res.append(np.hypot(f["a_re"][k, 0], f["a_im"][k, 0]) / float(a0))
        print(f"  {d}: a(r=0, pulse centre)/a0 at t = 0, Z_R/4, Z_R/2, 3Z_R/4, Z_R: " + " ".join(f"{v:.3f}" for v in res))
