"""checks of the laser envelope solver against analytic results
usage: laser_check.py vacuum DIR k0 w0 focus a0 | linear DIR a0 w0 L0 xi0 | channel DIR k0 w0 | selffocus DIR a0 [DIR a0 ...]"""
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
    slip = a[0, 4] - a[-1, 4]   # the pulse slips backwards (towards smaller xi)
    print(f"  spot size {np.abs(a[:, 3] / w_th - 1).max():.1e}, peak amplitude {np.abs(a[:, 2] / a_th - 1).max():.1e}, "
          f"int |a|^2 conserved to {np.abs(a[:, 6] / a[0, 6] - 1).max():.1e}; centroid slip {slip:.4f} "
          f"(group velocity 1 - 1/(k0 w0)^2: {t[-1] / (k0 * w0) ** 2:.4f})")
elif what == "linear":   # (d^2/ds^2 + 1) psi = <a^2>/2,  s = -xi (distance behind the head), E_z = d psi/ds
    d = sys.argv[2]; a0, w0, L0, x0 = [float(v) for v in sys.argv[3:7]]
    f = read_fields(d + "/fields_000000.bin"); r = f["r"]
    xi, x0 = -f["xi"][::-1], -x0   # integrate from the head: s increasing
    f = {k: f[k][::-1] for k in ("psi", "ez")}
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
          f"slip {a[0, 4] - a[-1, 4]:.4f} (theory {(1 + 4 / w0 ** 2) * t / (2 * k0 ** 2):.4f})")
elif what == "selffocus":
    for d, a0 in zip(sys.argv[2::2], sys.argv[3::2]):
        res = []
        for n in (0, 10, 20, 30, 40):
            f = read_fields(f"{d}/fields_{n:06d}.bin"); k = np.argmin(abs(f["xi"] + 30))
            res.append(np.hypot(f["a_re"][k, 0], f["a_im"][k, 0]) / float(a0))
        print(f"  {d}: a(r=0, pulse centre)/a0 at t = 0, Z_R/4, Z_R/2, 3Z_R/4, Z_R: " + " ".join(f"{v:.3f}" for v in res))
