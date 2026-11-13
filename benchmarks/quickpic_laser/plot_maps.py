"""Maps of |a| and n_e in the (xi, x) plane for the strongly nonlinear case (case2.in) at s = 20, 50, 100, 150, 200
for the xi schemes (trapezoid, BDF2, backward Euler), and a noise measure of the envelope: the energy of |a|^2 at
transverse wavenumbers above 2/w0 relative to the total (on the slices with |a| > 0.5 max), which grows if the
envelope solver amplifies noise.
usage: python3 plot_maps.py [case] -> maps_case<case>.png, noise on stdout"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools"))
from quarz_read import read_fields, xz_plane  # noqa: E402

case = int(sys.argv[1]) if len(sys.argv) > 1 else 2
S = [20, 50, 100, 150, 200] if case == 2 else [50, 100, 200, 300, 400, 500]
schemes = ["trap", "bdf2", "euler"]
fig, axs = plt.subplots(len(schemes), len(S), figsize=(2.6 * len(S), 2.4 * len(schemes)), sharex=True, sharey=True)
for i, sch in enumerate(schemes):
    for j, s in enumerate(S):
        fn = os.path.join(here, "out_c%d_%s" % (case, sch), "fields_%06d.bin" % round(s / 5))
        ax = axs[i, j]
        if not os.path.exists(fn):
            ax.set_title("missing", fontsize=7)
            continue
        f = read_fields(fn)
        x, ne = xz_plane(f, "ne")
        a = np.abs(f["a_re"] + 1j * f["a_im"])
        aa = np.concatenate([a[:, :0:-1], a], axis=1)
        ax.pcolormesh(f["xi"], x, -ne.T, cmap="Blues_r", vmin=-10, vmax=0, shading="auto")
        am = np.ma.masked_less(aa.T, 0.05)
        ax.pcolormesh(f["xi"], x, am, cmap="hot_r", vmin=0, vmax=6 if case == 2 else 3, shading="auto", alpha=0.8)
        ax.set_ylim(-15, 15)
        if i == 0:
            ax.set_title("s = %d" % s, fontsize=9)
        if j == 0:
            ax.set_ylabel("%s\n$k_p x$" % sch, fontsize=8)
        if i == len(schemes) - 1:
            ax.set_xlabel(r"$k_p\xi$", fontsize=8)
        # noise measure: radial "high-k" content of |a|^2 via the second difference
        r = f["r"]
        I = a ** 2
        lap = np.abs(I[:, 2:] - 2 * I[:, 1:-1] + I[:, :-2]) / (r[1] - r[0]) ** 2
        rows = a.max(axis=1) > 0.5 * a.max()
        print("case %d %-5s s = %3d  max|a| = %.3f  <|d2 |a|^2/dr2|> / max|a|^2 = %.4f" %
              (case, sch, s, a.max(), lap[rows].mean() / a.max() ** 2))
fig.tight_layout()
fig.savefig(os.path.join(here, "maps_case%d.png" % case), dpi=110)
