"""Quick-look plots of a QUARZ field file (optional; needs numpy + matplotlib).

usage:  python plot_fields.py out/fields_000000.bin [rmax]
Writes <file>.png: E_z, (E_r - B_theta) and n_e, plus the on-axis E_z.
With modes = 1 the maps are shown in the (xi, x) plane through the axis
(theta = 0 above, theta = pi below the axis), otherwise versus r.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from quarz_read import read_fields, xz_plane

fn = sys.argv[1]
rmax = float(sys.argv[2]) if len(sys.argv) > 2 else None
d = read_fields(fn)
r, xi = d["r"], d["xi"]
d["wr"] = d["er"] - d["bth"]
if d["mode1"]:
    d["wr_c"] = d["er_c"] - d["bth_c"]
fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
panels = [("ez", r"$E_z$", "RdBu_r", True), ("wr", r"$E_r-B_\theta$", "RdBu_r", True), ("ne", r"$n_e$", "Greys", False)]
for a, (name, title, cmap, sym) in zip(ax.flat[:3], panels):
    if d["mode1"]:
        y, f = xz_plane(d, name)
        if name == "wr":   # radial component: sign flips across the axis in the x-plane
            y, f0 = xz_plane(d, "wr")
            n = len(r)
            f = np.concatenate([-(d["wr"] - d["wr_c"])[:, :0:-1], d["wr"] + d["wr_c"]], axis=1)
            title = r"$W_x = E_x - B_y$"
        ylabel = r"$x$  [$c/\omega_p$]"
    else:
        y, f = r, d[name]
        ylabel = r"$r$  [$c/\omega_p$]"
    v = np.percentile(abs(f), 99.5)
    kw = dict(vmin=-v, vmax=v) if sym else dict(vmin=0, vmax=min(v, 5))
    m = a.pcolormesh(xi, y, f.T, shading="auto", cmap=cmap, **kw)
    a.set_xlabel(r"$\xi = z - t$  [$c/\omega_p$]"); a.set_ylabel(ylabel)
    a.set_title(title)
    if rmax: a.set_ylim(-rmax if d["mode1"] else 0, rmax)
    fig.colorbar(m, ax=a)
a = ax.flat[3]
a.plot(xi, d["ez"][:, 0], label=r"$E_z(r=0)$")
if d["mode1"]:
    a.plot(xi, d["er_c"][:, 0] - d["bth_c"][:, 0], label=r"$W_x(r=0)$ (m=1)")
a.set_xlabel(r"$\xi$"); a.legend(); a.set_title(f"t = {d['t']:g}")
fig.savefig(fn.replace(".bin", ".png"), dpi=130)
print("wrote", fn.replace(".bin", ".png"))
