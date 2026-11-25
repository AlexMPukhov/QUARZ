"""Compare QUARZ (trapezoid and BDF2 in xi), GEM-PIC-RZ and FBPIC on the laser wakefield cases of Meng et al.,
PPCF (2026), doi 10.1088/1361-6587/aebd1b, and, if present, curves digitized from their figures (digit_*.txt:
xi, E_z OSIRIS, E_z QuickPIC-FD; not distributed with QUARZ).

usage: python3 compare.py [--quarz DIR] [--fbpic DIR] [--gem DIR] [--digit DIR]
       defaults: this directory (out_c1_trap, ...), ../../../fbpic, ../../../gempic/benchmarks/quickpic_laser, ../../..
Output: compare_case1.png, compare_case2.png, laser_case<n>.txt (laser diagnostics) and the error table on stdout.
Error measure: the normalised rms error of the on-axis E_z, rms(E_z - E_z,ref) / max|E_z,ref| over the xi range."""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools"))
from quarz_read import head_first, read_fields  # noqa: E402


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


QDIR = opt("--quarz", here)
FDIR = opt("--fbpic", os.path.join(here, "..", "..", "..", "fbpic"))
GDIR = opt("--gem", os.path.join(here, "..", "..", "..", "gempic", "benchmarks", "quickpic_laser"))
DDIR = opt("--digit", os.path.join(here, "..", "..", ".."))
CASES = {1: dict(k0=34.0, dt=5.0, gdz=0.5, s=[50, 100, 200, 300, 400, 500], digit={50: "fig1b", 500: "fig1d"}, xr=(-4, 11), xf=(-4, 5.5)),
         2: dict(k0=20.0, dt=5.0, gdz=0.5, s=[20, 50, 100, 150, 200], digit={200: "fig3b"}, xr=(-4, 7), xf=(-4, 7))}
XI = np.arange(-4.0, 11.0001, 0.01)
# QUARZ runs: out_c<case>_<name>; trap, bdf2, euler = the decks with laser.xi_scheme; intfwhm = tau as intensity FWHM;
# case 3: trap = peak |a| = 2.1, energy = a0 of the same energy as a Gaussian (2.1 e^{-1/2});
# fine = dxi/2, dr/2, ds/2, 2x ppc
QRUNS = ["trap", "bdf2", "euler", "intfwhm", "fine"]


def qdt(scheme):
    return 2.5 if "fine" in scheme else 5.0


def quarz_axis(case, scheme, s):
    fn = os.path.join(QDIR, "out_c%d_%s" % (case, scheme), "axis_%06d.txt" % round(s / qdt(scheme)))
    if not os.path.exists(fn):
        return None
    d = np.loadtxt(fn)[::-1]   # QUARZ writes xi = z - t (head last); here xi = t - z as FBPIC/the paper
    return -d[:, 0], d[:, 1]


def quarz_laser(case, scheme, s):
    """max |a^| and max |E_L| / k0 = max |(d_xi - i k0) a^| / k0 over the box, and int |E_L|^2 dV / k0^2"""
    fn = os.path.join(QDIR, "out_c%d_%s" % (case, scheme), "fields_%06d.bin" % round(s / qdt(scheme)))
    if not os.path.exists(fn):
        return None
    f = head_first(read_fields(fn))   # from the head; s = -xi = t - z, the xi of this comparison
    a = f["a_re"] + 1j * f["a_im"]
    k0 = CASES[case]["k0"]
    dxi = f["s"][1] - f["s"][0]
    E = (np.gradient(a, dxi, axis=0) - 1j * k0 * a) / k0
    r = f["r"]
    dr = np.diff(r)
    V = np.zeros_like(r)                       # ring volumes of the hat functions
    V[:-1] += np.pi * dr * (2 * r[:-1] + dr) / 3
    V[1:] += np.pi * dr * (r[:-1] + 2 * r[1:]) / 3
    return np.abs(a).max(), np.abs(E).max(), (np.abs(E) ** 2 * V[None, :]).sum() * dxi, f["s"], np.abs(a[:, 0])


def fbpic(case, s):
    fn = os.path.join(FDIR, "snap_%d_s%03d.npz" % (case, s))
    if not os.path.exists(fn):
        return None
    d = np.load(fn)
    o = np.argsort(d["xi"])
    return d["xi"][o], d["Ez_axis"][o], d["a_axis"][o], float(d["U1"])


def gem(case, s):
    k0 = CASES[case]["k0"]
    fn = os.path.join(GDIR, "gem_case%d" % case, "axis_%05d.txt" % round(s / CASES[case]["gdz"]))
    if not os.path.exists(fn):
        return None
    d = np.loadtxt(fn)
    kp_um = 2 * np.pi / 0.8 / k0                     # k_p in 1/um (lambda0 = 0.8 um)
    xi = 12.0 - d[:, 0] * kp_um
    o = np.argsort(xi)
    from scipy.signal import hilbert
    env = np.abs(hilbert(d[:, 1]))
    return xi[o], d[o, 2] * k0, env[o]


def digit(name):
    fn = os.path.join(DDIR, "digit_%s.txt" % name)
    if not os.path.exists(fn):
        return None
    d = np.loadtxt(fn)
    out = []
    for col in (1, 2):
        m = np.isfinite(d[:, col])
        out.append((d[m, 0], d[m, col]))
    return out


def on(xi, y, x=XI):
    return np.interp(x, xi, y, left=np.nan, right=np.nan)


def err(a, ref, xr):
    m = (XI >= xr[0]) & (XI <= xr[1]) & np.isfinite(a) & np.isfinite(ref)
    return np.sqrt(np.mean((a[m] - ref[m]) ** 2)) / np.abs(ref[m]).max()


for case, P in CASES.items():
    print("\n=== case %d (k0/kp = %g), E_z on the axis; error = rms(dEz)/max|Ez_ref| over xi in [%g, %g] ===" %
          (case, P["k0"], *P["xr"]))
    shown = [s for s in P["s"] if s in P["digit"]] or P["s"][-1:]
    fig, axs = plt.subplots(len(shown), 1, figsize=(8, 3.2 * len(shown)), squeeze=False)
    for ip, s in enumerate(P["s"]):
        curves = {}
        for sch in QRUNS:
            q = quarz_axis(case, sch, s)
            if q:
                curves["QUARZ " + sch] = on(*q)
        g = gem(case, s)
        if g:
            curves["GEM-PIC-RZ"] = on(g[0], g[1])
        fb = fbpic(case, s)
        if fb:
            curves["FBPIC"] = on(fb[0], fb[1])
        dg = digit(P["digit"][s]) if s in P["digit"] else None
        if dg:
            curves["OSIRIS (digitized)"] = on(*dg[0])
            curves["QuickPIC-FD (digitized)"] = on(*dg[1])
        refs = [r for r in ("OSIRIS (digitized)", "FBPIC") if r in curves]
        for ref in refs:
            for k, v in curves.items():
                if k != ref:
                    print("  s = %3d  %-24s vs %-19s: %5.2f %%   (front, xi in [%g, %g]: %5.2f %%)" % (s, k, ref, 100 * err(v, curves[ref], P["xr"]), *P["xf"], 100 * err(v, curves[ref], P["xf"])))
        if s in shown:
            ax = axs[shown.index(s), 0]
            sty = {"OSIRIS (digitized)": ("k", "-", 2.0), "QuickPIC-FD (digitized)": ("tab:red", "-.", 1.2),
                   "FBPIC": ("tab:gray", "-", 1.0), "GEM-PIC-RZ": ("tab:green", "-", 1.0),
                   "QUARZ trap": ("tab:blue", "--", 1.2), "QUARZ bdf2": ("tab:orange", ":", 1.2),
                   "QUARZ euler": ("tab:brown", "-.", 1.0), "QUARZ intfwhm": ("tab:purple", "--", 0.8), "QUARZ fine": ("tab:cyan", ":", 1.5)}
            for k, v in curves.items():
                c, ls, lw = sty[k]
                ax.plot(XI, v, color=c, ls=ls, lw=lw, label=k)
            ax.set_title("case %d, s = %d c/$\\omega_p$" % (case, s))
            ax.set_xlabel(r"$k_p\xi$")
            ax.set_ylabel(r"$E_z$ ($m c\omega_p/e$)")
            ax.set_xlim(-4, 11)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(QDIR, "compare_case%d.png" % case), dpi=120)
    # laser diagnostics
    rows = []
    U0q = quarz_laser(case, "trap", 0)
    fb0 = sorted(glob.glob(os.path.join(FDIR, "snap_%d_s*.npz" % case)))
    U0f = None
    for fn in fb0:
        d = np.load(fn)
        if float(d["s"]) < 0:
            U0f = float(d["U1"])
    print("  laser: s, QUARZ max|a| (trap, bdf2), max|E_L|/k0, U/U0 (|E_L|^2 dV); FBPIC max|E_x|/E_L on axis, U/U0; "
          "GEM-PIC-RZ max|E_x|/E_L on axis (FBPIC, GEM-PIC-RZ: |xi| < 3, the laser)")
    for s in P["s"]:
        qt, qb, fb, g = quarz_laser(case, "trap", s), quarz_laser(case, "bdf2", s), fbpic(case, s), gem(case, s)
        row = [s, qt[0] if qt else np.nan, qb[0] if qb else np.nan, qt[1] if qt else np.nan,
               qt[2] / U0q[2] if (qt and U0q) else np.nan,
               fb[2][np.abs(fb[0]) < 3].max() if fb else np.nan, fb[3] / U0f if (fb and U0f) else np.nan, g[2][np.abs(g[0]) < 3].max() if g else np.nan]
        rows.append(row)
        print("  s = %3d  %6.3f %6.3f  %6.3f  %6.4f | %6.3f %6.4f | %6.3f" % tuple(row))
    np.savetxt(os.path.join(QDIR, "laser_case%d.txt" % case), np.array(rows), fmt="%.5f",
               header="s  QUARZ_max|a|_trap  QUARZ_max|a|_bdf2  QUARZ_max|E_L|/k0  QUARZ_U/U0  FBPIC_max|Ex|/E_L  "
                      "FBPIC_U/U0  GEM_max|Ex|/E_L")

# ---- case 3: LG_01 vortex (Sec. 4.2), E_z along x = 4 c/omega_p (mode 0; the wake is axisymmetric)
R3 = 4.0
dig3 = {40: "fig6d", 60: "fig6e", 80: "fig6f"}
if os.path.isdir(os.path.join(QDIR, "out_c3_trap")):
    print("\n=== case 3 (LG_01, k0/kp = 20), E_z at r = %g; error over xi in [-4, 11] ===" % R3)
    fig, axs = plt.subplots(3, 1, figsize=(8, 9.6))
    for ip, s in enumerate([40, 60, 80]):
        curves = {}
        for run in ("trap", "bdf2", "energy"):
            fn = os.path.join(QDIR, "out_c3_%s" % run, "fields_%06d.bin" % round(s / 5))
            if os.path.exists(fn):
                f = read_fields(fn)
                curves["QUARZ " + run] = on(f["xi"], np.array([np.interp(R3, f["r"], row) for row in f["ez"]]))
        fn = os.path.join(FDIR, "snap_3_s%03d.npz" % s)
        if os.path.exists(fn):
            d = np.load(fn)
            o = np.argsort(d["xi"])
            curves["FBPIC"] = on(d["xi"][o], np.array([np.interp(R3, d["r"], row) for row in d["Ez"]])[o])
        dg = digit(dig3[s])
        if dg:
            curves["OSIRIS (digitized)"] = on(*dg[0])
            curves["QuickPIC-FD (digitized)"] = on(*dg[1])
        # the vortex laser has an m = 0 longitudinal field (d_x of a^ e^{i theta}) at r = 4: full-PIC E_z oscillates at the
        # laser period; second number: all curves averaged over one laser period 2 pi / k0
        nw = int(round(2 * np.pi / 20.0 / (XI[1] - XI[0])))
        avg = {k: np.convolve(np.nan_to_num(v), np.ones(nw) / nw, mode="same") * np.where(np.isfinite(v), 1, np.nan)
               for k, v in curves.items()}
        for ref in [r for r in ("OSIRIS (digitized)", "FBPIC") if r in curves]:
            for k, v in curves.items():
                if k != ref:
                    print("  s = %3d  %-24s vs %-19s: %5.2f %%   (averaged over a laser period: %5.2f %%)" %
                          (s, k, ref, 100 * err(v, curves[ref], (-4, 11)), 100 * err(avg[k], avg[ref], (-3.5, 10.5))))
        ax = axs[ip]
        for k, v in curves.items():
            ax.plot(XI, v, lw=1.2 if "QUARZ" in k else 1.0, ls="--" if "QUARZ" in k else "-", label=k)
        ax.set_title("case 3 (LG$_{01}$), s = %d c/$\omega_p$, $E_z$ at $k_p r$ = %g" % (s, R3))
        ax.set_xlabel(r"$k_p\xi$"); ax.set_ylabel(r"$E_z$ ($m c\omega_p/e$)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(QDIR, "compare_case3.png"), dpi=120)
