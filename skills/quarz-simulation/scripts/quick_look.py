#!/usr/bin/env python3
"""Quick look at a QUARZ run from its text output (axis_NNNNNN.txt, beams.txt); needs numpy only.

  python3 quick_look.py out                  # code units
  python3 quick_look.py out --density 7e14   # also metres, GV/m, particle numbers

Prints one line per output step (largest on-axis wake field and where it is, largest electron density on the
axis) and, for every beam, the first and the last line of beams.txt. With matplotlib it also writes
<dir>/quicklook.png: on-axis E_z of the first and the last output step, the largest E_z against the propagation
distance, and the beam moments against the distance.
Units of the files: lengths c/omega_p, times 1/omega_p, fields E0 = m c omega_p / e, densities n0;
xi = z - ct: the beams move towards +xi; the head of the box is at its largest xi (the last line of axis_*.txt).
"""
import argparse
import glob
import math
import os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--density", type=float, help="reference density n0 [cm^-3] for SI values")
a = p.parse_args()

kp = E0 = None
if a.density:
    kp = 5.3141e5 / math.sqrt(a.density) * 1e-2          # c/omega_p [m]
    E0 = 96.16 * math.sqrt(a.density) / 1e9              # GV/m

files = sorted(glob.glob(os.path.join(a.dir, "axis_*.txt")))
if not files:
    raise SystemExit(f"no axis_NNNNNN.txt in {a.dir}: is output.every > 0 and output.axis = 1?")

# ---- on-axis fields: columns xi  Ez(0)  psi(0)  ne(0)  rho_beam(0)  (Er-Bth)/r at the first node
print(f"# {a.dir}: on-axis wake field, one line per output step")
print("#  step        t" + ("     z[m]" if kp else "") + "   max|Ez|/E0" + ("  max|Ez|[GV/m]" if E0 else "") +
      "   at xi   max ne(0)/n0")
rows = []
for f in files:
    step = int(os.path.basename(f)[5:11])
    with open(f) as fh:
        t = float(fh.readline().split("=")[1])
    d = np.loadtxt(f)
    k = int(np.argmax(np.abs(d[:, 1])))
    rows.append((step, t, abs(d[k, 1]), d[k, 0], d[:, 3].max()))
    print(f"{step:7d} {t:9.5g}" + (f" {t * kp:8.4f}" if kp else "") + f" {abs(d[k, 1]):12.5g}" +
          (f" {abs(d[k, 1]) * E0:14.5g}" if E0 else "") + f" {d[k, 0]:8.2f} {d[:, 3].max():13.4g}")
rows = np.array(rows)

# ---- the wake along the box at the first output step: a jump far behind the head that is not there in the beam
#      density is noise of clustering plasma macro-particles (long boxes), not physics
d = np.loadtxt(files[0])
edges = np.linspace(d[:, 0].max(), d[:, 0].min(), 11)   # from the head (largest xi) to the tail
print(f"# {os.path.basename(files[0])}: max |Ez|/E0 and max ne/n0 on the axis in 10 windows from the head to the tail")
print("#   xi from " + " ".join(f"{e:8.1f}" for e in edges[:-1]))
sel = [(d[:, 0] <= hi) & (d[:, 0] >= lo) for hi, lo in zip(edges[:-1], edges[1:])]
print("#   |Ez|    " + " ".join(f"{np.abs(d[m, 1]).max():8.4f}" for m in sel))
print("#   ne      " + " ".join(f"{d[m, 3].max():8.2f}" for m in sel))


def read_ez(fn):
    """E_z map of a fields_NNNNNN.bin (version 3, see output_reference.md; version 2 of QUARZ v1 with xi = t - z
    converted): r, xi, ez[xi, r] or None."""
    with open(fn, "rb") as fh:
        ver, M, K, _ = np.fromfile(fh, dtype=np.int32, count=4)
        if ver not in (2, 3):
            return None
        np.fromfile(fh, dtype=np.float64, count=1)
        r = np.fromfile(fh, dtype=np.float64, count=M)
        xi = np.fromfile(fh, dtype=np.float64, count=K)
        for _ in range(np.fromfile(fh, dtype=np.int32, count=1)[0]):
            name = fh.read(16).split(b"\0")[0].decode()
            data = np.fromfile(fh, dtype=np.float64, count=M * K)
            if name == "ez":
                return (r, -xi[::-1], data.reshape(K, M)[::-1]) if ver == 2 else (r, xi, data.reshape(K, M))
    return None


# the axis node of a fine mesh is a noisy witness (one plasma ring at the axis gives a density spike and a jump of
# E_z there): judge the noise by the field off the axis, where the beam particles are
ff = files[0].replace("axis_", "fields_").replace(".txt", ".bin")
ez = read_ez(ff) if os.path.exists(ff) else None
if ez is not None and ez[0][-1] >= 0.5:
    r, xi, e = ez
    j = int(np.argmin(np.abs(r - 0.5)))
    w = r[r <= 1.0]
    mean = np.abs((e[:, :len(w)] * w).sum(1) / w.sum()) if r[-1] >= 1.0 else None
    sel2 = [(xi <= hi) & (xi >= lo) for hi, lo in zip(edges[:-1], edges[1:])]
    print(f"#   |Ez| at r = {r[j]:.2f}: " + " ".join(f"{np.abs(e[m, j]).max():8.4f}" if m.any() else "       -" for m in sel2))
    if mean is not None:
        print("#   |<Ez>| r < 1:   " + " ".join(f"{mean[m].max():8.4f}" if m.any() else "       -" for m in sel2))
else:
    print("#   (no field file of this step with 'ez' out to r = 0.5: the off-axis wake, the better witness of noise, is not shown)")

# ---- beams.txt: name step t alive npart charge gamma_mean gamma_rms xi_mean xi_rms r_rms emit_nx x_mean y_mean
#                 emit_ny sigma_x sigma_y
beams = {}
bf = os.path.join(a.dir, "beams.txt")
if os.path.exists(bf):
    with open(bf) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            w = line.split()
            beams.setdefault(w[0], []).append([float(x) for x in w[1:]])
    npart = a.density * (kp * 100) ** 3 if kp else None   # real particles per unit of the column 'charge'
    print("# beams (first and last step): step  t  alive  charge" + ("  particles" if npart else "") +
          "  <gamma>  rms gamma  <xi>  rms xi  sigma_x  emit_nx")
    for name, r in beams.items():
        r = np.array(r)
        for row in (r[0], r[-1]):
            print(f"{name:>12s} {int(row[0]):6d} {row[1]:9.5g} {int(row[2]):9d} {row[4]:10.5g}" +
                  (f" {row[4] * npart:10.4g}" if npart else "") +
                  f" {row[5]:10.6g} {row[6]:9.4g} {row[7]:9.5g} {row[8]:8.4g} {row[14]:8.4g} {row[10]:9.4g}")
        if r[-1, 4] < 0.99 * r[0, 4]:
            print(f"#   {name}: {100 * (1 - r[-1, 4] / r[0, 4]):.1f} % of the charge lost (outside the box from the start, left it through the tail, or hit the wall)")

# ---- figure (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("no figure: matplotlib is not installed (the tables above are complete)")
    raise SystemExit(0)

zs = kp if kp else 1.0
zl = "z [m]" if kp else "t [1/omega_p]"
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for f, c in ((files[0], "C7"), (files[-1], "C0")):
    d = np.loadtxt(f)
    ax[0, 0].plot(d[:, 0], d[:, 1], color=c, lw=0.6, label=os.path.basename(f))
ax[0, 0].set(xlabel="xi [c/omega_p] (head of the box at the right)", ylabel="E_z on the axis [E0]")
ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(rows[:, 1] * zs, rows[:, 2], "o-")
ax[0, 1].set(xlabel=zl, ylabel="max |E_z| on the axis [E0]")
for name, r in beams.items():
    r = np.array(r)
    ax[1, 0].plot(r[:, 1] * zs, r[:, 14], label=f"{name}: sigma_x")
    ax[1, 1].plot(r[:, 1] * zs, r[:, 5] / r[0, 5], label=f"{name}: <gamma> / <gamma>(0)")
ax[1, 0].set(xlabel=zl, ylabel="rms size [c/omega_p]")
ax[1, 1].set(xlabel=zl, ylabel="mean energy, relative")
for x in (ax[1, 0], ax[1, 1]):
    if beams:
        x.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(a.dir, "quicklook.png")
fig.savefig(out, dpi=130)
print(out)
