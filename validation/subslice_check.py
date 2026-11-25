"""Section 13: adaptive sub-slicing of the plasma push (pusher.max_cells_per_step).
Pinched witness + ion motion (paper case) on the stretched grid (axis cell 5e-4), box to xi = 12:
behind the bubble the plasma electrons cross the axis through the small cells.
usage: subslice_check.py ref_dir plain_dir subsliced_dir     (axis_000000.txt of each)
Prints the r.m.s. errors of E_z(0) and psi(0) behind the closure spike (xi -11.9..-9.2) and around
the witness (-7.2..-6.4), relative to the fine-step reference, and the integral of E_z over the
closure spike (a near-singular caustic: only its integral is meaningful)."""
import re
import sys

import numpy as np


def load(d):
    a = np.loadtxt(d + "/axis_000000.txt", comments="#")[::-1]   # from the head
    a[:, 0] *= -1   # s = -xi: distance behind the head of the box
    return a[:, 0], a[:, 1], a[:, 2]


# the windows below are in s = -xi (distance behind the head), as returned by load()
def l2(x0, f0, x1, f1, a, b):
    xs = np.arange(a, b, 0.005)
    r0, r1 = np.interp(xs, x0, f0), np.interp(xs, x1, f1)
    return np.sqrt(np.mean((r1 - r0) ** 2) / np.mean(r0 ** 2))


def spike(x, e):
    m = (x >= 8.9) & (x <= 9.2)
    return np.trapezoid(e[m], x[m])


ref, runs = sys.argv[1], sys.argv[2:]
xr, er, pr = load(ref)
print("  reference %s: integral of Ez over the closure spike (xi -9.2..-8.9) = %.3f" % (ref, spike(xr, er)))
for d in runs:
    x, e, p = load(d)
    try:
        n = re.search(r"\((\d+) extra sub-slices\)", open(d + ".log").read())
        n = int(n.group(1)) if n else 0
    except OSError:
        n = 0
    print("  %-14s extra sub-slices %5d | behind the spike: Ez %.1e psi %.1e | witness: Ez %.1e psi %.1e | spike integral %.3f"
          % (d, n, l2(xr, er, x, e, 9.2, 11.9), l2(xr, pr, x, p, 9.2, 11.9),
             l2(xr, er, x, e, 6.4, 7.2), l2(xr, pr, x, p, 6.4, 7.2), spike(x, e)))
