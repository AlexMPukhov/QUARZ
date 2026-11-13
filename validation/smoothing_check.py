"""Section 14: regularization of the bubble-back singularity (plasma.smooth_length).
The criterion is the convergence of the fields AFTER the closure spike: the spike itself (the
axis caustic of a cold plasma) is unphysical, its height need not converge.
usage: smoothing_check.py ref_dir run_dir...
For each run: r.m.s. change of E_z(0) and psi(0) relative to ref_dir behind the bubble
(xi -11.9..-9.2) and around the witness (-7.2..-6.4); the E_z minimum and axis density maximum in the
closure region (8.5..9.6) are printed for information only."""
import re
import sys

import numpy as np


def load(d):
    a = np.loadtxt(d + "/axis_000000.txt", comments="#")[::-1]   # from the head
    a[:, 0] *= -1   # s = -xi: distance behind the head of the box
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# the windows below are in s = -xi (distance behind the head), as returned by load()
def l2(x0, f0, x1, f1, a, b):
    xs = np.arange(a, b, 0.005)
    r0, r1 = np.interp(xs, x0, f0), np.interp(xs, x1, f1)
    return np.sqrt(np.mean((r1 - r0) ** 2) / np.mean(r0 ** 2))


def label(d):
    return re.sub(r"^out_sm_", "", d)


xr, er, pr, nr = load(sys.argv[1])
m = (xr > 8.5) & (xr < 9.6)
print("  reference %-14s                                        (closure: min Ez %6.1f, max ne(0) %8.0f)"
      % (label(sys.argv[1]), er[m].min(), nr[m].max()))
for d in sys.argv[2:]:
    x, e, p, n = load(d)
    m = (x > 8.5) & (x < 9.6)
    print("  %-22s behind: Ez %.1e psi %.1e | witness: Ez %.1e psi %.1e  (closure: min Ez %6.1f, max ne(0) %8.0f)"
          % (label(d), l2(xr, er, x, e, 9.2, 11.9), l2(xr, pr, x, p, 9.2, 11.9),
             l2(xr, er, x, e, 6.4, 7.2), l2(xr, pr, x, p, 6.4, 7.2), e[m].min(), n[m].max()))
