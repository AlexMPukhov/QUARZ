"""Section 26: the longitudinal convention xi = z - t of the inputs and outputs (beams move towards +xi).
usage: convention_check.py OUT XI_MIN XI_MAX XI0 STEP_END
  (1) the field file covers [xi.min, xi.max] in increasing xi (the head of the box is the last row);
  (2) the beam charge per slice (field file) and the particle dump are centred at the deck's xi0;
  (3) between the dumps of steps 0 and STEP_END every particle slips by dxi = int (v_z - 1) dt < 0
      (trapezoidal rule over the two dumps), i.e. backwards, towards smaller xi."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_fields  # noqa: E402

out = sys.argv[1]
xmin, xmax, xi0 = map(float, sys.argv[2:5])
n1 = int(sys.argv[5])
f = read_fields(f"{out}/fields_000000.bin")
xi = f["xi"]
dx = xi[1] - xi[0]
ok1 = abs(xi[0] - xmin) < 1e-9 and abs(xi[-1] - xmax) < 1e-9 and np.all(np.diff(xi) > 0)
rho = np.abs(f["rhob"]) @ (np.gradient(f["r"]) * f["r"])   # beam charge per slice
xc = (rho * xi).sum() / rho.sum()
b0, b1 = read_beam(f"{out}/beam_witness_000000.bin"), read_beam(f"{out}/beam_witness_{n1:06d}.bin")
xb = np.average(b0["xi"], weights=b0["w"])
vz = lambda b: b["pz"] / np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)
T = read_fields(f"{out}/fields_{n1:06d}.bin")["t"] - f["t"]
slip = b1["xi"] - b0["xi"]
ref = 0.5 * (vz(b0) + vz(b1) - 2) * T
print(f"  field file: xi = {xi[0]:g} ... {xi[-1]:g}, increasing: {'OK' if ok1 else 'FAIL'};  "
      f"beam charge centred at {xc:.4f}, particles at {xb:.4f} (deck xi0 = {xi0:g})")
print(f"  slip over t = {T:g}: mean {slip.mean():.5f} (int (v_z - 1) dt: {ref.mean():.5f}), "
      f"max |slip - ref| / |ref| = {np.max(np.abs(slip - ref) / np.abs(ref)):.1e}")
