#!/usr/bin/env python3
"""Write a LASY file (thetaMode, normalized vector potential, lasy's layout) of a Gaussian pulse at its focus in QUARZ units.

    python3 tools/make_lasy_gaussian.py FILE k0 w0 L0 a0 [n0_cm3]
envelope a(u, r) = a0 exp(-r^2/w0^2) exp(-u^2/L0^2) at the file plane, carrier omega_L = k0 omega_p (u in 1/omega_p,
r in 1/k_p, n0_cm3 default 1e17: the SI units of the file). Used by validation section 11 (laser.file)."""
import sys
import numpy as np
import h5py

fn = sys.argv[1]
k0, w0, L0, a0 = (float(x) for x in sys.argv[2:6])
n0 = float(sys.argv[6]) if len(sys.argv) > 6 else 1e17
wp = 5.64146e4 * np.sqrt(n0)
kp = wp / 299792458.0
du, dr = L0 / 75, w0 / 100
u = np.arange(-4 * L0, 4 * L0 + du / 2, du)
r = np.arange(0, 5 * w0 + dr / 2, dr)
env = a0 * np.exp(-r[None, :] ** 2 / w0 ** 2) * np.exp(-u[:, None] ** 2 / L0 ** 2)
with h5py.File(fn, "w") as f:
    for k, v in dict(openPMD="1.1.0", basePath="/data/%T/", meshesPath="meshes/", software="lasy", softwareVersion="0.6.2").items():
        f.attrs[k] = np.bytes_(v)
    d = f.create_dataset("data/0/meshes/laserEnvelope", data=env[None].astype(np.complex128))
    for k, v in dict(geometry="thetaMode", dataOrder="C", envelopeField="normalized_vector_potential").items():
        d.attrs[k] = np.bytes_(v)
    d.attrs["axisLabels"] = np.array([b"t", b"r"])
    d.attrs["gridSpacing"] = np.array([du / wp, dr / kp])
    d.attrs["gridGlobalOffset"] = np.array([u[0] / wp, 0.0])
    d.attrs["position"] = np.array([0.0, 0.0])
    d.attrs["gridUnitSI"] = 1.0
    d.attrs["unitSI"] = 1.0
    d.attrs["angularFrequency"] = k0 * wp
    d.attrs["polarization"] = np.array([1.0, 0.0])
