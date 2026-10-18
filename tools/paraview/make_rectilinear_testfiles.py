#!/usr/bin/env python3
"""Write small openPMD HDF5 files with rectilinear meshes (h5py only), for testing readers:

  cart3d_xyz.h5      cartesian 3D, axisLabels [x, y, z]; x and z explicit (openPMD 2.0 form: data sets
                     referenced by gridCoordinates), y regular
  cart3d_zyx.h5      the same field with axisLabels [z, y, x] (C order, readers must transpose); x explicit in
                     the interim form (attribute gridCoordinates_x)
  cart2d_xz.h5       cartesian 2D [x, z]; z explicit (data set)
  cart3d_regular.h5  cartesian 3D without explicit coordinates (regression: must stay a regular image)
  theta_dataset.h5   thetaMode m = 2 (modes 0, cos 1, sin 1), [r, z]; r explicit (data set), z regular
Field rho = analytic function of the coordinates, so a reader can be checked point by point.

    python3 make_rectilinear_testfiles.py OUTDIR
"""
import os
import sys

import h5py
import numpy as np


def s(x):
    return np.bytes_(x)


def series(path):
    f = h5py.File(path, "w")
    for k, v in dict(openPMD="1.1.0", basePath="/data/%T/", meshesPath="meshes/",
                     iterationEncoding="groupBased", iterationFormat="/data/%T/", software="QUARZ test",
                     softwareVersion="1", date="2026-10-18 00:00:00 +0000").items():
        f.attrs[k] = s(v)
    f.attrs["openPMDextension"] = np.uint32(0)
    it = f.create_group("data/0")
    it.attrs["time"] = 0.0
    it.attrs["dt"] = 1.0
    it.attrs["timeUnitSI"] = 1.0
    it.create_group("meshes")
    return f, it


def mesh(it, name, data, labels, geometry, spacing, offset, unit, coords=None, interim=None, params=None):
    d = it["meshes"].create_dataset(name, data=data)
    d.attrs["geometry"] = s(geometry)
    if params:
        d.attrs["geometryParameters"] = s(params)
    d.attrs["dataOrder"] = s("C")
    d.attrs["axisLabels"] = np.array([s(x) for x in labels])
    d.attrs["gridSpacing"] = np.array(spacing, float)
    d.attrs["gridGlobalOffset"] = np.array(offset, float)
    d.attrs["gridUnitSI"] = float(unit)
    d.attrs["unitDimension"] = np.zeros(7)
    d.attrs["timeOffset"] = 0.0
    d.attrs["position"] = np.zeros(len(labels))
    d.attrs["unitSI"] = 2.0
    if coords:   # openPMD 2.0 form: one data set per explicit axis, referenced by path
        paths = []
        for lab in labels:
            if lab in coords:
                p = it.name + "/gridCoordinates/" + name + "_" + lab
                it.create_dataset(p, data=np.asarray(coords[lab], float))
                paths.append(p)
            else:
                paths.append("")
        d.attrs["gridCoordinates"] = np.array([s(x) for x in paths])
    if interim:  # openPMD 1.1 interim form (QUARZ): attribute gridCoordinates_<label>
        for lab, c in interim.items():
            d.attrs["gridCoordinates_" + lab] = np.asarray(c, float)
        d.attrs["gridType"] = s("rectilinear")


def stretched(n, h0, q):
    return np.concatenate([[0.0], np.cumsum(h0 * q ** np.arange(n - 1))])


def rho(x, y, z):
    return 1.0 + np.sin(3 * x) * np.cos(2 * y) + 0.1 * z ** 2


def main(out):
    os.makedirs(out, exist_ok=True)
    unit = 1e-6
    nx, ny, nz = 7, 5, 9
    x = stretched(nx, 0.05, 1.3)                      # positions in grid units, relative to the offset
    y = np.arange(ny) * 0.2
    z = stretched(nz, 0.1, 1.2)
    off = [-0.3, -0.4, 2.0]
    X, Y, Z = np.meshgrid(off[0] + x, off[1] + y, off[2] + z, indexing="ij")
    field = rho(X, Y, Z) / 2.0                        # unitSI = 2

    f, it = series(os.path.join(out, "cart3d_xyz.h5"))
    mesh(it, "rho", field, ["x", "y", "z"], "cartesian", [0.05, 0.2, 0.1], off, unit, coords={"x": x, "z": z})
    f.close()

    f, it = series(os.path.join(out, "cart3d_regular.h5"))   # regular mesh: readers must keep vtkImageData
    xr, zr = np.arange(nx) * 0.05, np.arange(nz) * 0.1
    Xr, Yr, Zr = np.meshgrid(off[0] + xr, off[1] + y, off[2] + zr, indexing="ij")
    mesh(it, "rho", rho(Xr, Yr, Zr) / 2.0, ["x", "y", "z"], "cartesian", [0.05, 0.2, 0.1], off, unit)
    f.close()

    f, it = series(os.path.join(out, "cart3d_zyx.h5"))
    mesh(it, "rho", np.transpose(field, (2, 1, 0)).copy(), ["z", "y", "x"], "cartesian", [0.1, 0.2, 0.05],
         [off[2], off[1], off[0]], unit, coords={"z": z}, interim={"x": x})
    f.close()

    f, it = series(os.path.join(out, "cart2d_xz.h5"))
    X2, Z2 = np.meshgrid(off[0] + np.arange(nx) * 0.1, off[2] + z, indexing="ij")
    mesh(it, "rho", rho(X2, 0.0, Z2) / 2.0, ["x", "z"], "cartesian", [0.1, 0.1], [off[0], off[2]], unit,
         coords={"z": z})
    f.close()

    nr, nzt = 12, 10
    r = stretched(nr, 0.02, 1.25)
    zt = np.arange(nzt) * 0.3
    R, ZT = np.meshgrid(r, -1.0 + zt, indexing="ij")
    modes = np.stack([1.0 + R ** 2, R * np.cos(ZT), 0.5 * R * np.sin(ZT)]) / 2.0
    f, it = series(os.path.join(out, "theta_dataset.h5"))
    mesh(it, "rho", modes, ["r", "z"], "thetaMode", [0.02, 0.3], [0.0, -1.0], unit, coords={"r": r},
         params="m=2;imag=+")
    f.close()


if __name__ == "__main__":
    main(sys.argv[1])
