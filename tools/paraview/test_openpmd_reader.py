#!/usr/bin/env python3
"""Test ParaView's openPMD reader (paraview/algorithms/openpmd.py) on rectilinear meshes without
ParaView: the reader is a VTKPythonAlgorithmBase and runs with the `vtk` Python package; a stub
`paraview` package provides print_error.

    python3 test_openpmd_reader.py READER.py CASE...      (READER.py: openpmd.py or openPMDRectilinearReader.py)

CASE = quarz:<series pattern>  (QUARZ native thetaMode output, e.g. out/openpmd/data_%06T.h5)
     | cart:<file>             (a cartesian test file written by make_rectilinear_testfiles.py)
     | theta:<file>            (a thetaMode test file written by make_rectilinear_testfiles.py)
     | regular:<file>          (a regular cartesian file: must stay vtkImageData)
With BLOCK_H5PY_IN_READER=1 the reader cannot fall back to h5py (tests openPMD-api's coordinate data sets).
Checks the point coordinates and the field values against tools/openpmd_rectilinear.py (h5py), an
independent reader, and prints PASS/FAIL per case."""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from openpmd_rectilinear import read_mesh  # noqa: E402


def load_reader(reader_py):
    """The reader class from ParaView's paraview/algorithms/openpmd.py, or from the standalone plugin
    openPMDRectilinearReader.py (its ParaView decorators are stubbed)."""
    tmp = tempfile.mkdtemp()
    pkg = os.path.join(tmp, "paraview")
    os.makedirs(os.path.join(pkg, "algorithms"))
    os.makedirs(os.path.join(pkg, "util"))
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("def print_error(*a):\n    import sys\n    print('print_error:', *a, file=sys.stderr)\n")
    open(os.path.join(pkg, "algorithms", "__init__.py"), "w").close()
    open(os.path.join(pkg, "util", "__init__.py"), "w").close()
    with open(os.path.join(pkg, "util", "vtkAlgorithm.py"), "w") as f:   # no-op ParaView decorators
        f.write("from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase\n"
                "class _D:\n"
                "    def __getattr__(self, name):\n"
                "        return lambda *a, **k: (lambda f: f)\n"
                "smproxy = smproperty = smdomain = smhint = _D()\n")
    sys.path.insert(0, tmp)
    if "smproxy.reader" in open(reader_py).read():   # standalone plugin
        shutil.copy(reader_py, os.path.join(tmp, "pv_plugin.py"))
        import pv_plugin
        return pv_plugin.openPMDRectilinearReader
    shutil.copy(reader_py, os.path.join(pkg, "algorithms", "openpmd.py"))
    from paraview.algorithms.openpmd import openPMDReader
    return openPMDReader


def run(Reader, pattern, arrays, thetas=8):
    d = tempfile.mkdtemp()
    pmd = os.path.join(d, "series.pmd")
    with open(pmd, "w") as f:
        f.write(os.path.abspath(pattern) + "\n")
    r = Reader()
    r.SetFileName(pmd)
    r._filename = pmd
    # the reader joins dirname(pmd) + "/" + pattern: give it the absolute pattern relative to "/"
    with open(pmd, "w") as f:
        f.write(os.path.relpath(os.path.abspath(pattern), d) + "\n")
    r.SetNumThetas(thetas)
    r.UpdateInformation()
    sel = r.GetDataArraySelection()
    for i in range(sel.GetNumberOfArrays()):
        name = sel.GetArrayName(i)
        if name in arrays:
            sel.EnableArray(name)
        else:
            sel.DisableArray(name)
    for name in ("GetSpeciesSelection", "GetParticleArraySelection"):
        getattr(r, name)().DisableAllArrays()
    r.Update()
    return r.GetOutputDataObject(0).GetPartition(0)


def vtk_array(ds, name):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(ds.GetPointData().GetArray(name))


def check_theta(Reader, pattern, h5file, name, tol=1e-12):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    nt = 8
    ds = run(Reader, pattern, [name], thetas=nt)
    m = read_mesh(h5file, name)
    r, z = m.axes["r"], m.axes["z"]
    nr, nz = len(r), len(z)
    pts = vtk_to_numpy(ds.GetPoints().GetData()).reshape(nz, nr, nt, 3)
    rad = np.hypot(pts[0, :, 0, 0], pts[0, :, 0, 1])
    err_r = np.max(np.abs(rad - r)) / r.max()
    err_z = np.max(np.abs(pts[:, 0, 0, 2] - z)) / np.max(np.abs(z))
    v = vtk_array(ds, name).reshape(nz, nr, nt)
    thetas = np.linspace(0, 2 * np.pi, nt)
    ref = np.stack([sum(([m.data[0]] + [m.data[2 * k - 1] * np.cos(k * t) + m.data[2 * k] * np.sin(k * t)
                                         for k in range(1, m.modes)]), np.zeros_like(m.data[0])) for t in thetas], -1)
    err_v = np.max(np.abs(v - np.transpose(ref, (1, 0, 2)))) / max(np.max(np.abs(ref)), 1e-300)
    ok = max(err_r, err_z, err_v) < tol
    print(f"  {type(ds).__name__}, {nr} radial nodes (h {np.diff(r).min():.3g}..{np.diff(r).max():.3g}),"
          f" {nz} z nodes; max rel. error: r {err_r:.1e}, z {err_z:.1e}, {name} {err_v:.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_regular(Reader, h5file, name, tol=1e-12):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    ds = run(Reader, h5file, [name])
    m = read_mesh(h5file, name)
    if not ds.IsA("vtkImageData"):
        print(f"  {type(ds).__name__}: expected vtkImageData  FAIL")
        return False
    x0 = np.array(ds.GetOrigin()) + np.array(ds.GetSpacing()) * np.array(ds.GetExtent()[::2])
    err = max(abs(x0[a] - m.axes[lab][0]) for a, lab in enumerate("xyz"))
    v = vtk_array(ds, name).reshape(ds.GetDimensions()[::-1]).transpose()
    err_v = np.max(np.abs(v - m.data)) / np.max(np.abs(m.data))
    ok = max(err / np.max(np.abs(m.axes["z"])), err_v) < tol
    print(f"  {type(ds).__name__} (regular), origin error {err:.1e}, {name} {err_v:.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_cart(Reader, h5file, name, tol=1e-12):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    ds = run(Reader, h5file, [name])
    m = read_mesh(h5file, name)
    vtk_axes = [vtk_to_numpy(c) for c in (ds.GetXCoordinates(), ds.GetYCoordinates(), ds.GetZCoordinates())] \
        if hasattr(ds, "GetXCoordinates") else None
    if vtk_axes is None:
        print(f"  {type(ds).__name__}: not a rectilinear grid  FAIL")
        return False
    # data axis shown along VTK x, y, z: the reader keeps the order for labels "x ... z", else reverses it
    nd = len(m.labels)
    keep = m.labels[0] == "x" and m.labels[-1] == "z"
    src = list(range(nd)) if keep else {3: [2, 1, 0], 2: [1, 2, 0], 1: [2, 1, 0]}[nd]
    err = 0.0
    for a in range(3):
        if src[a] < nd if a < len(src) else False:
            lab = m.labels[src[a]]
            err = max(err, np.max(np.abs(vtk_axes[a] - m.axes[lab])) / np.max(np.abs(m.axes[lab])))
    v = vtk_array(ds, name)
    dims = [len(c) for c in vtk_axes]
    vv = v.reshape(dims[::-1]).transpose()                      # [i_x, i_y, i_z] of the VTK grid
    perm = [src[a] for a in range(3) if a < len(src) and src[a] < nd]
    ref = np.transpose(m.data, perm).reshape([dims[a] for a in range(3) if a < len(src) and src[a] < nd])
    vv = vv.reshape(ref.shape)
    err_v = np.max(np.abs(vv - ref)) / np.max(np.abs(ref))
    ok = max(err, err_v) < tol
    print(f"  {type(ds).__name__}, labels {m.labels}, dims {dims}; max rel. error: coordinates {err:.1e},"
          f" {name} {err_v:.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    Reader = load_reader(sys.argv[1])
    if os.environ.get("BLOCK_H5PY_IN_READER"):  # the reader must then use openPMD-api for the coordinates
        import builtins
        _import = builtins.__import__

        def _guarded(name, globals=None, *args, **kwargs):
            if name == "h5py" and globals and globals.get("__name__") == "paraview.algorithms.openpmd":
                raise ImportError("h5py blocked for the reader (test)")
            return _import(name, globals, *args, **kwargs)
        builtins.__import__ = _guarded
    ok = True
    for case in sys.argv[2:]:
        kind, path = case.split(":", 1)
        print(case)
        try:
            if kind == "quarz":
                ok &= check_theta(Reader, path, path.replace("%06T", "000000"), "psi")
            elif kind == "regular":
                ok &= check_regular(Reader, path, "rho")
            elif kind == "theta":
                ok &= check_theta(Reader, path, path, "rho")
            else:
                ok &= check_cart(Reader, path, "rho")
        except Exception as e:  # e.g. the unmodified reader on a rectilinear thetaMode mesh
            print(f"  {type(e).__name__}: {e}  FAIL")
            ok = False
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
