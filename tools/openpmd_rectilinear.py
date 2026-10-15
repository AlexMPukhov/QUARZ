"""Read openPMD meshes on regular or rectilinear (non-uniform) grids, e.g. the native records of
GEM-PIC-RZ (native_E_r, ...) and of QUARZ (output.openpmd_grid = native).

A rectilinear mesh is a regular openPMD mesh with, for some axes, explicit node positions: the mesh
attribute gridCoordinates names one 1D data set per axis ("" = regular axis), as proposed for openPMD
2.0 (docs/openpmd_rectilinear.md); the interim form of openPMD 1.1 files (QUARZ) keeps them in the
attribute gridCoordinates_<axis label>. Regular axes: gridGlobalOffset + (i + position) * gridSpacing. Positions are in units of gridUnitSI, relative to
gridGlobalOffset. Works with geometry "thetaMode" and with "other" + geometryParameters
"thetaMode;...", the interim marking of rectilinear thetaMode meshes.

    from openpmd_rectilinear import read_mesh, theta_slice
    m = read_mesh("diag/openpmd/openpmd_100.h5", "native_E_z")     # SI units
    m.axes["r"], m.axes["z"]       # node positions [m] of the component (absolute)
    F = theta_slice(m, theta=0.0)   # F(r, z) in the plane theta (thetaMode)

Needs h5py and numpy (HDF5 files; for ADIOS2 files use openpmd-api to get the same attributes).
"""
import h5py
import numpy as np


class Mesh:
    def __init__(self, data, axes, labels, geometry, modes):
        self.data = data          # [modes][axis 0][axis 1] for thetaMode, else the dataset shape, SI
        self.axes = axes          # {label: positions [m]}
        self.labels = labels      # axis labels in data order
        self.geometry = geometry  # "thetaMode", "cartesian", ...
        self.modes = modes        # thetaMode: number of azimuthal modes (m = 0 .. modes - 1)


def _s(v):
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def _attr(obj, parent, name, default=None):
    """attribute of the record component, else of the record (mesh)"""
    if name in obj.attrs:
        return obj.attrs[name]
    if parent is not None and name in parent.attrs:
        return parent.attrs[name]
    return default


def read_mesh(path, name, component=None, iteration=None):
    f = h5py.File(path, "r")
    base = _s(f.attrs["basePath"])
    meshes = _s(f.attrs["meshesPath"])
    its = sorted(f["data"].keys(), key=int)
    it = str(iteration) if iteration is not None else its[0]
    rec = f[base.replace("%T", it) + meshes + name]
    if isinstance(rec, h5py.Group):  # vector record: component dataset (or a constant)
        if component is None:
            raise ValueError(f"{name} has components {list(rec.keys())}: give component=")
        obj, parent = rec[component], rec
    else:
        obj, parent = rec, None
    data = obj[...] * float(_attr(obj, parent, "unitSI", 1.0))
    labels = [_s(x) for x in _attr(obj, parent, "axisLabels")]
    order = _s(_attr(obj, parent, "dataOrder", "C"))
    if order == "F":
        labels = labels[::-1]
    unit = float(_attr(obj, parent, "gridUnitSI", 1.0))
    off = np.asarray(_attr(obj, parent, "gridGlobalOffset"), float)
    dx = np.asarray(_attr(obj, parent, "gridSpacing"), float)
    pos = np.asarray(_attr(obj, parent, "position", np.zeros(len(labels))), float)
    geometry = _s(_attr(obj, parent, "geometry", "cartesian"))
    params = _s(_attr(obj, parent, "geometryParameters", ""))
    if geometry == "other" and params.startswith("thetaMode"):
        geometry = "thetaMode"
    lead = 1 if geometry == "thetaMode" else 0  # the mode index precedes the spatial axes
    axes = {}
    for d, lab in enumerate(labels):
        n = data.shape[lead + d]
        coords = None
        paths = _attr(obj, parent, "gridCoordinates")  # proposal for openPMD 2.0: data set per axis
        if paths is not None and _s(paths[d]) != "":
            coords = f[_s(paths[d])][...]
        if coords is None:  # interim form (QUARZ, openPMD 1.1 files): attribute per axis
            coords = _attr(obj, parent, "gridCoordinates_" + lab)
        if coords is not None:  # rectilinear axis: explicit node positions
            x = np.asarray(coords, float)
            if len(x) != n:
                raise ValueError(f"{name}: gridCoordinates_{lab} has {len(x)} entries, the data {n}")
        else:  # regular axis
            x = (np.arange(n) + pos[d]) * dx[d]
        axes[lab] = (off[d] + x) * unit
    modes = (data.shape[0] + 1) // 2 if geometry == "thetaMode" else 0
    return Mesh(data, axes, labels, geometry, modes)


def theta_slice(m, theta):
    """thetaMode: F = F_0 + sum_m (C_m cos(m theta) + S_m sin(m theta)), dataset index 0, 2m - 1, 2m"""
    if m.geometry != "thetaMode":
        raise ValueError("not a thetaMode mesh")
    F = m.data[0].copy()
    for k in range(1, m.modes):
        F += m.data[2 * k - 1] * np.cos(k * theta) + m.data[2 * k] * np.sin(k * theta)
    return F
