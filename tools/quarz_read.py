"""Readers for QUARZ output files (optional helper; formats are documented in README.md).

fields_NNNNNN.bin (version 3):
    int32 version(=3), int32 M (radial nodes), int32 K (xi slices), int32 mode1, float64 t,
    float64 r[M], float64 xi[K], int32 ncomp, then ncomp x { char name[16], float64 data[K][M] }
    (xi = z - t, increasing: the head of the box is the last slice)
    Version 2 (QUARZ v1: xi = t - z, the head of the box first) is converted on reading: the slices are
    reversed and xi negated, so that both versions give the same arrays.
    mode 0 names: psi ez er eth br bth bz ne ni rhob
    mode 1 names: <name>_c, <name>_s with  f(r, theta) = f0 + f_c cos(theta) + f_s sin(theta)
beam_<name>_NNNNNN.bin:
    header: char magic[8] = "QUARZDMP", int32 version (=1), int32 columns (7, or 10 with spin), int64 n;
    then n records of float64: x, y, p_x, p_y, p_z, xi, w (with spin: + s_x, s_y, s_z)
    A dump without the header is one of QUARZ v1.0.0 (int32 n, then the records; xi = t - z): read_beam converts it
    (xi negated) with a warning; tools/convert_deck.py --beam-file writes it as a v2 dump.
"""
import warnings

import numpy as np


def read_fields(fn):
    with open(fn, "rb") as f:
        ver, M, K, m1 = np.fromfile(f, dtype=np.int32, count=4)
        if ver not in (2, 3):
            raise ValueError("unknown QUARZ field file version %d" % ver)
        if ver == 2:
            warnings.warn(f"{fn}: QUARZ v1 field file (xi = t - z), converted to xi = z - t", stacklevel=2)
        t = np.fromfile(f, dtype=np.float64, count=1)[0]
        r = np.fromfile(f, dtype=np.float64, count=M)
        xi = np.fromfile(f, dtype=np.float64, count=K)
        nc = np.fromfile(f, dtype=np.int32, count=1)[0]
        out = {"t": t, "r": r, "xi": xi, "mode1": bool(m1)}
        for _ in range(nc):
            name = f.read(16).split(b"\0")[0].decode()
            out[name] = np.fromfile(f, dtype=np.float64, count=M * K).reshape(K, M)
    if ver == 2:   # v1: xi = t - z, head first -> xi = z - t, head last
        out["xi"] = -xi[::-1]
        for k, v in out.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                out[k] = v[::-1]
    return out


def xz_plane(d, name):
    """Field in the (xi, x) plane through the axis (theta = 0 for x > 0, theta = pi for x < 0).
    Returns x (2M-1 points, -R..R) and array [K, 2M-1]."""
    r = d["r"]
    f0 = d[name]
    fc = d.get(name + "_c", np.zeros_like(f0))
    right = f0 + fc
    left = f0 - fc
    x = np.concatenate([-r[:0:-1], r])
    return x, np.concatenate([left[:, :0:-1], right], axis=1)


DUMP_MAGIC = b"QUARZDMP"
DUMP_VERSION = 1


def read_dump(fn):
    """native beam dump -> (array [n, 7 or 10] in the file's columns, v1). v1 = True: a dump of QUARZ v1.0.0 without
    header, whose xi column is still t - z (not converted here)."""
    import os
    size = os.path.getsize(fn)
    with open(fn, "rb") as f:
        head = f.read(24)
        if head[:8] == DUMP_MAGIC:
            ver, nc = np.frombuffer(head[8:16], np.int32)
            n = int(np.frombuffer(head[16:24], np.int64)[0])
            if ver != DUMP_VERSION or nc not in (7, 10) or size != 24 + 8 * nc * n:
                raise ValueError(f"{fn}: not a valid QUARZ beam dump (version {ver}, {nc} columns, n = {n})")
            f.seek(24)
            return np.fromfile(f, dtype=np.float64, count=nc * n).reshape(n, nc), False
        n = int(np.frombuffer(head[:4], np.int32)[0]) if len(head) >= 4 else -1
        nc = 7 if n == 0 else ((size - 4) // (8 * n) if n > 0 else 0)
        if n < 0 or nc not in (7, 10) or size != 4 + 8 * nc * n:
            raise ValueError(f"{fn}: not a QUARZ beam dump")
        f.seek(4)
        return np.fromfile(f, dtype=np.float64, count=nc * n).reshape(n, nc), True


def write_dump(fn, a):
    """array [n, 7 or 10] (xi = z - t) -> native beam dump with header"""
    a = np.ascontiguousarray(a, dtype=np.float64)
    with open(fn, "wb") as f:
        f.write(DUMP_MAGIC + np.array([DUMP_VERSION, a.shape[1]], np.int32).tobytes() + np.int64(a.shape[0]).tobytes())
        f.write(a.tobytes())


def read_beam(fn):
    """beam_<name>_<step>.bin -> dict x y px py pz xi w (+ sx sy sz for a beam with spin)"""
    a, v1 = read_dump(fn)
    if v1:
        warnings.warn(f"{fn}: QUARZ v1 beam dump (no header, xi = t - z), converted to xi = z - t", stacklevel=2)
        a = a.copy()
        a[:, 5] = 0.0 - a[:, 5]
    nc = a.shape[1]
    names = ["x", "y", "px", "py", "pz", "xi", "w"] + (["sx", "sy", "sz"] if nc == 10 else [])
    return dict(zip(names, a.T))


def read_spinlog(fn):
    """spin.txt -> dict beam_name -> columns step t sx sy sz P"""
    data = {}
    for line in open(fn):
        if line.startswith("#"):
            continue
        p = line.split()
        data.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return {k: dict(zip(["step", "t", "sx", "sy", "sz", "P"], np.array(v).T)) for k, v in data.items()}


def read_beamlog(fn):
    """beams.txt -> dict beam_name -> columns"""
    cols = ["step", "t", "alive", "npart", "charge", "gamma_mean", "gamma_rms", "xi_mean", "xi_rms",
            "r_rms", "emit_nx", "x_mean", "y_mean", "emit_ny", "sigma_x", "sigma_y"]
    data = {}
    for line in open(fn):
        if line.startswith("#"):
            continue
        p = line.split()
        data.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return {k: dict(zip(cols, np.array(v).T)) for k, v in data.items()}


def read_slices(fn):
    a = np.loadtxt(fn)
    return dict(zip(["xi", "charge", "x_mean", "y_mean", "sigma_x", "sigma_y", "gamma_mean", "emit_nx"], a.T))


def head_first(d):
    """Fields from read_fields() reordered from the head of the box backwards, with the coordinate
    s = -xi (distance behind the head; increasing). Convenient for wake analysis, where the plasma
    response is integrated from the head. The arrays [K][M] are reversed along K, "xi" too."""
    K = len(d["xi"])
    out = {k: (v[::-1] if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == K else v) for k, v in d.items()}
    out["xi"] = d["xi"][::-1]
    out["s"] = -out["xi"]
    return out
