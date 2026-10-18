# Rectilinear meshes: patches for openPMD-api, openPMD-viewer, openPMD-validator and ParaView

Reference implementations of the proposal in [`../openpmd_rectilinear.md`](../openpmd_rectilinear.md)
(openPMD-standard issue #142, pull request #299 against `upcoming-2.0.0`), kept here as `git format-patch`
series and as branches `rectilinear-meshes` of the forks
[AlexMPukhov/openpmd-api](https://github.com/AlexMPukhov/openpmd-api/tree/rectilinear-meshes),
[AlexMPukhov/openpmd-viewer](https://github.com/AlexMPukhov/openpmd-viewer/tree/rectilinear-meshes) and
[AlexMPukhov/openpmd-validator](https://github.com/AlexMPukhov/openpmd-validator/tree/rectilinear-meshes),
until they are submitted as pull requests to the three projects.

| project | base commit | apply |
|---|---|---|
| openPMD-api | `ecaa0943` (dev, 2026-10-15) | `git checkout -b rectilinear-meshes ecaa0943 && git am openPMD-api/*.patch` |
| openPMD-viewer | `a80f6e1` | `git checkout -b rectilinear-meshes a80f6e1 && git am openPMD-viewer/*.patch` |
| openPMD-validator | `3988dbf` (1.1.0.6) | `git checkout -b rectilinear-meshes 3988dbf && git am openPMD-validator/*.patch` |
| ParaView | `1e98e01` (master, 2026-10-18) | `git checkout -b openpmd-rectilinear 1e98e01 && git am ParaView/*.patch` (merge request on gitlab.kitware.com; a standalone plugin for current ParaView versions is in [`tools/paraview/`](../../tools/paraview/)) |

Tested (2026-10-16):
- openPMD-api: the full `SerialIOTests` (64 test cases), `CoreTests`, `JSONTests`, `AuxiliaryTests` with HDF5,
  JSON and ADIOS2 2.10.2; the new test `rectilinear_meshes` (group-, file- and variable-based, staggered
  component, regular mesh, dangling reference); the Python round trip for json, bp, bp4, bp5, h5; the
  examples `3c_write_rectilinear_serial` (C++, Python); `openpmd-pipe` to JSON and to variable-based ADIOS2;
  an MPI build with parallel HDF5 (QUARZ, 2 ranks, bit-identical to serial). Files written by released
  openPMD-api versions read as before; released versions read the new files and ignore the coordinates.
- openPMD-viewer: `tests/test_rectilinear.py` with the h5py backend and with the openpmd-api backend (patched
  and released openPMD-api); native files of GEM-PIC-RZ and QUARZ.
- openPMD-validator: files of the openPMD-api test and example, GEM-PIC-RZ and QUARZ (0 errors), and a
  corrupted file (dangling reference, wrong length, non-monotonic coordinates, position not 0: all reported).
