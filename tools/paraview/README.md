# ParaView: openPMD rectilinear meshes

ParaView's openPMD reader (`paraview/algorithms/openpmd.py`) reads only regular meshes. The files here
teach it the rectilinear (non-uniform) meshes of the openPMD 2.0 proposal
([`docs/openpmd_rectilinear.md`](../../docs/openpmd_rectilinear.md)), as written by QUARZ
(`output.openpmd_grid = native`) and GEM-PIC-RZ.

- **`openPMDRectilinearReader.py`** — a plugin for existing ParaView versions (5.10 or newer; ParaView's
  Python needs `numpy` and `openpmd_api`). In ParaView: *Tools → Manage Plugins → Load New…*, choose this
  file (tick *Auto Load* to keep it). Then open a `.pmd` file — one text line with the file name pattern of
  the series, e.g. `data_%06T.h5` next to the data files — and pick **openPMD Reader (rectilinear
  meshes)**.
  - **Two outputs**: *Output-0* = the meshes (fields), *Output-1* = the particles (empty unless
    *Species* are ticked). ParaView may show a 3D structured grid only as its **outline** (for thetaMode:
    two circles and a rectangle). To see a field: click *Output-0* in the Pipeline Browser, set the
    representation to *Surface* and the colouring from *Solid Color* to a field (e.g. `E`, component Z,
    or `n_e`); for the r–z map apply *Filters → Slice* with normal Y (plane θ = 0, π). The plugin asks
    for *Surface* by default (a representation hint; not yet confirmed in every ParaView version).
  - Cartesian meshes with explicit coordinates become `vtkRectilinearGrid`s.
  - thetaMode meshes (also the interim marking `geometry = other`, `geometryParameters = thetaMode;…`)
    become a 3D `vtkStructuredGrid` on the true radial and longitudinal nodes; the property *Thetas* sets
    the number of azimuthal samples (slice it with a plane for the r–z map).
  - The coordinates of the openPMD 2.0 form are data sets. Released openPMD-api versions cannot read them
    yet; the plugin then reads them from HDF5 files with `h5py` (if ParaView's Python has it). ADIOS2 files
    with explicit coordinates need an openPMD-api with coordinate data sets
    ([our patches](../../docs/openpmd_patches/)). The interim form of QUARZ (attribute `gridCoordinates_r`)
    works with any openPMD-api.
  - Regular meshes are read exactly as by the built-in reader.
- **The same change for ParaView itself**: [`docs/openpmd_patches/ParaView/`](../../docs/openpmd_patches/ParaView/)
  (`git am` on ParaView master `1e98e01`, 2026-10-18), to be proposed as a merge request on
  gitlab.kitware.com. The plugin is generated from the patched module (its reader class plus ParaView's
  plugin decorators).
- **Tests without ParaView** (the reader is a `VTKPythonAlgorithmBase` and runs with the `vtk` Python
  package): `make_rectilinear_testfiles.py OUTDIR` writes small cartesian (3D in both axis orders, 2D) and
  thetaMode files; `test_openpmd_reader.py READER.py CASE…` checks point coordinates and values against an
  independent h5py reader (`tools/openpmd_rectilinear.py`). Run in validation section 10 when `vtk` is
  installed. Results (2026-10-18, VTK 9.7.1): coordinates and values to round-off for QUARZ native output
  (m = 0 and m = 1), the synthetic files, with released openPMD-api + h5py and with openPMD-api with
  coordinate data sets (h5py blocked); regular files unchanged; piece requests (parallel reading) give
  consistent coordinate slices. Not yet tried in the ParaView GUI.
