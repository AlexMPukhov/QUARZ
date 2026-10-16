# Rectilinear meshes in openPMD

Proposal for the openPMD standard 2.0 (pull request against the `upcoming-2.0.0` branch of
github.com/openPMD/openPMD-standard; follows issue #142 "Mesh gridSpacing/Offset: Array"), by the developers
of GEM-PIC-RZ and QUARZ. This file explains the proposal and the forms our two codes write now.

## Summary

A mesh may have explicit, non-equidistant coordinates along some of its axes. It stays a tensor product of 1D
axes, i.e. a *rectilinear* mesh, as in VTK's `vtkRectilinearGrid`.
- **Reference attribute:** the new optional mesh attribute `gridCoordinates` holds one string per axis. An
  empty string means a regular axis. Otherwise the string is the path of a 1D data set with the positions of
  the data points along that axis.
- **Coordinates as data sets, not attributes:** there is no size limit (HDF5 attributes are limited to 64 kB
  without dense storage), the coordinates can be read in chunks, and one coordinate set can be shared by many
  records and iterations. This is the mechanism discussed in #142 by A. Huebl and F. Pöschel.

## Motivation

Many codes use grids that are non-uniform along some axes but rectilinear:
- stretched radial grids in cylindrical and thetaMode codes, fine near the axis and coarse outside;
- longitudinal grids refined at a laser pulse or a witness bunch;
- non-linear binning of diagnostics (PIConGPU's radiation and calorimeter plugins, #142).

openPMD has one `gridSpacing` per axis, so today such data must be interpolated to a uniform grid. That loses
accuracy (staggered and axis values are smeared). It also grows the files 5–50×, because the uniform grid
must resolve the finest cell everywhere. Codes therefore invent private conventions that no tool reads.

## Specification (text of the pull request, in "Mesh based records")

**New optional attribute of a `mesh record`: `gridCoordinates`**
- **Type:** 1D array of N *(string)*, one per axis in the order of `axisLabels`.
- **Empty string:** the axis is regularly spaced (`gridSpacing`, `gridGlobalOffset` and `position` as now).
- **Otherwise:** the absolute path, in the same file, of a 1D data set of *(floatX)* with the coordinates of
  the data points along that axis.
- **Staggering:** the attribute may also be set on a component of a vector record, and then takes precedence.
  Staggered components (e.g. on cell faces and cell centres of a non-uniform grid) can so have different
  coordinates. All components of a record keep the same extent, as now (openPMD-api checks it on reading);
  staggered data with different extents (e.g. N + 1 faces and N centres) go into separate records.

**Coordinate data sets**
- One value per data point: the length equals the extent of the record (component) along the axis.
- Values are in grid units (× the axis's `gridUnitSI` gives SI) and relative to the axis's
  `gridGlobalOffset`. A moving window therefore needs new offsets, not new coordinates.
- Values are strictly increasing.
- Recommended location: per iteration, `<basePath>gridCoordinates/<free name>`. For grids constant in time,
  `/data/gridCoordinates/<free name>`, repeated in every file with `fileBased` encoding.
- A data set may be referenced by any number of records and components.
- A `unitSI` attribute of a coordinate data set, if present (openPMD-api writes `unitSI = 1` for every data
  set), is ignored: the unit is the `gridUnitSI` of the axis.

**Changes to existing attributes**
- `gridSpacing`: informative for axes with explicit coordinates. Writers should store the smallest spacing.
- `position`: `0.0` along axes with explicit coordinates, because the coordinates give the positions of the
  data points.
- `thetaMode`: "regularly-spaced mesh in the r-z plane" becomes "mesh in the r-z plane (regularly spaced, or
  rectilinear …)".

**Why positions, not spacings**
- Positions describe staggered and cell-centred data directly.
- They avoid summation errors.
- They are what visualization tools take (`vtkRectilinearGrid`, `pcolormesh`).

Cell edges (N + 1 values, for finite-volume codes) are left as an open question.

## Example (GEM-PIC-RZ, iteration 4, `h5dump`-style)

```
/data/4/gridCoordinates/r_faces          float64 [101]          # stretched radial grid, faces
/data/4/gridCoordinates/r_centers        float64 [100]          # ... and cell centres
/data/4/gridCoordinates/zeta_slices      float64 [992]          # non-uniform longitudinal slices
/data/4/meshes/native_E_z                float64 [3][100][992]  # modes (m = 0, cos 1, sin 1) x r x z
  geometry            = "thetaMode"            # files written now: "other", see below
  geometryParameters  = "m=2;imag=+"
  axisLabels          = ["r", "z"]
  gridCoordinates     = ["/data/4/gridCoordinates/r_centers", "/data/4/gridCoordinates/zeta_slices"]
  gridSpacing         = [0.3002, 0.25]         # informative: smallest spacings
  gridGlobalOffset    = [0.0, 122.554]         # z offset = s, the window position
  gridUnitSI          = 1.2732e-07             # 1/k0 [m]
  position            = [0.0, 0.0]
  unitSI              = 4.0134e+12
```

## What our codes write now (openPMD 1.1 files)

openPMD 1.1 has no rectilinear meshes. So that readers of regular grids skip these meshes, both codes write
them with `geometry = "other"` and the intended geometry in `geometryParameters` (`"thetaMode;m=2;imag=+"`).
Once 2.0 is out, the geometry becomes `thetaMode` and the files conform.
- **GEM-PIC-RZ** (`output.openpmd_grid = native | both`, records `native_*`) already writes the coordinates
  as data sets referenced by `gridCoordinates`, exactly as proposed.
- **QUARZ** (`output.openpmd_grid = native`) writes through openPMD-api. Built against an openPMD-api with
  coordinate data sets (branch below; detected by CMake), it writes the data-set form: the radial nodes in
  `<basePath>gridCoordinates/r`, referenced by every mesh. With a released openPMD-api it keeps the interim
  form: the nodes in the mesh attribute `gridCoordinates_r`, with `gridType = "rectilinear"`. Both forms
  also carry the older `quarz_r_nodes`. Its ξ axis is regular.
- **Readers:** `tools/openpmd_rectilinear.py` (Python, in both repositories) reads both forms, and so does
  GEM-PIC-RZ's `h5_extract`.

## Implementations

Reference implementations, as patch series in the QUARZ repository
([`docs/openpmd_patches/`](https://github.com/AlexMPukhov/QUARZ/tree/main/docs/openpmd_patches); pull requests to
follow the standard PR):
- **openPMD-api** (C++ and Python, all backends):
  - `Iteration::gridCoordinates`, a container of 1D data sets in `<basePath>gridCoordinates/` (written if not
    empty, read if present; files without it are unchanged);
  - `Mesh::setGridCoordinates()` / `gridCoordinates()`, and `MeshRecordComponent::setGridCoordinates()` for
    staggered components;
  - `Iteration::gridCoordinatesPath(name)` builds the reference, `Iteration::gridCoordinatesOf(mesh[, comp])`
    resolves it (one optional data set per axis);
  - Python: `Iteration.grid_coordinates`, `.grid_coordinates_path()`, `.grid_coordinates_of()`,
    `Mesh.grid_coordinates`;
  - tests for HDF5, JSON, ADIOS2 (file-, group- and variable-based, coordinates that change between steps),
    examples `3c_write_rectilinear_serial` (C++, Python). Released openPMD-api versions read the files and
    ignore the coordinates.
- **openPMD-viewer:** `FieldMetaInformation` takes the coordinates (positions of the data points; `dx` the
  smallest spacing; `rectilinear`, `get_edges()`); both backends; slicing in coordinate space; the thetaMode
  3D reconstruction interpolates between the radial nodes; 2D plots with `pcolormesh`. Also reads the interim
  forms of our codes.
- **openPMD-validator:** checks the references (absolute paths in the file), 1D floating-point data sets, one
  value per data point along the axis, strictly increasing values, and warns if `position` is not 0 along an
  explicit axis.
- **Still needed:** ParaView and VisIt readers (both support rectilinear grids), openPMD-api's `openpmd-pipe`
  copies the coordinate data sets only after a small change (not yet done).
