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
  coordinates and extents.

**Coordinate data sets**
- One value per data point: the length equals the extent of the record (component) along the axis.
- Values are in grid units (× the axis's `gridUnitSI` gives SI) and relative to the axis's
  `gridGlobalOffset`. A moving window therefore needs new offsets, not new coordinates.
- Values are strictly increasing.
- Recommended location: per iteration, `<basePath>gridCoordinates/<free name>`. For grids constant in time,
  `/data/gridCoordinates/<free name>`, repeated in every file with `fileBased` encoding.
- A data set may be referenced by any number of records and components.

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
- **QUARZ** (`output.openpmd_grid = native`) writes through openPMD-api, which cannot yet write data sets
  outside records. It keeps the radial nodes in the mesh attribute `gridCoordinates_r` (and the older
  `quarz_r_nodes`), with `gridType = "rectilinear"`. Its ξ axis is regular. It will switch to the data-set
  form when openPMD-api supports it.
- **Readers:** `tools/openpmd_rectilinear.py` (Python, in both repositories) reads both forms, and so does
  GEM-PIC-RZ's `h5_extract`.

## Needed in the ecosystem

- **openPMD-api:** read and write `gridCoordinates` and coordinate data sets (e.g.
  `Mesh::setGridCoordinates(axis, std::vector<double>)`, which writes the data set and the reference).
- **openPMD-viewer:**
  - build `FieldMetaInformation` from the coordinates;
  - plot with `pcolormesh` on the nodes.
- **openPMD-validator:**
  - check the paths, lengths and monotonicity;
  - check `position = 0` along explicit axes.
- **ParaView, VisIt:** map to rectilinear grids, which they already support.
