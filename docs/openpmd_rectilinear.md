# Proposal: rectilinear meshes in openPMD

*Draft for the openPMD-standard repository (github.com/openPMD/openPMD-standard), proposed by the developers of
GEM-PIC-RZ and QUARZ. Both codes already write meshes in this form; see "Reference implementations".*

## Summary

Allow mesh records whose nodes are **not equally spaced along some axes**, while the mesh stays a tensor
product of 1D axes (a *rectilinear* grid). For each such axis, the node positions are stored in an attribute
`gridCoordinates_<axis label>`. Axes without this attribute stay regular, so a regular mesh is a special case
and existing files remain valid.

## Motivation

Many codes use grids that are non-uniform along one or more axes but still rectilinear. Examples:
- stretched radial grids in cylindrical and quasi-cylindrical (thetaMode) codes, fine near the axis and coarse
  far out;
- longitudinal grids that are fine at a laser pulse or a witness bunch and coarse elsewhere;
- moving-window codes whose window grid is given by reference points with interpolated steps.

openPMD 1.1 describes a mesh by one `gridSpacing` per axis, so such data must be interpolated to a uniform
grid before writing. That costs:
- **Accuracy:** the stored data are no longer the simulation's. Staggered and axis values are smeared, and
  diagnostics or restarts built on the files differ from the code.
- **Size:** the uniform grid must resolve the finest cell everywhere. In our cases the files grow 5–50× (a
  stretched radial grid with 40 nm near the axis and 3 µm at the edge).
- **Clarity:** each code invents a private convention, e.g. geometry `other` plus a code-specific node
  attribute, which no tool can read.

## Specification (additions to "Mesh based records")

### New mesh attribute

- `gridType`
  - type: *(string)*
  - optional; default `regular`
  - values:
    - `regular`: every axis is equally spaced, as in openPMD 1.1;
    - `rectilinear`: axes may have explicit node positions, see `gridCoordinates_<a>`.

### New attributes for `gridType = rectilinear`

- `gridCoordinates_<a>`, where `<a>` is an entry of `axisLabels`
  - type: *(1D array of floatX)*
  - optional per axis; on the mesh record, or on a record component (see "Staggering")
  - description: positions of the data points along axis `<a>`, in units of `gridUnitSI`, relative to the
    corresponding entry of `gridGlobalOffset`. The array length equals the extent of the dataset along `<a>`,
    and the values are strictly increasing.
  - The position of data point `i` along `<a>` is `gridGlobalOffset[a] + gridCoordinates_<a>[i]`, times
    `gridUnitSI` in SI.
  - Axes without `gridCoordinates_<a>` are regular and follow the existing rules (`gridSpacing`,
    `gridGlobalOffset`, `position`).

### Changes to existing attributes

- `gridSpacing` remains required. For a rectilinear axis its value is informative; writers should store the
  smallest spacing along that axis, so that tools can estimate resolution and memory.
- `position` of a record component: on rectilinear axes the entry must be `0`. The node positions already
  include any staggering.
- `geometry` / `geometryParameters`: unchanged. The extension applies to every geometry, including `thetaMode`,
  where the mode index stays the first dataset index and is not an axis.

### Staggering

If the components of a record live on different nodes, e.g. E_r on cell faces and E_z on cell centres:
- each record component may carry its own `gridCoordinates_<a>`, which takes precedence over the mesh-level
  attribute, as `position` does today;
- components may then have different extents along `<a>`.

Alternatively, writers may store such components as separate scalar meshes.

### Why one attribute per axis

The axes have different numbers of nodes, so their positions cannot be one rectangular array attribute.
Putting the axis label in the attribute name keeps every attribute a plain 1D array, which all backends
(HDF5, ADIOS2, JSON) and openPMD-api store directly.

**Note on HDF5.** Attributes larger than 64 kB (more than 8192 float64 nodes) need dense attribute storage,
which the HDF5 ≥ 1.8 file format supports. Writers set `H5Pset_libver_bounds(fapl, H5F_LIBVER_V18,
H5F_LIBVER_LATEST)`. ADIOS2 has no such limit.

## Compatibility and transition

A reader that does not know `gridType` would place rectilinear data at `gridGlobalOffset + i·gridSpacing`,
silently wrong. We therefore suggest:
1. **In the standard:** `gridType` enters with the next minor or major version. Readers of that version must
   check it and reject `rectilinear` meshes they cannot handle.
2. **Until then (interim convention, used by GEM-PIC-RZ and QUARZ):**
   - writers mark rectilinear meshes with `geometry = "other"` and put the intended geometry first in
     `geometryParameters`, e.g. `"thetaMode;m=2;imag=+"`;
   - existing readers skip or reject geometry `other`, so they cannot misplace the data;
   - the same files become standard-conforming by changing `geometry` to `thetaMode` once the extension is
     accepted.
3. **Writers that need every tool to work today** may store both a regular (interpolated) mesh and the
   rectilinear one in the same iteration, as GEM-PIC-RZ does by default (`E`, `B`, `n_e` regular;
   `native_E_r`, … rectilinear).

## Example (GEM-PIC-RZ, `h5dump`-style, iteration 4)

```
/data/4/meshes/native_E_z            float64 [3][100][992]      # modes (m = 0, cos 1, sin 1) x r x z
  geometry            = "other"                                 # interim; "thetaMode" once adopted
  geometryParameters  = "thetaMode;m=2;imag=+"
  gridType            = "rectilinear"
  axisLabels          = ["r", "z"]
  dataOrder           = "C"
  gridUnitSI          = 1.2732e-07                              # 1/k0 in m
  gridGlobalOffset    = [0.0, 122.554]                          # z offset: s (moving window)
  gridSpacing         = [0.3002, 0.25]                          # informative: smallest spacings
  gridCoordinates_r   = [0.150, 0.450, 0.750, ...]   (100)      # cell centres of a stretched grid
  gridCoordinates_z   = [0.0, 0.499, 0.998, ..., 300.16] (992)  # non-uniform slices (0.25 ... 0.5)
  position            = [0.0, 0.0]
  unitSI              = 4.0134e+12
  unitDimension       = [1, 1, -3, -1, 0, 0, 0]
  timeOffset          = 0.0
```

## Reference implementations

- **GEM-PIC-RZ** (github.com/AlexMPukhov/GEM-PIC-RZ):
  - `src/output_hdf5.cpp` writes the records `native_*` (`output.openpmd_grid = native | both`);
  - `tools/h5_extract.cpp` (C++) and `tools/openpmd_rectilinear.py` (Python, h5py) read them;
  - `tests/openpmd.sh` checks that the native records reproduce the code's data exactly and pass
    `openPMD_check_h5`.
- **QUARZ** (github.com/AlexMPukhov/QUARZ):
  - `src/OpenPMDWriter.cpp` (openPMD-api) writes `gridType` and `gridCoordinates_r` for its non-uniform
    radial grid with `output.openpmd_grid = native`; its ξ axis is regular.

## Changes needed in the ecosystem

- **openPMD-api:** `Mesh::gridType()` and `gridCoordinates(axis)` accessors; the writers need nothing new
  (attributes are written today).
- **openPMD-viewer:**
  - build the axes of `FieldMetaInformation` from `gridCoordinates_<a>` when present;
  - plot with `pcolormesh` on the node coordinates instead of `imshow` with an extent.
- **openPMD-validator:**
  - accept `gridType`;
  - check the length and monotonicity of `gridCoordinates_<a>`;
  - require `position = 0` on rectilinear axes.
- **ParaView and VisIt readers:** map to rectilinear grids (`vtkRectilinearGrid`), which they support natively.

## Open questions for the community

1. Attribute per axis (`gridCoordinates_<a>`) or a dataset per axis in a reserved group (e.g.
   `<mesh>/coordinates/<a>`)? Attributes are simpler for writers. Datasets avoid the HDF5 size limit and allow
   chunked reading.
2. Should cell edges (N + 1 values) be allowed in addition to data-point positions, for finite-volume codes?
3. Is `gridType = "rectilinear"` the right name? VTK and NetCDF use "rectilinear" for this tensor-product case.
