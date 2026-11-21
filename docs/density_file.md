# Density profile files

A simple, self-describing file format for the density of a plasma or gas target: measured gas-jet profiles
(interferometry, Abel inversion), fluid simulations of nozzles and capillaries, and the hand-off from a
hydrodynamic code to a PIC code. It is meant to be shared by several codes (GEM-PIC-RZ, QUARZ, the 3D codes): a
code reads the geometries it can represent and rejects the others with a clear message.

- **Version:** 1.
- **Geometries:** `z` (profile along the axis), `rz` (axisymmetric), `xz` (Cartesian 2D), `xyz` (3D). z is the
  propagation direction of the driver.
- **The header defines the geometry:** dimension, axes, units, grid, data order, and whether the values are
  densities or a dimensionless factor. A code takes all of this from the file, not from its input deck.
- **Two carriers:** a text table (this document) and an openPMD mesh (HDF5), which holds the same information in
  its attributes (last section).

## Text files

**Lines.** A line starting with `#` belongs to the header if it has the form `# key = value` with one of the keys
below; any other comment line is free text (`##` starts a comment that may contain anything). All other lines are
data: numbers separated by blanks, tabs, commas or semicolons. Blank lines are ignored. An unknown key of the
form `# word = ...` is an error, so that a misspelled key is not silently dropped.

| key | value | meaning |
|-----|-------|---------|
| `density_profile` | `1` | format version (optional; a reader rejects versions it does not know) |
| `geometry` | `z` \| `rz` \| `xz` \| `xyz` | required |
| `columns` | column names with units | columns layout, see below |
| `density_unit` | unit | grid layout: unit of the values |
| `z`, `r`, `x`, `y` | axis | grid layout: one line per axis of the geometry |
| `order` | axis names | grid layout: order of the values, the last axis varies fastest (default: the order of the axis lines) |
| `transverse` | function of r, x, y | geometry `z` only: transverse factor, see below |

**Units.**
- Lengths: `nm`, `um`, `mm`, `cm`, `m` (a code may accept its own normalized units, e.g. `k0^-1`; files meant
  for exchange use physical units). A unit is required.
- Density: `cm^-3` or `m^-3` (a code may accept more, e.g. `n_c`) means the **absolute number density**. No unit,
  `1` or `-` means a dimensionless **factor** on a reference density that the code takes from its own input.
- Values must be finite and not negative.

**Columns layout.** One row per grid point: the coordinates and the density. The line `columns` names the columns
and gives each unit in brackets, `[...]` or `(...)`. The density column is called `n`, `n_e`, `n_gas`, … (any name
starting with `n`) or `density`.

- Geometry `z`: z strictly increasing.
- Geometries `rz`, `xz`, `xyz`: the rows may come in any order; they must fill the grid of the distinct coordinate
  values (every grid point once). The grid is rectilinear; the spacings need not be uniform.

**Grid layout.** The axes stand in the header and the data are the values alone, with any number of values per
line. This is the compact form for large 2D and 3D profiles. One line per axis:

```
# z = uniform <origin> <step> <count> <unit>        equidistant points origin + k step, k = 0 ... count - 1
# r = list <x1> <x2> <x3> ... <unit>                explicit coordinates, strictly increasing
```

The values run through the grid with the last axis of `order` varying fastest (`order = z r`: all r of the first
z, then all r of the second z, …). Their number must equal the product of the axis lengths.

**Transverse function (geometry `z`).** A file with a profile along z may carry the transverse shape as a
function of `r`, `x`, `y` (r² = x² + y²), with numbers with units, the operators `+ - * / ^` and the usual
functions (`exp`, `sqrt`, `min`, `max`, `abs`, …):

```
# transverse = "exp(-(r / (50 um))^2)"
```

The density is then n(z) · T(r, x, y); negative values of T count as zero. Without the line the profile is
uniform across. The file alone so defines the target.

**Units of this code.** QUARZ also accepts lengths in `kp^-1` (c/ω_p) and densities in `n0`; physical units
need `units.n0_cm3`.

**Between and outside the points.** The values are point values at the coordinates. The reference
interpolation is linear along each axis (bilinear, trilinear). Inside the first radius of an `rz` file the
density is the value at that radius. What a code does outside the file (zero, or the edge value continued) is
its own, documented choice; QUARZ and GEM-PIC-RZ take zero unless told otherwise.

## Examples

Geometry `z`, columns layout (a gas jet measured on the laser axis):

```
# gas jet, shot 1234, backing pressure 40 bar
# density_profile = 1
# geometry = z
# columns  = z[mm]  n_e[cm^-3]
0.00   0
0.05   2.0e17
0.10   7.5e17
0.15   1.5e18
0.30   3.0e18
1.70   3.0e18
2.00   0
```

Geometry `z`, grid layout, with a Gaussian transverse profile:

```
# density_profile = 1
# geometry = z
# density_unit = cm^-3
# z = uniform 0 0.5 5 mm
# transverse = "exp(-(r / (60 um))^2)"
0  3.0e18  3.0e18  3.0e18  0
```

Geometry `rz`, columns layout (an Abel-inverted measurement, a fluid simulation):

```
# density_profile = 1
# geometry = rz
# columns  = z[mm]  r[um]  n_e[cm^-3]
0.0    0   1.0e18
0.0   50   1.2e18
0.0  100   0.6e18
0.5    0   3.0e18
0.5   50   3.3e18
0.5  100   1.5e18
1.0    0   1.0e18
1.0   50   1.2e18
1.0  100   0.6e18
```

The same in the grid layout:

```
# density_profile = 1
# geometry = rz
# density_unit = cm^-3
# z = uniform 0 0.5 3 mm
# r = list 0 50 100 um
# order = z r
1.0e18  1.2e18  0.6e18
3.0e18  3.3e18  1.5e18
1.0e18  1.2e18  0.6e18
```

Geometry `xyz`, grid layout (3D codes; a jet that is not symmetric about the laser axis), a factor on the
reference density of the code:

```
# density_profile = 1
# geometry = xyz
# density_unit = 1
# x = uniform -100 100 3 um
# y = uniform -100 100 3 um
# z = list 0 0.3 1.7 2.0 mm
# order = z y x
0 0 0   0 0 0   0 0 0
0.8 1.0 0.8   0.9 1.0 0.9   0.8 1.0 0.8
0.8 1.0 0.8   0.9 1.0 0.9   0.8 1.0 0.8
0 0 0   0 0 0   0 0 0
```

Geometry `xz` is written in the same way with the axes `x` and `z`.

## openPMD files

The same content as a scalar mesh record of an openPMD file (HDF5), default name `density`:

| item | openPMD |
|------|---------|
| geometry | `axisLabels`: `z`; `r`, `z`; `x`, `z`; `x`, `y`, `z` (any order, with `dataOrder`). Of a `thetaMode` record `[mode][r][z]` the mode m = 0 is the profile |
| uniform axis | `gridSpacing`, `gridGlobalOffset`, `position` |
| explicit coordinates | `gridCoordinates`: the rectilinear meshes of [openpmd_rectilinear.md](openpmd_rectilinear.md) |
| length unit | `gridUnitSI` |
| absolute density | `unitDimension` of a number density (L⁻³), `unitSI` to m⁻³ |
| factor | `unitDimension` all zero |
| transverse function | string attribute `transverseFunction` of a mesh with the axis `z` only (not part of the openPMD standard) |

A charge or mass density is not a number density: convert it before writing the file.

## Readers

| code | reads | status |
|------|-------|--------|
| GEM-PIC-RZ | `z`, `rz`; text (both layouts) and openPMD | `plasma.density_file`, `ion.density_file`; `xz` and `xyz` are rejected (no automatic azimuthal average) |
| QUARZ | `z`, `rz`; text (both layouts) and openPMD | `plasma.density_file`, `<species>.density_file` (README §5); lengths also in `kp^-1`, densities also in `n0`; `xz` and `xyz` are rejected |
| GEM-PIC-3D | `xyz`, and `z`, `rz` (rotated about the axis); text and openPMD | `plasma.density_file` |

Portable reader: `src/DensityFile.hpp`, `src/DensityFile.cpp` (QUARZ, GEM-PIC-3D) read all four geometries and
depend on the host code only through two unit conversions; they can be copied into another code unchanged
(openPMD part with `-DDENSITY_FILE_USE_HDF5`). Tests: `tests/test_density_file.cpp`.
