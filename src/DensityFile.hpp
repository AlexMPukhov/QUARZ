// ============================================================================
//  DensityFile.hpp -- reader of density profile files (plasma / gas targets)
//
//  Format: docs/density_file.md (version 1), one self-describing format shared
//  by several codes. Geometries z | rz | xz | xyz; a text table (columns layout
//  or grid layout) or an openPMD mesh (HDF5).
//
//  This reader is independent of the code it is built into: it only needs the
//  two unit conversions of DensityFileUnits. A code accepts the geometries it
//  can represent and rejects the others (DensityFile::geometry). The same two
//  files (DensityFile.hpp / .cpp) can be copied into another code unchanged;
//  the openPMD part is compiled with -DDENSITY_FILE_USE_HDF5 (HDF5 >= 1.10).
// ============================================================================
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace densfile {

// Conversions of a number with the unit named in the file to the units of the code. They throw
// std::invalid_argument (with a readable message) for an unknown unit.
struct DensityFileUnits {
    std::function<double(double, const std::string&)> length;   // e.g. (0.05, "mm") -> code length
    std::function<double(double, const std::string&)> density;  // e.g. (3e18, "cm^-3") -> code density
};

struct DensityFile {
    std::string path;
    std::string format;     // "text" or "openPMD"
    std::string detail;     // geometry and layout (text), mesh and iteration (openPMD)
    std::string geometry;   // "z", "rz", "xz", "xyz"
    std::string nUnit;      // density unit as stated in the file
    bool absolute = false;  // values are densities (code units); else a dimensionless factor
    // strictly increasing coordinates (code units). z always; a: r (rz) or x (xz, xyz); b: y (xyz)
    std::vector<double> z, a, b;
    std::vector<double> v;  // values [(iz * na + ia) * nb + ib], na = max(1, a.size()), nb = max(1, b.size())
    double peak = 0.0, zPeak = 0.0, aPeak = 0.0, bPeak = 0.0;
    std::string transverse;  // geometry z: text of the transverse function of r, x, y (empty: none)

    size_t na() const { return a.empty() ? 1 : a.size(); }
    size_t nb() const { return b.empty() ? 1 : b.size(); }
    // Linear interpolation at the file's coordinates, without the transverse function. Outside the file 0, or
    // the edge value (hold). z: (zz); rz: (zz, r), inside the first radius its value; xz: (zz, x); xyz: (zz, x, y)
    double at(double zz, double aa = 0.0, double bb = 0.0, bool hold = false) const;
};

// Reads a text table or an openPMD file (recognized by the HDF5 signature). `mesh` and `iteration` (< 0: the
// last) select the openPMD record. Throws std::runtime_error with the file name in the message.
DensityFile readDensityFile(const std::string& path, const std::string& mesh, long iteration,
                            const DensityFileUnits& units);

// True if this build reads openPMD (HDF5) density files.
bool densityFileHasHDF5();

}  // namespace densfile
