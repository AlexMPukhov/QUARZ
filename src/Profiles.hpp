// Plasma density profiles n(r) * f(z)  (host-side, evaluated at loading).
#pragma once

#include "DensityFile.hpp"
#include "Parser.hpp"
#include "Types.hpp"
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace quarz {

class Config;

// piecewise-linear table; constant extrapolation outside
class Table1D {
public:
    Table1D() = default;
    explicit Table1D(std::vector<std::pair<double, double>> pts);
    double operator()(double x) const;
    bool empty() const { return pts_.empty(); }
private:
    std::vector<std::pair<double, double>> pts_;
};

class DensityProfile {
public:
    DensityProfile() = default;
    // reads keys  <prefix>.radial, <prefix>.rmax, <prefix>.channel_*, <prefix>.radial_table, <prefix>.z_table,
    // <prefix>.density_file (with file_z, file_outside, file_mesh, file_iteration)
    DensityProfile(const Config& cfg, const std::string& prefix);

    // parsed profile  <prefix>.density(x,y,z) = "..."  (z = lab position); replaces the built-ins
    // profile from a file  <prefix>.density_file = jet.txt  (docs/density_file.md; geometry z or rz, text or
    // openPMD): n = file(r, z - file_z) x its transverse function x density(x,y,z) if given; 0 beyond rmax.
    // Both kinds are evaluated through eval() / eval_avg() ("parsed").
    bool parsed() const { return parsed_ || file_; }
    bool from_file() const { return static_cast<bool>(file_); }
    double eval(double x, double y, double z) const;
    // lines for the run summary: ranges of the file against the run z0 ... z1, peak; warnings ("Warning: ...")
    // for a file cut at a finite density and for gradients shorter than 4 time steps dt (0: not checked)
    std::vector<std::string> file_report(double z0, double z1, double dt) const;
    // azimuthal average at radius r (m = 0 part, used by axisymmetric runs)
    double eval_avg(double r, double z) const;

    double radial(double r) const;    // n(r) / n0 at the entrance
    double longitudinal(double z) const;  // f(z)
    double operator()(double r, double z) const { return radial(r) * longitudinal(z); }

private:
    std::string type_ = "uniform";
    double rmax_ = 1e300;       // plasma column radius
    double depth_ = 0;          // channel: n = 1 + depth * (r/rc)^2
    double rc_ = 1;
    Table1D rtab_, ztab_;
    bool parsed_ = false;      // density(x,y,z) given
    Parser parser_;
    std::string prefix_;
    std::shared_ptr<const densfile::DensityFile> file_;
    double file_z_ = 0;         // z of the simulation at the file's z = 0
    bool file_hold_ = false;    // file_outside = hold: edge values beyond the file (else 0)
    double len_m_ = 0;          // c/omega_p [m] (0: units.n0_cm3 not given)
    bool has_trans_ = false;    // transverse function of the file T(r, x, y)
    Parser trans_;
};

} // namespace quarz
