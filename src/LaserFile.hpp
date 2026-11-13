// Laser envelope from a LASY file (https://github.com/LASY-org/lasy; openPMD LaserEnvelope extension).
//
// A LASY file holds the complex envelope of the transverse electric field (or of the normalized vector potential) at a
// plane z_f versus time, E_{x,y}(r, theta, t) = Re[p_{x,y} env e^{-i omega_L t}], as azimuthal modes (thetaMode:
// [C_0, C_1, S_1, ...] over (t, r)) or on a cartesian grid (t, y, x).
//
// QUARZ needs the envelope a^(r, xi) of the vector potential a = Re[a^ e^{i l theta} e^{-i k0 xi}] in the box at t = 0
// (a snapshot), not a plane history. Every plane wave (omega, kappa) of the file is propagated exactly in vacuum from the
// file plane to the position z = -xi of each slice: at the fixed retarded time it gains e^{i (k_z - omega) dz}, k_z =
// (omega^2 - kappa^2)^{1/2}; A_omega = E_omega / (i omega) per frequency (exact, also for broadband pulses). Placement:
// the file's time t_center sits at xi = xi0, and the file plane lies file_z ahead of the pulse centre at t = 0
// (file_z > 0: the pulse still has to travel to the plane, e.g. a file written at the focus):
//   dz(xi) = -(file_z + xi - xi0),   u_file(xi) = xi - xi0 + t_center.
// The radial transform is a Hankel transform of order |l| (the quadrature of GEM-PIC-RZ src/laser_file.cpp, exact for an
// interpolant regular at the axis); it is evaluated directly at the radial nodes of the (non-uniform) grid.
#pragma once

#include <complex>
#include <string>
#include <vector>

namespace quarz {

struct LaserFileInput {
    std::string path;
    long iteration = -1;         // < 0: the last iteration in the file
    double n0_cm3 = 0;           // the plasma density unit (omega_p, k_p)
    double xi0 = 0;              // xi of the file's t_center (1/k_p)
    double file_z = 0;           // distance from the pulse centre at t = 0 to the file plane (1/k_p)
    double t_center = 1e300;     // file time put at xi0 (1/omega_p, the file's time base); 1e300: the intensity centroid
    double a0 = 0;               // > 0: rescale so that the peak |a^| at the file plane is a0
    int l = 0;                   // azimuthal index of the envelope (laser.l): the file's mode e^{i l theta}
    double k0 = 0;               // carrier omega0/omega_p of the envelope; 0: the file's omega_L
};

struct LaserFileResult {
    std::vector<std::complex<double>> a;   // a^[k * nr + j] at the slices xi[k] and radii r[j]
    double k0 = 0;                         // the carrier used (omega0/omega_p)
    double omegaL = 0;                     // the file's omega_L / omega_p
    bool circular = false;                 // polarization (p_x, p_y) circular, else linear
    double peak_file = 0;                  // peak |a^| at the file plane (after a0)
    double mode_loss = 0;                  // share of the file's energy in other azimuthal modes (dropped)
    std::vector<std::string> info;
};

// The envelope at t = 0 on the slices xi[] (1/k_p) and the radial nodes r[] (1/k_p). Needs a build with HDF5.
LaserFileResult laser_from_file(const LaserFileInput& in, const std::vector<double>& r, const std::vector<double>& xi);

// True if this build can read LASY files (HDF5).
bool laser_file_available();

} // namespace quarz
