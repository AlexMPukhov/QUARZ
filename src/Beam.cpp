#include "Beam.hpp"
#include <functional>
#include "Ionization.hpp"
#include "Config.hpp"
#include "Pushers.hpp"
#include "SliceData.hpp"
#include "BeamKernels.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace quarz {

using namespace detail;

Beam::Beam(const Config& cfg, const std::string& name, const RadialGrid& grid, const BeamGrid& bg, bool mode1,
           const double* qm)
    : name_(name), grid_(grid), bg_(bg), m1_(mode1) {
    q_ = qm ? qm[0] : cfg.get_double(name + ".charge", -1.0);
    m_ = qm ? qm[1] : cfg.get_double(name + ".mass", 1.0);
    rigid_ = qm ? false : cfg.get_bool(name + ".rigid", false);
    adaptive_dt_ = cfg.get_bool(name + ".adaptive_dt", true);
    passive_ = cfg.get_bool(name + ".passive", false);
    const std::string pm = cfg.get_string(name + ".pusher", cfg.get_string("pusher.beam", "vay"));
    if (pm == "vay") pusher_ = BeamPusher::Vay;
    else if (pm == "boris") pusher_ = BeamPusher::Boris;
    else if (pm == "hc") pusher_ = BeamPusher::HC;
    else if (pm == "imp") pusher_ = BeamPusher::IMP;
    else if (pm == "imp_rr") pusher_ = BeamPusher::IMP_RR;
    else throw std::runtime_error("pusher.beam must be vay|boris|hc|imp|imp_rr");
    const double n0cm3 = cfg.get_double("pusher.rr_n0_cm3", 0.0);
    if (n0cm3 > 0) {
        const double re = 2.8179403262e-13;                              // cm
        const double kp = 5.64146e4 * std::sqrt(n0cm3) / 2.99792458e10;  // 1/cm
        rr_ = (2.0 / 3.0) * re * kp * q_ * q_ / m_;
        if (pusher_ != BeamPusher::IMP_RR)
            throw std::runtime_error("radiation reaction (pusher.rr_n0_cm3) requires pusher.beam = imp_rr");
    }
    rr_ *= cfg.get_double("pusher.rr_scale", 1.0);
    // ---- spin (Thomas-BMT), <beam>.spin = sx sy sz (initial direction)
    {
        const auto sv = cfg.get_list(name + ".spin");
        if (!sv.empty()) {
            if (sv.size() != 3) throw std::runtime_error(name + ".spin: give three numbers (initial spin direction)");
            double n2 = 0;
            for (int c = 0; c < 3; ++c) { s0_[c] = std::stod(sv[c]); n2 += s0_[c] * s0_[c]; }
            if (!(n2 > 0)) throw std::runtime_error(name + ".spin: zero vector");
            for (double& c : s0_) c /= std::sqrt(n2);
            spin_ = true;
            nc_ = 10;
            double a = -1;
            if (std::abs(m_ - 1.0) < 1e-6) a = 0.00115965218128;                  // electron, positron
            else if (std::abs(m_ - 1836.15267343) < 1.0) a = 1.79284734463;       // proton
            else if (std::abs(m_ - 206.7682830) < 0.1) a = 0.00116592061;         // muon
            anom_ = cfg.get_double(name + ".anomalous_moment", a);
            if (anom_ == -1) throw std::runtime_error(name + ".spin: give " + name + ".anomalous_moment (a = (g-2)/2) for this mass");
        }
    }
    // ---- multiple Coulomb scattering on the plasma (small-angle, Gaussian per step)
    // on by default when the physical density is known (cheap: a few % of a beam push, negligible
    // against the plasma sweep in realistic runs); <beam>.scattering = 0 switches it off
    {
        const double n0 = cfg.get_double("units.n0_cm3", cfg.get_double("pusher.rr_n0_cm3", 0.0));
        const bool expl = cfg.has(name + ".scattering");
        sc_.on = cfg.get_bool(name + ".scattering", n0 > 0 && !rigid_);
        if (sc_.on && !(n0 > 0)) throw std::runtime_error(name + ".scattering needs the plasma density units.n0_cm3");
        if (sc_.on && rigid_ && !expl) sc_.on = false;
    }
    if (sc_.on) {
        const double n0 = cfg.get_double("units.n0_cm3", cfg.get_double("pusher.rr_n0_cm3", 0.0));
        const double re = 2.8179403262e-13;                                // cm
        const double kp = 5.64146e4 * std::sqrt(n0) / 2.99792458e10;     // 1/cm
        const double qm = q_ / m_;
        sc_.kappa = kp * re * qm * qm * cfg.get_double("scattering.factor", 1.0);
        sc_.Z = cfg.get_double("scattering.Z", 1.0);
        sc_.zeta = cfg.get_double("scattering.ion_charge", sc_.Z);
        const double A = cfg.get_double("scattering.A", sc_.Z <= 1 ? 1.0 : 2.0 * sc_.Z);
        if (sc_.Z < 1 || sc_.zeta < 0 || sc_.zeta > sc_.Z) throw std::runtime_error("scattering: need Z >= 1 and 0 <= ion_charge <= Z");
        const double a0 = 5.29177210903e-9;                                 // Bohr radius, cm
        sc_.a0 = a0 * kp;
        sc_.bmax_i = (sc_.zeta >= sc_.Z) ? 1.0 : std::min(1.0, 0.885 * a0 * std::pow(sc_.Z, -1.0 / 3.0) * kp);
        sc_.RN = 1.2e-13 * std::cbrt(A) * kp;                             // nuclear radius
        sc_.lamC = 3.8615926796e-11 * kp / m_;                            // hbar / (m c), m in m_e
        sc_.lnbmax_i = std::log(sc_.bmax_i);
        sc_.lnRN = std::log(sc_.RN);
        sc_.lnlamC = std::log(sc_.lamC);
        sc_.lna0 = std::log(sc_.a0);
        sc_.Li = cfg.get_double("scattering.coulomb_log_ions", -1.0);
        sc_.Le = cfg.get_double("scattering.coulomb_log_electrons", -1.0);
        sc_.seed = static_cast<uint64_t>(cfg.get_int("scattering.seed", 1)) * 0x9e3779b97f4a7c15ULL ^
                   std::hash<std::string>{}(name);
        if (rigid_) throw std::runtime_error(name + ".scattering: not for rigid beams");
    }
    if (qm) {   // empty beam, filled by add_new
        std::vector<double> none;
        set_particles(none);
        np_global_ = 0;
        gref_ = 0; xiref_ = 0;
        return;
    }
    const std::string prof = cfg.get_string(name + ".profile", "gaussian");
    analytic_ = cfg.get_bool(name + ".analytic", false);
    if (prof == "parsed") {
        parsed_ = true;
        if (!read_function(cfg, name + ".density", 3, dens_, {"x", "y", "xi"}))
            throw std::runtime_error(name + ": profile = parsed needs " + name + ".density(x,y,xi) = \"...\"");
    }
    if (analytic_ && parsed_) {
        // noise-free rigid beam from the parsed density, deposited directly on the grid
        rigid_ = true;
        an_gamma_ = cfg.get_double(name + ".gamma");
        Np_ = 0;
        for (View1D* v : {&x_, &y_, &px_, &py_, &pz_, &xi_, &w_}) *v = View1D(name + ".empty", 0);
        return;
    }
    if (parsed_) { init_parsed(cfg); return; }   // reference values set by the generator
    if (analytic_) {
        // noise-free rigid beam deposited directly on the grid (no macro-particles)
        rigid_ = true;
        an_n0_ = cfg.get_double(name + ".n0");
        an_sr_ = cfg.get_double(name + ".sigma_r");
        an_xi0_ = cfg.get_double(name + ".xi0");
        an_gamma_ = cfg.get_double(name + ".gamma");
        an_flat_ = (prof == "flattop");
        if (an_flat_) an_len_ = cfg.get_double(name + ".length");
        else if (prof == "gaussian") an_sxi_ = cfg.get_double(name + ".sigma_xi");
        else throw std::runtime_error(name + ": analytic beams need profile gaussian|flattop");
        an_cut_head_ = cfg.get_double(name + ".xi_cut_head", -1e300);
        an_x0_ = cfg.get_double(name + ".x0", 0.0); an_y0_ = cfg.get_double(name + ".y0", 0.0);
        an_xs_ = cfg.get_double(name + ".x_slope", 0.0); an_ys_ = cfg.get_double(name + ".y_slope", 0.0);
        if (!m1_ && (an_x0_ != 0 || an_y0_ != 0 || an_xs_ != 0 || an_ys_ != 0))
            throw std::runtime_error(name + ": transverse offsets/tilts need azimuthal mode 1 (modes = 1)");
        Np_ = 0;
        for (View1D* v : {&x_, &y_, &px_, &py_, &pz_, &xi_, &w_}) *v = View1D(name + ".empty", 0);
        return;
    }
    if (prof == "file") init_file(cfg.get_string(name + ".file"));
    else init_gaussian(cfg);
}

// reference gamma and xi (initial means of the whole beam, identical on all ranks): the
// moment sums are accumulated relative to them, which avoids the cancellation in
// <g^2> - <g>^2 for a beam with a small relative energy spread
void Beam::set_reference() {
    gref_ = 0; xiref_ = 0;
    const auto S = sums();
    if (S[0] > 0) { gref_ = S[1] / S[0]; xiref_ = S[3] / S[0]; }
}

// Analytic Gaussian displaced by c = (c_x, c_y) = d (cos phi, sin phi):
//   exp(-(r^2 + d^2 - 2 r d cos(th - phi)) / 2 s^2) = e^{-(r-d)^2/2s^2} [e^{-z} I0(z) + 2 e^{-z} I1(z) cos(th - phi) + ...]
//   z = r d / s^2  ->  mode 0: e^{-(r-d)^2/2s^2} e^{-z}I0(z);  mode 1 coefficient f1 = e^{-(r-d)^2/2s^2} e^{-z}I1(z) e^{-i phi}
void Beam::deposit_analytic(const View3D& src) const {
    const GridD g = grid_.d;
    const int K = bg_.nloc, M = grid_.N + 1;
    const Real xmin = bg_.xi_min, dxi = bg_.dxi;
    const int k0 = bg_.k0;   // local slice k is global slice k0 + k (same rounding as a serial run)
    const Real q = q_, n0 = an_n0_, s2 = an_sr_ * an_sr_, xi0 = an_xi0_, sxi = an_sxi_, len = an_len_;
    const Real cut = an_cut_head_, x0 = an_x0_, y0 = an_y0_, xs = an_xs_, ys = an_ys_;
    const bool flat = an_flat_, m1 = m1_;
    const Real gam = an_gamma_;
    const Real vz = Kokkos::sqrt(Real(1) - Real(1) / (gam * gam));
    const Real omvz = Real(1) / (gam * gam * (Real(1) + vz));
    Kokkos::parallel_for("beam.analytic", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {K, M}),
                         KOKKOS_LAMBDA(int k, int j) {
        const Real xi = xmin + (k0 + k) * dxi;
        if (xi < cut) return;
        Real fxi;
        if (flat) fxi = (Kokkos::fabs(xi - xi0) <= Real(0.5) * len) ? Real(1) : Real(0);
        else fxi = Kokkos::exp(-Real(0.5) * (xi - xi0) * (xi - xi0) / (sxi * sxi));
        if (fxi == Real(0)) return;
        const Real cx = x0 + xs * (xi - xi0), cy = y0 + ys * (xi - xi0);
        const Real d = Kokkos::sqrt(cx * cx + cy * cy);
        const Real r = g.r(j);
        const Real z = r * d / s2;
        Real i0, i1;
        bessel_i01_scaled(z, i0, i1);
        const Real env = q * n0 * fxi * Kokkos::exp(-Real(0.5) * (r - d) * (r - d) / s2);
        const Real rho0 = env * i0;
        src(k, j, B_RHO0) += rho0;
        src(k, j, B_RT0) += rho0 * omvz;
        src(k, j, B_JZ0) += rho0 * vz;
        if (m1 && d > Real(0)) {
            const Real cp = cx / d, sp = cy / d;          // f1 = env i1 e^{-i phi}
            const Real f1r = env * i1 * cp, f1i = -env * i1 * sp;
            src(k, j, B_RHO1R) += f1r;  src(k, j, B_RHO1I) += f1i;
            src(k, j, B_RT1R) += f1r * omvz;  src(k, j, B_RT1I) += f1i * omvz;
            src(k, j, B_JZ1R) += f1r * vz;  src(k, j, B_JZ1I) += f1i * vz;
        }
    });
}

// ---------------------------------------------------------------------------
// Parsed rigid beam: modes of rho(r, theta, xi) by azimuthal quadrature at every node
//   rho0 = < rho >,  rho1 = < rho e^{-i theta} >   (NQ-point rule, exact for |m| < NQ - 1)
void Beam::deposit_analytic_parsed(const View3D& src) const {
    const GridD g = grid_.d;
    const int K = bg_.nloc, M = grid_.N + 1;
    const Real xmin = bg_.xi_min, dxi = bg_.dxi;
    const int k0 = bg_.k0;   // local slice k is global slice k0 + k (same rounding as a serial run)
    const Real q = q_;
    const bool m1 = m1_;
    const ParserExec f = dens_.exec();
    const Real gam = an_gamma_;
    const Real vz = Kokkos::sqrt(Real(1) - Real(1) / (gam * gam));
    const Real omvz = Real(1) / (gam * gam * (Real(1) + vz));
    constexpr int NQ = 32;
    Kokkos::parallel_for("beam.analytic_parsed", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {K, M}),
                         KOKKOS_LAMBDA(int k, int j) {
        const Real xi = xmin + (k0 + k) * dxi, r = g.r(j);
        Real n0 = 0, nr = 0, ni = 0;
        for (int l = 0; l < NQ; ++l) {
            const Real th = Real(2) * Real(PI) * l / NQ, c = Kokkos::cos(th), s = Kokkos::sin(th);
            const Real n = f(r * c, r * s, xi);
            n0 += n; nr += n * c; ni -= n * s;
        }
        n0 /= NQ; nr /= NQ; ni /= NQ;
        if (j == 0) { nr = 0; ni = 0; }
        const Real rho0 = q * n0;
        src(k, j, B_RHO0) += rho0;
        src(k, j, B_RT0) += rho0 * omvz;
        src(k, j, B_JZ0) += rho0 * vz;
        if (m1) {
            const Real fr = q * nr, fi = q * ni;
            src(k, j, B_RHO1R) += fr;  src(k, j, B_RHO1I) += fi;
            src(k, j, B_RT1R) += fr * omvz;  src(k, j, B_RT1I) += fi * omvz;
            src(k, j, B_JZ1R) += fr * vz;  src(k, j, B_JZ1I) += fi * vz;
        }
    });
}

// ---------------------------------------------------------------------------
// Parsed macro-particle beam.  Particles are placed deterministically (quiet start):
//   radial: the cells of the simulation grid inside r_max, ppc_r sub-rings each, particles at the
//           volume centroid of the sub-ring (fine grid near the axis -> fine sampling there);
//   xi    : xi_range split into cells of dxi/ppc_xi;   azimuth: ntheta equidistant angles.
// weight = density(x,y,xi) * (sub-ring volume) * dxi_sub / ntheta.  Momenta: mean values from the
// parsed functions ux_mean(x,y,xi), uy_mean, uz_mean (or gamma); Gaussian spreads ux_std(x,y,xi), uy_std,
// uz_std (or u_std = sx sy sz).
void Beam::init_parsed(const Config& cfg) {
    const std::string p = name_;
    const auto xr = cfg.get_list(p + ".xi_range");
    if (xr.size() != 2) throw std::runtime_error(p + ".xi_range = xi_min xi_max is required for parsed beams");
    const double x0 = std::stod(xr[0]), x1 = std::stod(xr[1]);
    const double rmax = cfg.get_double(p + ".r_max");
    const auto ppc = cfg.get_list(p + ".ppc");
    const int pr = ppc.size() > 0 ? std::stoi(ppc[0]) : 1;
    const int pxi = ppc.size() > 1 ? std::stoi(ppc[1]) : 1;
    const int nth = cfg.get_int(p + ".ntheta", m1_ ? 8 : 1);
    if (m1_ && nth < 4) throw std::runtime_error(p + ".ntheta must be >= 4 with modes = 1");
    if (pr < 1 || pxi < 1 || nth < 1 || x1 <= x0 || rmax <= 0) throw std::runtime_error(p + ": bad parsed-beam sampling parameters");

    Parser ux, uy, uz;
    if (!read_function(cfg, p + ".ux_mean", 3, ux, {"x", "y", "xi"}, true)) ux = Parser("0", {});
    if (!read_function(cfg, p + ".uy_mean", 3, uy, {"x", "y", "xi"}, true)) uy = Parser("0", {});
    if (!read_function(cfg, p + ".uz_mean", 3, uz, {"x", "y", "xi"}, true)) {
        if (!cfg.has(p + ".gamma")) throw std::runtime_error(p + ": give " + p + ".uz_mean(x,y,xi) or " + p + ".gamma");
        const double g = cfg.get_double(p + ".gamma");
        uz = Parser(std::to_string(std::sqrt(g * g - 1.0)), {});
    }
    // Gaussian momentum spread: ux_std(x,y,xi), uy_std, uz_std (functions or numbers);
    // u_std = sx sy sz (numbers) is still accepted.  Negative values are treated as 0.
    Parser sx("0", {}), sy("0", {}), sz("0", {});
    const auto us = cfg.get_list(p + ".u_std");
    if (!us.empty()) {
        if (us.size() != 3) throw std::runtime_error(p + ".u_std needs 3 values (or use ux_std(x,y,xi) etc.)");
        sx = Parser(us[0], {}); sy = Parser(us[1], {}); sz = Parser(us[2], {});
    }
    read_function(cfg, p + ".ux_std", 3, sx, {"x", "y", "xi"}, true);
    read_function(cfg, p + ".uy_std", 3, sy, {"x", "y", "xi"}, true);
    read_function(cfg, p + ".uz_std", 3, sz, {"x", "y", "xi"}, true);
    auto spread = [](const Parser& f, double x, double y, double xi) { const double v = f(x, y, xi); return v > 0 ? v : 0.0; };
    std::mt19937_64 rng(cfg.get_int(p + ".seed", 12345));
    std::normal_distribution<double> N01(0.0, 1.0);

    const double dxs = bg_.dxi / pxi;
    const int nxs = std::max(1, static_cast<int>(std::lround((x1 - x0) / dxs)));
    std::vector<double> X, Y, PX, PY, PZ, XI, W;
    for (int j = 0; j < grid_.N && grid_.r[j] < rmax; ++j) {
        const double h = grid_.h[j];
        for (int m = 0; m < pr; ++m) {
            const double ra = grid_.r[j] + m * h / pr, rb = std::min(rmax, grid_.r[j] + (m + 1) * h / pr);
            if (rb <= ra) continue;
            const double r = (2.0 / 3.0) * (rb * rb * rb - ra * ra * ra) / (rb * rb - ra * ra);
            const double vol = PI * (rb * rb - ra * ra) * (x1 - x0) / nxs / nth;
            const double off = ((j * pr + m) % 2) * 0.5;
            for (int k = 0; k < nxs; ++k) {
                const double xi = x0 + (k + 0.5) * (x1 - x0) / nxs;
                for (int l = 0; l < nth; ++l) {
                    const double th = 2 * PI * (l + off) / nth;
                    const double x = r * std::cos(th), y = r * std::sin(th);
                    double n;
                    if (nth == 1) {   // axisymmetric run: azimuthal average (m = 0 part of the profile)
                        n = 0;
                        for (int q = 0; q < 16; ++q) n += dens_(r * std::cos(2 * PI * q / 16), r * std::sin(2 * PI * q / 16), xi);
                        n /= 16;
                    } else {
                        n = dens_(x, y, xi);
                    }
                    if (!(n > 0)) continue;
                    const double px = ux(x, y, xi) + spread(sx, x, y, xi) * N01(rng);
                    const double py = uy(x, y, xi) + spread(sy, x, y, xi) * N01(rng);
                    const double pz = uz(x, y, xi) + spread(sz, x, y, xi) * N01(rng);
                    if (pz <= 0) throw std::runtime_error(p + ": uz <= 0 at xi = " + std::to_string(xi));
                    X.push_back(x); Y.push_back(y); PX.push_back(px); PY.push_back(py); PZ.push_back(pz);
                    XI.push_back(xi); W.push_back(n * vol);
                }
            }
        }
    }
    if (X.empty()) throw std::runtime_error(p + ": the parsed density is zero everywhere inside r_max / xi_range");
    std::vector<double> all(7 * X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        const double q[7] = {X[i], Y[i], PX[i], PY[i], PZ[i], XI[i], W[i]};
        std::copy(q, q + 7, all.begin() + 7 * i);
    }
    adopt_local(all);
}

void Beam::init_gaussian(const Config& cfg) {
    const std::string p = name_;
    const std::string lprof = cfg.get_string(p + ".profile", "gaussian");   // gaussian | flattop
    const double n0 = cfg.get_double(p + ".n0");
    const double sr = cfg.get_double(p + ".sigma_r");
    const double xi0 = cfg.get_double(p + ".xi0");
    const double gamma0 = cfg.get_double(p + ".gamma");
    const double espread = cfg.get_double(p + ".espread", 0.0);
    const double emit = cfg.get_double(p + ".emit_n", 0.0);
    const double cut = cfg.get_double(p + ".cut", 3.0);
    // transverse centroid:  x_c(xi) = x0 + x_slope (xi - xi0);  angles xp0, yp0 (p_x = p_z xp0)
    const double cx0 = cfg.get_double(p + ".x0", 0.0), cy0 = cfg.get_double(p + ".y0", 0.0);
    const double cxs = cfg.get_double(p + ".x_slope", 0.0), cys = cfg.get_double(p + ".y_slope", 0.0);
    const double xp0 = cfg.get_double(p + ".xp0", 0.0), yp0 = cfg.get_double(p + ".yp0", 0.0);
    if (!m1_ && (cx0 != 0 || cy0 != 0 || cxs != 0 || cys != 0 || xp0 != 0 || yp0 != 0))
        throw std::runtime_error(p + ": transverse offsets/tilts need azimuthal mode 1 (modes = 1)");
    Np_ = cfg.get_int(p + ".nparticles", 100000);
    // quiet start: every sampled particle is replicated at nsym azimuths around the beam
    // centroid (removes the m = 1 sampling noise of a symmetric beam exactly)
    const int nrot = cfg.get_int(p + ".nsym", 1);
    // mirror_y: add the image (x, -y, p_x, -p_y) of every particle (keeps a beam that is
    // displaced/tilted only in x exactly symmetric in y)
    const bool mirror = cfg.get_bool(p + ".mirror_y", false);
    const int nsym = nrot * (mirror ? 2 : 1);
    if (nrot < 1 || Np_ % nsym != 0)
        throw std::runtime_error(p + ".nparticles must be a multiple of nsym (x2 with mirror_y)");
    const unsigned long long seed = cfg.get_int(p + ".seed", 12345);
    const double xcut = cfg.get_double(p + ".xi_cut_head", -1e300);

    double sxi = 0, len = 0;
    if (lprof == "gaussian") sxi = cfg.get_double(p + ".sigma_xi");
    else if (lprof == "flattop") len = cfg.get_double(p + ".length");
    else throw std::runtime_error(p + ".profile must be gaussian|flattop|file");

    const double Pr = 1.0 - std::exp(-0.5 * cut * cut);
    auto Phi = [](double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); };
    double wtot;
    if (lprof == "gaussian") {
        const double lo = std::max(-cut, (xcut - xi0) / sxi);
        if (lo >= cut) throw std::runtime_error(p + ": xi_cut_head removes the whole beam");
        wtot = n0 * std::pow(2 * PI, 1.5) * sr * sr * sxi * Pr * (Phi(cut) - Phi(lo));
    } else {
        const double lo = std::max(xi0 - 0.5 * len, xcut), hi = xi0 + 0.5 * len;
        if (lo >= hi) throw std::runtime_error(p + ": xi_cut_head removes the whole beam");
        wtot = n0 * 2 * PI * sr * sr * Pr * (hi - lo);
    }
    const double wmac = wtot / Np_;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double spx = (sr > 0) ? emit / sr : 0.0;

    const int Ntot = Np_;
    RefAcc acc;
    std::vector<double> local;
    for (int i0 = 0; i0 < Ntot; i0 += nsym) {
        double x, y;
        do { x = sr * N01(rng); y = sr * N01(rng); } while (x * x + y * y > cut * cut * sr * sr);
        double xi;
        do {
            if (lprof == "gaussian") { do { xi = N01(rng); } while (std::abs(xi) > cut); xi = xi0 + sxi * xi; }
            else xi = xi0 + len * (U01(rng) - 0.5);
        } while (xi < xcut);
        double px = spx * N01(rng), py = spx * N01(rng);
        const double gam = gamma0 * (1.0 + espread * N01(rng));
        px += gam * xp0;
        py += gam * yp0;
        const double pz2 = gam * gam - 1.0 - px * px - py * py;
        if (pz2 <= 0) throw std::runtime_error(p + ": gamma too small for the transverse momentum spread");
        // thermal part of the transverse momentum (the angle xp0, yp0 is not rotated)
        const double tpx = px - gam * xp0, tpy = py - gam * yp0;
        const bool keep = local_xi(xi);
        for (int l = 0; l < nsym; ++l) {
            const int lr = l % nrot;
            const double sgn = (l >= nrot) ? -1.0 : 1.0;   // mirrored copies: y -> -y
            const double ph = 2 * PI * lr / nrot, cp = std::cos(ph), sp = std::sin(ph);
            const double q[7] = {cp * x - sp * y + cx0 + cxs * (xi - xi0), sgn * (sp * x + cp * y) + cy0 + cys * (xi - xi0),
                                 cp * tpx - sp * tpy + gam * xp0, sgn * (sp * tpx + cp * tpy) + gam * yp0,
                                 std::sqrt(pz2), xi, wmac};
            accumulate(acc, q);
            if (keep) local.insert(local.end(), q, q + 7);
        }
    }
    finish_init(local, acc);
}

bool Beam::local_xi(double xi) const {
    // as restrict_to_local: particles outside the box stay with the first / last rank
    const int k = slice_of(xi);
    if (bg_.k0 > 0 && k < bg_.k0) return false;
    if (bg_.k0 + bg_.nloc < bg_.nxi && k >= bg_.k0 + bg_.nloc) return false;
    return true;
}

void Beam::accumulate(RefAcc& a, const double* p) const {
    const double w = p[6];
    ++a.n;
    if (w == 0) return;
    const double g = std::sqrt(1 + p[2] * p[2] + p[3] * p[3] + p[4] * p[4]);   // as in sums()
    a.w += w;
    a.wg += w * g;
    a.wxi += w * p[5];
}

void Beam::finish_init(std::vector<double>& local, const RefAcc& a) {
    np_global_ = a.n;
    if (spin_) {   // generators give 7 doubles per particle; append the initial spin
        const size_t n = local.size() / 7;
        std::vector<double> e(10 * n);
        for (size_t i = 0; i < n; ++i) {
            std::copy(local.begin() + 7 * i, local.begin() + 7 * i + 7, e.begin() + 10 * i);
            e[10 * i + 7] = s0_[0]; e[10 * i + 8] = s0_[1]; e[10 * i + 9] = s0_[2];
        }
        local.swap(e);
    }
    set_particles(local);
    std::vector<double>().swap(local);
    gref_ = 0; xiref_ = 0;
    if (a.w > 0) { gref_ = a.wg / a.w; xiref_ = a.wxi / a.w; }
}

void Beam::adopt_local(const std::vector<double>& all) {
    RefAcc acc;
    std::vector<double> local;
    for (size_t i = 0; i < all.size(); i += 7) {
        accumulate(acc, &all[i]);
        if (local_xi(all[i + 5])) local.insert(local.end(), all.begin() + i, all.begin() + i + 7);
    }
    finish_init(local, acc);
}

// text file, one particle per line:  x  y  p_x  p_y  p_z  xi  w
void Beam::init_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error(name_ + ": cannot open beam file '" + filename + "'");
    std::vector<double> v[7];
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double a[7];
        if (!(ss >> a[0] >> a[1] >> a[2] >> a[3] >> a[4] >> a[5] >> a[6])) continue;
        for (int c = 0; c < 7; ++c) v[c].push_back(a[c]);
    }
    std::vector<double> all(7 * v[0].size());
    for (size_t i = 0; i < v[0].size(); ++i)
        for (int c = 0; c < 7; ++c) all[7 * i + c] = v[c][i];
    adopt_local(all);
}

} // namespace quarz
