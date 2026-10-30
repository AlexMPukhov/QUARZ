#include "Beam.hpp"
#include <functional>
#include "Ionization.hpp"
#include "Config.hpp"
#include "Pushers.hpp"
#include "SliceData.hpp"

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

namespace {
// exponentially scaled modified Bessel functions e^{-z} I0(z), e^{-z} I1(z), z >= 0
// (Abramowitz & Stegun 9.8.1-9.8.4, |rel. error| < 2e-7)
KOKKOS_INLINE_FUNCTION void bessel_i01_scaled(Real z, Real& i0, Real& i1) {
    if (z < Real(3.75)) {
        const Real t = (z / Real(3.75)) * (z / Real(3.75));
        const Real I0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        const Real I1 = z * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        const Real e = Kokkos::exp(-z);
        i0 = I0 * e; i1 = I1 * e;
    } else {
        const Real u = Real(3.75) / z;
        const Real s = Real(1) / Kokkos::sqrt(z);
        i0 = s * (0.39894228 + u * (0.01328592 + u * (0.00225319 + u * (-0.00157565 + u * (0.00916281 + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377))))))));
        i1 = s * (0.39894228 + u * (-0.03988024 + u * (-0.00362018 + u * (0.00163801 + u * (-0.01031555 + u * (0.02282967 + u * (-0.02895312 + u * (0.01787654 - u * 0.00420059))))))));
    }
}
} // namespace

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

    // Rods: <beam>.rods = momentum samples per radial position (even: antithetic pairs +-g). A rod keeps its
    // transverse sample (r, theta, the unit Gaussian draws g of the momentum spreads) at every xi, so the
    // transverse mixing of the rods stays smooth along xi: no macro-particle noise in xi, also after the
    // bunch has evolved. <beam>.dr: radial shell width of the loader (default: the cells of the grid).
    const int nrod = cfg.get_int(p + ".rods", 0);
    if (nrod < 0 || nrod % 2) throw std::runtime_error(p + ".rods must be an even number (antithetic pairs)");
    const double drl = cfg.get_double(p + ".dr", 0.0);
    // Shot noise of the real particles: the macro-particle stands for N = w nu real particles (nu per unit
    // weight); its weight is redrawn from Poisson(N) (Gaussian N + sqrt(N) g above N = 100). Each phase-space
    // cell then carries the count noise of the real beam: the right noise power at every k (up to the
    // sampling limit 1/dxi) and for every radial kernel.  <beam>.shot_noise = amplitude factor (1 = physical).
    const double shot = cfg.get_double(p + ".shot_noise", 0.0);
    double nu = 0;
    if (shot > 0) {
        const double n0 = cfg.get_double("units.n0_cm3", 0.0);
        if (!(n0 > 0)) throw std::runtime_error(p + ".shot_noise needs the density unit units.n0_cm3");
        const double kp_inv = 2.99792458e10 / (5.64146e4 * std::sqrt(n0));   // c/omega_p in cm
        nu = n0 * kp_inv * kp_inv * kp_inv;
    }
    std::mt19937_64 rng_shot(cfg.get_int(p + ".seed", 12345) ^ 0x5d1f2a3b7c4e9081ULL);
    std::normal_distribution<double> G01(0.0, 1.0);
    auto noisy = [&](double w) {
        if (!(shot > 0)) return w;
        const double N = w * nu;
        double Nn;
        if (N < 100 && shot == 1.0) Nn = static_cast<double>(std::poisson_distribution<long long>(N)(rng_shot));
        else Nn = std::max(0.0, N + shot * std::sqrt(N) * G01(rng_shot));
        return Nn / nu;
    };

    std::vector<std::array<double, 3>> shells;   // ra, rb, parity of the azimuthal offset
    if (drl > 0) {
        for (int j = 0; j * drl < rmax; ++j)
            for (int m = 0; m < pr; ++m) {
                const double ra = j * drl + m * drl / pr, rb = std::min(rmax, j * drl + (m + 1) * drl / pr);
                if (rb > ra) shells.push_back({ra, rb, double((j * pr + m) % 2)});
            }
    } else {
        for (int j = 0; j < grid_.N && grid_.r[j] < rmax; ++j) {
            const double h = grid_.h[j];
            for (int m = 0; m < pr; ++m) {
                const double ra = grid_.r[j] + m * h / pr, rb = std::min(rmax, grid_.r[j] + (m + 1) * h / pr);
                if (rb > ra) shells.push_back({ra, rb, double((j * pr + m) % 2)});
            }
        }
    }

    const double dxs = bg_.dxi / pxi;
    const int nxs = std::max(1, static_cast<int>(std::lround((x1 - x0) / dxs)));
    std::vector<double> X, Y, PX, PY, PZ, XI, W;
    auto density = [&](double r, double x, double y, double xi) {
        if (nth != 1) return dens_(x, y, xi);
        double n = 0;   // axisymmetric run: azimuthal average (m = 0 part of the profile)
        for (int q = 0; q < 16; ++q) n += dens_(r * std::cos(2 * PI * q / 16), r * std::sin(2 * PI * q / 16), xi);
        return n / 16;
    };
    for (const auto& sh : shells) {
        const double ra = sh[0], rb = sh[1];
        const double r = (2.0 / 3.0) * (rb * rb * rb - ra * ra * ra) / (rb * rb - ra * ra);
        const double vol = PI * (rb * rb - ra * ra) * (x1 - x0) / nxs / nth;
        const double off = sh[2] * 0.5;
        if (nrod == 0) {
            for (int k = 0; k < nxs; ++k) {
                const double xi = x0 + (k + 0.5) * (x1 - x0) / nxs;
                for (int l = 0; l < nth; ++l) {
                    const double th = 2 * PI * (l + off) / nth;
                    const double x = r * std::cos(th), y = r * std::sin(th);
                    const double n = density(r, x, y, xi);
                    if (!(n > 0)) continue;
                    const double px = ux(x, y, xi) + spread(sx, x, y, xi) * N01(rng);
                    const double py = uy(x, y, xi) + spread(sy, x, y, xi) * N01(rng);
                    const double pz = uz(x, y, xi) + spread(sz, x, y, xi) * N01(rng);
                    if (pz <= 0) throw std::runtime_error(p + ": uz <= 0 at xi = " + std::to_string(xi));
                    X.push_back(x); Y.push_back(y); PX.push_back(px); PY.push_back(py); PZ.push_back(pz);
                    XI.push_back(xi); W.push_back(noisy(n * vol));
                }
            }
            continue;
        }
        for (int l = 0; l < nth; ++l) {
            const double th = 2 * PI * (l + off) / nth;
            const double x = r * std::cos(th), y = r * std::sin(th);
            // nrod/2 draws, each component scaled to unit variance over the set (with the antithetic partners:
            // exact first and second moments of the momentum spreads in every shell)
            std::vector<std::array<double, 3>> gs(nrod / 2);
            double v2[3] = {0, 0, 0};
            for (auto& g : gs)
                for (int c = 0; c < 3; ++c) { g[c] = N01(rng); v2[c] += g[c] * g[c]; }
            for (auto& g : gs)
                for (int c = 0; c < 3; ++c) g[c] *= v2[c] > 0 ? std::sqrt(gs.size() / v2[c]) : 0.0;
            for (int q = 0; q < nrod / 2; ++q) {
                const auto& g = gs[q];
                for (int s = 1; s >= -1; s -= 2) {
                    for (int k = 0; k < nxs; ++k) {
                        const double xi = x0 + (k + 0.5) * (x1 - x0) / nxs;
                        const double n = density(r, x, y, xi);
                        if (!(n > 0)) continue;
                        const double px = ux(x, y, xi) + spread(sx, x, y, xi) * s * g[0];
                        const double py = uy(x, y, xi) + spread(sy, x, y, xi) * s * g[1];
                        const double pz = uz(x, y, xi) + spread(sz, x, y, xi) * s * g[2];
                        if (pz <= 0) throw std::runtime_error(p + ": uz <= 0 at xi = " + std::to_string(xi));
                        X.push_back(x); Y.push_back(y); PX.push_back(px); PY.push_back(py); PZ.push_back(pz);
                        XI.push_back(xi); W.push_back(noisy(n * vol / nrod));
                    }
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

void Beam::deposit(const View3D& src) const {
    if (passive_) return;
    if (analytic_ && parsed_) { deposit_analytic_parsed(src); return; }
    if (analytic_) { deposit_analytic(src); return; }
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto pz = pz_; auto xi = xi_; auto w = w_;
    const Real q = q_;
    const Real xmin = bg_.xi_min, idxi = Real(1) / bg_.dxi;
    const int k0 = bg_.k0, nloc = bg_.nloc;
    const bool ngp = bg_.ngp;
    const Real R = grid_.R;
    const bool m1 = m1_;
    Kokkos::parallel_for("beam.deposit", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real X = x(i), Y = y(i);
        const Real r = Kokkos::sqrt(X * X + Y * Y);
        if (r > R) return;
        const Real c = r > Real(0) ? X / r : Real(1), s = r > Real(0) ? Y / r : Real(0);
        const Real sx = (xi(i) - xmin) * idxi;
        // xi shape: linear (CIC) or nearest slice (NGP, slice-local: needed for the xi decomposition)
        const int k = ngp ? static_cast<int>(Kokkos::floor(sx + Real(0.5))) : static_cast<int>(Kokkos::floor(sx));
        if (k < k0 - 1 || k >= k0 + nloc) return;
        const Real tk = ngp ? Real(0) : sx - k;
        const Real p2 = px(i) * px(i) + py(i) * py(i);
        const Real gam = Kokkos::sqrt(Real(1) + p2 + pz(i) * pz(i));
        const Real omvz = (Real(1) + p2) / (gam * (gam + pz(i)));   // 1 - v_z without cancellation
        const Real vz = pz(i) / gam, vx = px(i) / gam, vy = py(i) / gam;
        const Real vr = c * vx + s * vy, vt = c * vy - s * vx;
        int j; Real t;
        g.weights(r, j, t);
        const Real Wr[2] = {Real(1) - t, t};
        const Real Wk[2] = {Real(1) - tk, tk};
        for (int a = 0; a < (ngp ? 1 : 2); ++a) {
            const int kk = k + a - k0;          // local slice
            if (kk < 0 || kk >= nloc) continue;
            for (int b = 0; b < 2; ++b) {
                const int jj = j + b;
                const Real d = wi * q * Wk[a] * Wr[b] * g.inv_V(jj) * idxi;
                Kokkos::atomic_add(&src(kk, jj, B_RT0), d * omvz);
                Kokkos::atomic_add(&src(kk, jj, B_JZ0), d * vz);
                Kokkos::atomic_add(&src(kk, jj, B_JP1R), d * vr);
                Kokkos::atomic_add(&src(kk, jj, B_JP1I), d * vt);
                Kokkos::atomic_add(&src(kk, jj, B_RHO0), d);
                if (m1) {
                    Kokkos::atomic_add(&src(kk, jj, B_RT1R), d * omvz * c);
                    Kokkos::atomic_add(&src(kk, jj, B_RT1I), -d * omvz * s);
                    Kokkos::atomic_add(&src(kk, jj, B_JZ1R), d * vz * c);
                    Kokkos::atomic_add(&src(kk, jj, B_JZ1I), -d * vz * s);
                    Kokkos::atomic_add(&src(kk, jj, B_RHO1R), d * c);
                    Kokkos::atomic_add(&src(kk, jj, B_RHO1I), -d * s);
                    Kokkos::atomic_add(&src(kk, jj, B_JP0R), d * vx);
                    Kokkos::atomic_add(&src(kk, jj, B_JP0I), d * vy);
                    const Real c2 = c * c - s * s, s2 = Real(2) * c * s;
                    Kokkos::atomic_add(&src(kk, jj, B_JP2R), d * (vx * c2 + vy * s2));
                    Kokkos::atomic_add(&src(kk, jj, B_JP2I), d * (vy * c2 - vx * s2));
                }
            }
        }
    });
}

namespace {
// delta electrons: ln W_max (m c^2) and kappa = (E + m c^2)/(p c) of a projectile of mass M (electron
// masses), charge sign qs, Lorentz factor gam, u2 = (beta gamma)^2  (see Ionization.hpp)
KOKKOS_INLINE_FUNCTION void delta_kin(Real M, Real qs, Real gam, Real u2, Real& lnwm, Real& kap) {
    Real wm;
    if (M == Real(1)) wm = qs < Real(0) ? Real(0.5) * (gam - Real(1)) : gam - Real(1);
    else wm = Real(2) * u2 / (Real(1) + Real(2) * gam / M + Real(1) / (M * M));
    lnwm = Kokkos::log(wm);
    kap = (gam + Real(1) / M) / Kokkos::sqrt(u2);
}
} // namespace

void Beam::deposit_impact(const View3D& imp) const {
    if (passive_) return;
    const GridD g = grid_.d;
    const Real z2 = q_ * q_;
    const bool dl = imp.extent(2) > 3;   // delta-electron sums
    const Real Mb = m_, qs = q_;
    if (analytic_) {   // rigid analytic beam: m = 0 density at every node
        const int K = bg_.nloc, M = grid_.N + 1;
        const Real xmin = bg_.xi_min, dxi = bg_.dxi;
        const int k0 = bg_.k0;
        const Real gam = an_gamma_;
        const Real b2 = Real(1) - Real(1) / (gam * gam), be = std::sqrt(b2);
        const Real L = std::log(b2 * gam * gam) - b2;
        const Real c0 = z2 * be, c1 = z2 * L / be, c2 = z2 / be;
        Real lw = 0, ka = 0;
        if (dl) delta_kin(Mb, qs, gam, b2 * gam * gam, lw, ka);
        const Real c3 = c2 * lw, c4 = c2 * ka;
        if (parsed_) {
            const ParserExec f = dens_.exec();
            constexpr int NQ = 32;
            Kokkos::parallel_for("beam.impact_parsed", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {K, M}),
                                 KOKKOS_LAMBDA(int k, int j) {
                const Real xi = xmin + (k0 + k) * dxi, r = g.r(j);
                Real n = 0;
                for (int l = 0; l < NQ; ++l) {
                    const Real th = Real(2) * Real(PI) * l / NQ;
                    n += f(r * Kokkos::cos(th), r * Kokkos::sin(th), xi);
                }
                n /= NQ;
                imp(k, j, 0) += c0 * n; imp(k, j, 1) += c1 * n; imp(k, j, 2) += c2 * n;
                if (dl) { imp(k, j, 3) += c3 * n; imp(k, j, 4) += c4 * n; }
            });
            return;
        }
        const Real n0 = an_n0_, s2 = an_sr_ * an_sr_, xi0 = an_xi0_, sxi = an_sxi_, len = an_len_;
        const Real cut = an_cut_head_, x0 = an_x0_, y0 = an_y0_, xs = an_xs_, ys = an_ys_;
        const bool flat = an_flat_;
        Kokkos::parallel_for("beam.impact_analytic", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {K, M}),
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
            Real i0, i1;
            bessel_i01_scaled(r * d / s2, i0, i1);
            const Real n = n0 * fxi * Kokkos::exp(-Real(0.5) * (r - d) * (r - d) / s2) * i0;
            imp(k, j, 0) += c0 * n; imp(k, j, 1) += c1 * n; imp(k, j, 2) += c2 * n;
            if (dl) { imp(k, j, 3) += c3 * n; imp(k, j, 4) += c4 * n; }
        });
        return;
    }
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto pz = pz_; auto xi = xi_; auto w = w_;
    const Real xmin = bg_.xi_min, idxi = Real(1) / bg_.dxi;
    const int k0 = bg_.k0, nloc = bg_.nloc;
    const bool ngp = bg_.ngp;
    const Real R = grid_.R;
    Kokkos::parallel_for("beam.impact", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real X = x(i), Y = y(i);
        const Real r = Kokkos::sqrt(X * X + Y * Y);
        if (r > R) return;
        const Real sx = (xi(i) - xmin) * idxi;
        const int k = ngp ? static_cast<int>(Kokkos::floor(sx + Real(0.5))) : static_cast<int>(Kokkos::floor(sx));
        if (k < k0 - 1 || k >= k0 + nloc) return;
        const Real tk = ngp ? Real(0) : sx - k;
        const Real u2 = px(i) * px(i) + py(i) * py(i) + pz(i) * pz(i);
        const Real g2 = Real(1) + u2;
        const Real b2 = u2 / g2, be = Kokkos::sqrt(b2);
        if (!(be > Real(0))) return;
        const Real L = Kokkos::log(u2) - b2;   // ln(beta^2 gamma^2) - beta^2
        const Real c0 = z2 * be, c1 = z2 * L / be, c2 = z2 / be;
        Real lw = 0, ka = 0;
        if (dl) delta_kin(Mb, qs, Kokkos::sqrt(g2), u2, lw, ka);
        int j; Real t;
        g.weights(r, j, t);
        const Real Wr[2] = {Real(1) - t, t};
        const Real Wk[2] = {Real(1) - tk, tk};
        for (int a = 0; a < (ngp ? 1 : 2); ++a) {
            const int kk = k + a - k0;
            if (kk < 0 || kk >= nloc) continue;
            for (int b = 0; b < 2; ++b) {
                const int jj = j + b;
                const Real d = wi * Wk[a] * Wr[b] * g.inv_V(jj) * idxi;
                Kokkos::atomic_add(&imp(kk, jj, 0), d * c0);
                Kokkos::atomic_add(&imp(kk, jj, 1), d * c1);
                Kokkos::atomic_add(&imp(kk, jj, 2), d * c2);
                if (dl) {
                    Kokkos::atomic_add(&imp(kk, jj, 3), d * c2 * lw);
                    Kokkos::atomic_add(&imp(kk, jj, 4), d * c2 * ka);
                }
            }
        }
    });
}

void Beam::push(const View3D& fld, Real dt_kick, Real dt_drift, const View3D& pond, const View3D& diag, int step) {
    if (rigid_) return;
    // leapfrog start-up: momenta are given at t, the scheme needs them at t - dt_kick/2
    if (!started_) { advance(fld, -0.5 * dt_kick, -0.5 * dt_kick, false, pond); started_ = true; }
    advance(fld, dt_kick, dt_drift, true, pond, diag, step);
}

std::string Beam::scattering_description() const {
    if (!sc_.on) return "";
    std::ostringstream o;
    o << "Coulomb scattering on the plasma: Z = " << sc_.Z << ", ion charge " << sc_.zeta << ", k_p r_e (q/m)^2 = "
      << sc_.kappa << "; Coulomb logarithms "
      << (sc_.Li > 0 ? std::to_string(sc_.Li) : std::string("auto: ln(b_max / max(lambda_C/gamma, R_N))")) << " (ions), "
      << (sc_.Le > 0 ? std::to_string(sc_.Le) : std::string("auto: ln(b_max / (lambda_C sqrt(2/gamma)))")) << " (electrons)";
    return o.str();
}

double Beam::min_gamma_eff(double gthr) const {
    if (!adaptive_dt()) return std::numeric_limits<double>::infinity();
    auto px = px_; auto py = py_; auto pz = pz_; auto w = w_;
    const Real thr = gthr;
    Real g = std::numeric_limits<Real>::max();
    Kokkos::parallel_reduce("beam.min_gamma", Range(0, Np_), KOKKOS_LAMBDA(int i, Real& m) {
        if (w(i) == Real(0)) return;
        Real gi = Kokkos::sqrt(Real(1) + px(i) * px(i) + py(i) * py(i) + pz(i) * pz(i));
        if (gi < thr) gi = thr;
        if (gi < m) m = gi;
    }, Kokkos::Min<Real>(g));
    if (g == std::numeric_limits<Real>::max()) return std::numeric_limits<double>::infinity();
    return static_cast<double>(g) * m_ / std::abs(q_);
}

void Beam::add_new(const std::vector<double>& p7, const View3D& fld, Real dt_prev, const View3D& pond) {
    const size_t n = p7.size() / 7;
    if (n == 0) return;
    std::vector<double> e;
    if (spin_) {
        e.resize(10 * n);
        for (size_t i = 0; i < n; ++i) {
            std::copy(p7.begin() + 7 * i, p7.begin() + 7 * i + 7, e.begin() + 10 * i);
            e[10 * i + 7] = s0_[0]; e[10 * i + 8] = s0_[1]; e[10 * i + 9] = s0_[2];
        }
    }
    append(spin_ ? e.data() : p7.data(), n);
    // before the first push the start-up of push() moves all momenta back
    if (started_) advance(fld, -0.5 * dt_prev, -0.5 * dt_prev, false, pond, View3D(), 0, Np_ - static_cast<int>(n));
}

void Beam::advance(const View3D& fld, Real dt_kick, Real dt, bool move, const View3D& pond, const View3D& diag, int step, int i0) {
    const bool scat = sc_.on && move && diag.extent(0) > 0;
    const Scatter sc = sc_;
    const uint64_t skey = ion_mix(sc_.seed ^ ion_mix(static_cast<uint64_t>(step) + 0x51ed27ULL));
    const bool laser = pond.extent(0) > 0;
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto pz = pz_; auto xi = xi_; auto w = w_;
    const Real qm = q_ / m_;
    const Real xmin = bg_.xi_min, idxi = Real(1) / bg_.dxi;
    const Real xmax = bg_.xi_min + (bg_.nxi - 1) * bg_.dxi;
    const int k0 = bg_.k0, nloc = bg_.nloc;
    const bool ngp = bg_.ngp;
    const Real R = grid_.R;
    const int mode = static_cast<int>(pusher_);
    const Real rr = rr_;
    const bool m1 = m1_;
    const bool spin = spin_;
    const Real an = anom_;
    auto sx = sx_; auto sy = sy_; auto sz = sz_;
    Kokkos::parallel_for("beam.push", Range(i0, Np_), KOKKOS_LAMBDA(int i) {
        if (w(i) == Real(0)) return;
        const Real X = x(i), Y = y(i);
        const Real r = Kokkos::sqrt(X * X + Y * Y);
        const Real c = r > Real(0) ? X / r : Real(1), sn = r > Real(0) ? Y / r : Real(0);
        // --- gather (bilinear in r, xi) of all components, then mode sums at the azimuth
        // local slice index and weight (NGP: the particle's own slice only)
        Real s = (xi(i) - xmin) * idxi - k0;
        int k;
        Real tk;
        if (ngp) {
            k = static_cast<int>(Kokkos::floor(s + Real(0.5)));
            k = k < 0 ? 0 : (k > nloc - 1 ? nloc - 1 : k);
            tk = Real(0);
        } else {
            k = static_cast<int>(Kokkos::floor(s));
            if (k < 0) { k = 0; s = 0; }
            if (k > nloc - 2) { k = nloc - 2; s = nloc - 1; }
            tk = s - k;
        }
        const int k1 = ngp ? k : k + 1;
        int j; Real t;
        g.weights(r, j, t);
        auto bil = [&](int comp) {
            return (Real(1) - tk) * ((Real(1) - t) * fld(k, j, comp) + t * fld(k, j + 1, comp)) +
                   tk * ((Real(1) - t) * fld(k1, j, comp) + t * fld(k1, j + 1, comp));
        };
        Real Ex, Ey, Bx, By, Ez, Bz;
        if (m1) {
            eval_vector(bil(F_EP0R), bil(F_EP0I), bil(F_EP1R), bil(F_EP1I), bil(F_EP2R), bil(F_EP2I), c, sn, true, Ex, Ey);
            eval_vector(bil(F_BP0R), bil(F_BP0I), bil(F_BP1R), bil(F_BP1I), bil(F_BP2R), bil(F_BP2I), c, sn, true, Bx, By);
            Ez = eval_scalar(bil(F_EZ0), bil(F_EZ1R), bil(F_EZ1I), c, sn, true);
            Bz = eval_scalar(bil(F_BZ0), bil(F_BZ1R), bil(F_BZ1I), c, sn, true);
        } else {
            eval_vector(Real(0), Real(0), bil(F_EP1R), bil(F_EP1I), Real(0), Real(0), c, sn, false, Ex, Ey);
            eval_vector(Real(0), Real(0), bil(F_BP1R), bil(F_BP1I), Real(0), Real(0), c, sn, false, Bx, By);
            Ez = bil(F_EZ0);
            Bz = bil(F_BZ0);
        }
        const push::V3 E{Ex, Ey, Ez}, B{Bx, By, Bz};
        push::V3 u{px(i), py(i), pz(i)};
        const Real h = Real(0.5) * qm * dt_kick;
        // laser ponderomotive force (time-averaged):  du/dt = -qhat^2 grad<a^2> / (2 gamma_bar),  d/dz = -d/dxi
        push::V3 Fp{Real(0), Real(0), Real(0)};
        if (laser) {
            auto bilp = [&](int comp) {
                return (Real(1) - tk) * ((Real(1) - t) * pond(k, j, comp) + t * pond(k, j + 1, comp)) +
                       tk * ((Real(1) - t) * pond(k1, j, comp) + t * pond(k1, j + 1, comp));
            };
            const Real A = bilp(0), dAr = bilp(1), dAxi = bilp(2);
            const Real gb = Kokkos::sqrt(Real(1) + push::dot(u, u) + qm * qm * A);
            const Real f = -Real(0.5) * qm * qm / gb;
            Fp = push::V3{f * dAr * c, f * dAr * sn, -f * dAxi};
        }
        // Thomas-BMT precession, ds/dt = (q/m) s x W,
        // W = (a + 1/g) B - a g/(g+1) (beta.B) beta - (a + 1/(g+1)) beta x E, at the mean momentum ub;
        // Boris rotation (|s| exact to round-off). Spin lives at half steps like the momentum.
        auto rotate_spin = [&](const push::V3& ub, Real tau) {
            const Real gb = Kokkos::sqrt(Real(1) + push::dot(ub, ub));
            const push::V3 be = (Real(1) / gb) * ub;
            const push::V3 W = (an + Real(1) / gb) * B - (an * gb / (gb + Real(1)) * push::dot(be, B)) * be -
                               (an + Real(1) / (gb + Real(1))) * push::cross(be, E);
            const push::V3 tv = (Real(0.5) * qm * tau) * W;
            const push::V3 s0{sx(i), sy(i), sz(i)};
            const push::V3 s1 = s0 + push::cross(s0, tv);
            const push::V3 s2 = s0 + (Real(2) / (Real(1) + push::dot(tv, tv))) * push::cross(s1, tv);
            sx(i) = s2.x; sy(i) = s2.y; sz(i) = s2.z;
        };
        if (!move) {
            if (spin) rotate_spin(u, dt_kick);
            const Real g0 = Kokkos::sqrt(Real(1) + push::dot(u, u));
            u = u + (Real(2) * h) * (E + push::cross((Real(1) / g0) * u, B)) + dt_kick * Fp;
            px(i) = u.x; py(i) = u.y; pz(i) = u.z;
            return;
        }
        const push::V3 uold = u;
        u = u + (Real(0.5) * dt_kick) * Fp;   // half kick (Strang splitting around the Lorentz push)
        switch (mode) {
            case 0: u = push::vay(u, E, B, h); break;
            case 1: u = push::boris(u, E, B, h); break;
            case 2: u = push::higuera_cary(u, E, B, h); break;
            case 3: u = push::imp(u, E, B, h); break;
            default: {
                const Real nu = push::rr_rate(u, E, B, qm, rr);
                u = push::imp_rr(u, E, B, h, nu * dt_kick);
            }
        }
        u = u + (Real(0.5) * dt_kick) * Fp;
        if (spin) rotate_spin(Real(0.5) * (uold + u), dt_kick);
        if (scat) {
            // multiple Coulomb scattering, d<p_x^2>/dt = kappa [Z^2 n_i L_i + (n_e + (Z - zeta) n_i) L_e]
            // per transverse axis; elastic: |p| is kept. Random numbers from a hash of the particle's
            // state at the start of the step (independent of its memory index and rank).
            auto bild = [&](int comp) {
                return (Real(1) - tk) * ((Real(1) - t) * diag(k, j, comp) + t * diag(k, j + 1, comp)) +
                       tk * ((Real(1) - t) * diag(k1, j, comp) + t * diag(k1, j + 1, comp));
            };
            const Real ne = Kokkos::fmax(bild(1), Real(0)), ni = Kokkos::fmax(bild(2), Real(0));   // D_NE0, D_NI0
            const Real pp = Kokkos::sqrt(push::dot(u, u));
            // Coulomb logarithms from one log(gamma) (lnb* = precomputed logs of the lengths)
            const Real lg = Kokkos::log(Kokkos::sqrt(Real(1) + pp * pp));
            const Real Li = sc.Li > 0 ? sc.Li : sc.lnbmax_i - Kokkos::fmax(sc.lnlamC - lg, sc.lnRN);
            const Real lnbe = sc.lnlamC + Real(0.5) * (Real(0.6931471805599453) - lg);   // ln(lamC sqrt(2/gamma))
            const Real Le = sc.Le > 0 ? sc.Le : -lnbe;
            const Real Lb = sc.Le > 0 ? sc.Le : Kokkos::fmax(sc.lna0 - lnbe, Real(0));
            const Real D = sc.kappa * (sc.Z * sc.Z * ni * Kokkos::fmax(Li, Real(0)) + ne * Kokkos::fmax(Le, Real(0)) +
                                       (sc.Z - sc.zeta) * ni * Kokkos::fmax(Lb, Real(0)));
            if (D > Real(0)) {
                union { double d; uint64_t u; } bx{X}, by{Y}, bz{xi(i)}, bp{pz(i)};
                const uint64_t id = ion_mix(bx.u ^ ion_mix(by.u ^ ion_mix(bz.u ^ ion_mix(bp.u))));
                const Real u1 = Kokkos::fmax(ion_rand(skey, id, 0), Real(1e-300)), u2 = ion_rand(skey, id, 1);
                const Real s = Kokkos::sqrt(D * Kokkos::fabs(dt_kick)) * Kokkos::sqrt(Real(-2) * Kokkos::log(u1));
                const Real ph = Real(6.283185307179586) * u2;
                const push::V3 ub = u;
                u.x += s * Kokkos::cos(ph);
                u.y += s * Kokkos::sin(ph);
                const Real pt2 = u.x * u.x + u.y * u.y;
                const Real pz2 = pp * pp - pt2;
                if (pz2 > Real(0)) u.z = (u.z >= Real(0) ? Real(1) : Real(-1)) * Kokkos::sqrt(pz2);
                if (spin) {
                    // the deflection is electric (fields of charges at rest): the spin turns about the same
                    // axis by kappa theta, kappa = g beta^2 (a + 1/(g+1)) (T-BMT with B = 0, E perp beta)
                    const push::V3 ax = push::cross(ub, u);
                    const Real sa = Kokkos::sqrt(push::dot(ax, ax));
                    if (sa > Real(0)) {
                        const Real gg = Kokkos::sqrt(Real(1) + pp * pp);
                        const Real ang = pp * pp / gg * (an + Real(1) / (gg + Real(1))) * Kokkos::atan2(sa, push::dot(ub, u));
                        const push::V3 n = (Real(1) / sa) * ax;
                        const push::V3 s0{sx(i), sy(i), sz(i)};
                        const Real ca = Kokkos::cos(ang), si = Kokkos::sin(ang);
                        const push::V3 s1 = ca * s0 + si * push::cross(n, s0) + ((Real(1) - ca) * push::dot(n, s0)) * n;
                        sx(i) = s1.x; sy(i) = s1.y; sz(i) = s1.z;
                    }
                }
            }
        }
        const Real p2 = u.x * u.x + u.y * u.y;
        const Real gam = Kokkos::sqrt(Real(1) + p2 + u.z * u.z);
        const Real omvz = (Real(1) + p2) / (gam * (gam + u.z));
        const Real xn = X + u.x / gam * dt, yn = Y + u.y / gam * dt;
        x(i) = xn; y(i) = yn;
        px(i) = u.x; py(i) = u.y; pz(i) = u.z;
        xi(i) += omvz * dt;
        if (xn * xn + yn * yn >= R * R || xi(i) < xmin || xi(i) > xmax) w(i) = Real(0);   // left the box
    });
}

std::array<double, 17> Beam::sums() const {
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), x_);
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), y_);
    auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), px_);
    auto hpy = Kokkos::create_mirror_view_and_copy(HostSpace(), py_);
    auto hpz = Kokkos::create_mirror_view_and_copy(HostSpace(), pz_);
    auto hxi = Kokkos::create_mirror_view_and_copy(HostSpace(), xi_);
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), w_);
    std::array<double, 17> S{};   // S[0] = sum w, S[1..15] = sum w v, S[16] = number of live particles
    for (int i = 0; i < Np_; ++i) {
        const double w = hw(i);
        if (w == 0) continue;
        S[16] += 1;
        const double g = std::sqrt(1 + hpx(i) * hpx(i) + hpy(i) * hpy(i) + hpz(i) * hpz(i));
        const double X = hx(i), Y = hy(i), PX = hpx(i), PY = hpy(i);
        const double G = g - gref_, XI = hxi(i) - xiref_;
        const double v[15] = {G, G * G, XI, XI * XI, X * X + Y * Y, X, Y, X * X, Y * Y,
                              PX, PY, PX * PX, PY * PY, X * PX, Y * PY};
        S[0] += w;
        for (int c = 0; c < 15; ++c) S[c + 1] += w * v[c];
    }
    return S;
}

BeamDiag Beam::from_sums(const std::array<double, 17>& S, Real t) const {
    const Real q = q_;
    BeamDiag d;
    d.t = t;
    d.alive = static_cast<long>(S[16]);
    const double sw = S[0];
    if (sw > 0) {
        auto m = [&](int c) { return S[c + 1] / sw; };
        d.npart = sw;
        d.charge = q * sw;
        d.gamma_mean = gref_ + m(0);
        d.gamma_rms = std::sqrt(std::max(0.0, m(1) - m(0) * m(0)));
        d.xi_mean = xiref_ + m(2);
        d.xi_rms = std::sqrt(std::max(0.0, m(3) - m(2) * m(2)));
        d.r_rms = std::sqrt(m(4));
        d.x_mean = m(5);
        d.y_mean = m(6);
        double xx = m(7) - m(5) * m(5), yy = m(8) - m(6) * m(6);
        double pxx = m(11) - m(9) * m(9), pyy = m(12) - m(10) * m(10);
        double xpx = m(13) - m(5) * m(9), ypy = m(14) - m(6) * m(10);
        if (!m1_) {
            // m = 0: a macro-particle stands for a ring; its (x, y) is one point of it (the parsed loader puts all
            // of them at y = 0). The moments of the axisymmetric beam are the averages over the azimuth:
            // <x^2> = <r^2>/2, <p_x^2> = <p_perp^2>/2, <x p_x> = <r . p_perp>/2, means 0.
            d.x_mean = d.y_mean = 0;
            xx = yy = 0.5 * m(4);
            pxx = pyy = 0.5 * (m(11) + m(12));
            xpx = ypy = 0.5 * (m(13) + m(14));
        }
        d.sigma_x = std::sqrt(std::max(0.0, xx));
        d.sigma_y = std::sqrt(std::max(0.0, yy));
        d.emit_nx = std::sqrt(std::max(0.0, xx * pxx - xpx * xpx));
        d.emit_ny = std::sqrt(std::max(0.0, yy * pyy - ypy * ypy));
    }
    return d;
}

BeamDiag Beam::diagnostics(Real t) const { return from_sums(sums(), t); }

// per-bin sums over fixed bins [lo, hi): 9 numbers per bin
//   w, w x, w y, w x^2, w y^2, w gamma, w px, w px^2, w x px
std::vector<double> Beam::slice_sums(double lo, double hi, int nbins) const {
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), x_);
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), y_);
    auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), px_);
    auto hpy = Kokkos::create_mirror_view_and_copy(HostSpace(), py_);
    auto hpz = Kokkos::create_mirror_view_and_copy(HostSpace(), pz_);
    auto hxi = Kokkos::create_mirror_view_and_copy(HostSpace(), xi_);
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), w_);
    std::vector<double> B(9 * static_cast<size_t>(nbins), 0.0);
    const double db = (hi - lo) / nbins;
    for (int i = 0; i < Np_; ++i) {
        const double w = hw(i);
        if (w == 0) continue;
        const int b = static_cast<int>(std::floor((hxi(i) - lo) / db));
        if (b < 0 || b >= nbins) continue;
        const double g = std::sqrt(1 + hpx(i) * hpx(i) + hpy(i) * hpy(i) + hpz(i) * hpz(i));
        const double v[9] = {1.0, hx(i), hy(i), hx(i) * hx(i), hy(i) * hy(i), g, hpx(i), hpx(i) * hpx(i), hx(i) * hpx(i)};
        for (int c = 0; c < 9; ++c) B[9 * b + c] += w * v[c];
    }
    return B;
}

void Beam::write_slices_file(const std::string& filename, Real t, Real q, double lo, double hi, int nbins,
                             const std::vector<double>& B) {
    std::ofstream out(filename);
    out << "# t = " << t << "\n# xi  charge  x_mean  y_mean  sigma_x  sigma_y  gamma_mean  emit_nx\n";
    out << std::setprecision(8);
    const double db = (hi - lo) / nbins;
    for (int b = 0; b < nbins; ++b) {
        const double sw = B[9 * b];
        if (sw <= 0) continue;
        auto m = [&](int c) { return B[9 * b + c] / sw; };
        const double xx = m(3) - m(1) * m(1), yy = m(4) - m(2) * m(2);
        const double pxx = m(7) - m(6) * m(6), xpx = m(8) - m(1) * m(6);
        out << lo + (b + 0.5) * db << " " << q * sw << " " << m(1) << " " << m(2) << " "
            << std::sqrt(std::max(0.0, xx)) << " " << std::sqrt(std::max(0.0, yy)) << " " << m(5) << " "
            << std::sqrt(std::max(0.0, xx * pxx - xpx * xpx)) << "\n";
    }
}

// live particles as 7 doubles each: x y px py pz xi w
std::vector<double> Beam::packed(const Kokkos::View<int*, HostSpace>* sel) const {
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), x_);
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), y_);
    auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), px_);
    auto hpy = Kokkos::create_mirror_view_and_copy(HostSpace(), py_);
    auto hpz = Kokkos::create_mirror_view_and_copy(HostSpace(), pz_);
    auto hxi = Kokkos::create_mirror_view_and_copy(HostSpace(), xi_);
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), w_);
    Kokkos::View<double*, HostSpace> hs[3];
    if (spin_) { hs[0] = Kokkos::create_mirror_view_and_copy(HostSpace(), sx_);
                 hs[1] = Kokkos::create_mirror_view_and_copy(HostSpace(), sy_);
                 hs[2] = Kokkos::create_mirror_view_and_copy(HostSpace(), sz_); }
    std::vector<double> out;
    for (int i = 0; i < Np_; ++i) {
        if (hw(i) == 0) continue;
        if (sel && !(*sel)(i)) continue;
        out.insert(out.end(), {hx(i), hy(i), hpx(i), hpy(i), hpz(i), hxi(i), hw(i)});
        if (spin_) out.insert(out.end(), {hs[0](i), hs[1](i), hs[2](i)});
    }
    return out;
}

std::vector<View1D*> Beam::comps() {
    std::vector<View1D*> v = {&x_, &y_, &px_, &py_, &pz_, &xi_, &w_};
    if (spin_) { v.push_back(&sx_); v.push_back(&sy_); v.push_back(&sz_); }
    return v;
}
std::vector<const View1D*> Beam::comps() const {
    std::vector<const View1D*> v = {&x_, &y_, &px_, &py_, &pz_, &xi_, &w_};
    if (spin_) { v.push_back(&sx_); v.push_back(&sy_); v.push_back(&sz_); }
    return v;
}

std::array<double, 4> Beam::spin_sums() const {
    std::array<double, 4> S{0, 0, 0, 0};
    if (!spin_ || Np_ == 0) return S;
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), w_);
    auto a = Kokkos::create_mirror_view_and_copy(HostSpace(), sx_);
    auto b = Kokkos::create_mirror_view_and_copy(HostSpace(), sy_);
    auto c = Kokkos::create_mirror_view_and_copy(HostSpace(), sz_);
    for (int i = 0; i < Np_; ++i) {
        const double w = hw(i);
        if (w == 0) continue;
        S[0] += w; S[1] += w * a(i); S[2] += w * b(i); S[3] += w * c(i);
    }
    return S;
}

std::vector<double> Beam::packed_all() const {
    const auto v = comps();
    const size_t nc = static_cast<size_t>(nc_);
    std::vector<double> out(nc * static_cast<size_t>(Np_));
    for (size_t c = 0; c < nc; ++c) {
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), *v[c]);
        for (int i = 0; i < Np_; ++i) out[nc * static_cast<size_t>(i) + c] = h(i);
    }
    return out;
}

void Beam::dump(const std::string& filename, int stride) const {
    std::vector<double> p = packed();
    if (stride > 1) {
        std::vector<double> q;
        const size_t nc = static_cast<size_t>(nc_);
        for (size_t i = 0; i < p.size() / nc; i += static_cast<size_t>(stride))
            q.insert(q.end(), p.begin() + nc * i, p.begin() + nc * i + nc);
        p.swap(q);
    }
    // int32 n, then n records of 7 doubles (10 with spin: + s_x s_y s_z; readers infer it from the size)
    std::ofstream out(filename, std::ios::binary);
    const int32_t n = static_cast<int32_t>(p.size() / static_cast<size_t>(nc_));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(p.data()), sizeof(double) * p.size());
}

// replace the particle arrays by the packed list (host -> device)
void Beam::set_particles(const std::vector<double>& p) {
    const size_t nc = static_cast<size_t>(nc_);
    if (p.size() % nc != 0) throw std::runtime_error(name_ + ": particle data with a wrong number of components (spin?)");
    Np_ = static_cast<int>(p.size() / nc);
    const auto views = comps();
    const char* nm[10] = {".x", ".y", ".px", ".py", ".pz", ".xi", ".w", ".sx", ".sy", ".sz"};
    for (size_t c = 0; c < nc; ++c) {
        *views[c] = View1D(name_ + nm[c], Np_);
        auto h = Kokkos::create_mirror_view(*views[c]);
        for (int i = 0; i < Np_; ++i) h(i) = p[nc * static_cast<size_t>(i) + c];
        Kokkos::deep_copy(*views[c], h);
    }
}

int Beam::slice_of(double xi) const {
    return static_cast<int>(std::floor((xi - bg_.xi_min) / bg_.dxi + 0.5));
}

// keep only the particles whose (nearest) slice belongs to this rank
void Beam::restrict_to_local() {
    if (analytic_) return;
    const std::vector<double> all = packed();
    std::vector<double> keep;
    // particles outside the box stay with the first / last rank (as in a serial run)
    const int lo = bg_.k0 == 0 ? std::numeric_limits<int>::min() : bg_.k0;
    const int hi = bg_.k0 + bg_.nloc == bg_.nxi ? std::numeric_limits<int>::max() : bg_.k0 + bg_.nloc;
    const size_t nc = static_cast<size_t>(nc_);
    for (size_t i = 0; i < all.size(); i += nc) {
        const int k = slice_of(all[i + 5]);
        if (k >= lo && k < hi) keep.insert(keep.end(), all.begin() + i, all.begin() + i + nc);
    }
    set_particles(keep);
}

// particles that slipped behind the local slices (xi grows: dxi/dt = 1 - v_z >= 0) are removed
// and returned for the downstream rank; dead particles are dropped at the same time
std::vector<double> Beam::extract_outgoing() {
    if (analytic_ || Np_ == 0) return {};
    auto xi = xi_; auto w = w_;
    const Real xmin = bg_.xi_min, idxi = Real(1) / bg_.dxi;
    const int kend = bg_.k0 + bg_.nloc;
    int nout = 0, ndead = 0;
    Kokkos::parallel_reduce("beam.count_out", Range(0, Np_), KOKKOS_LAMBDA(int i, int& a, int& d) {
        if (w(i) == Real(0)) { ++d; return; }
        if (static_cast<int>(Kokkos::floor((xi(i) - xmin) * idxi + Real(0.5))) >= kend) ++a;
    }, nout, ndead);
    if (nout == 0 && ndead < Np_ / 4 + 1) return {};     // nothing to send; compact only occasionally
    const std::vector<double> all = packed();
    std::vector<double> keep, out;
    const size_t nc = static_cast<size_t>(nc_);
    for (size_t i = 0; i < all.size(); i += nc) {
        auto& dst = (slice_of(all[i + 5]) >= kend) ? out : keep;
        dst.insert(dst.end(), all.begin() + i, all.begin() + i + nc);
    }
    set_particles(keep);
    return out;
}

void Beam::append(const double* p, size_t n) {
    if (n == 0) return;
    std::vector<double> all = packed();
    all.insert(all.end(), p, p + static_cast<size_t>(nc_) * n);
    set_particles(all);
}

} // namespace quarz
