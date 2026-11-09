#include "Laser.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>

namespace quarz {

namespace {
struct Cx { Real re, im; };
KOKKOS_INLINE_FUNCTION Cx cadd(Cx a, Cx b) { return {a.re + b.re, a.im + b.im}; }
KOKKOS_INLINE_FUNCTION Cx csub(Cx a, Cx b) { return {a.re - b.re, a.im - b.im}; }
KOKKOS_INLINE_FUNCTION Cx cmul(Cx a, Cx b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
KOKKOS_INLINE_FUNCTION Cx cscale(Real s, Cx a) { return {s * a.re, s * a.im}; }
KOKKOS_INLINE_FUNCTION Cx cdiv(Cx a, Cx b) {
    const Real d = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}
} // namespace

Laser::Laser(const Config& cfg, const RadialGrid& grid, const FieldSolver& solver, const BeamGrid& box, Real t0, Real dt)
    : grid_(grid), box_(box), dt_(dt), solver_(solver) {
    const int M = grid.N + 1, KL = box.nloc;
    // ---- carrier: laser.k0 = omega0/omega_p, or laser.lambda0_um with units.n0_cm3
    if (cfg.has("laser.k0")) {
        k0_ = cfg.get_double("laser.k0");
    } else if (cfg.has("laser.lambda0_um")) {
        const double n0 = cfg.get_double("units.n0_cm3", 0.0);
        if (n0 <= 0) throw std::runtime_error("laser.lambda0_um needs the plasma density units.n0_cm3");
        const double lambda_p_um = 2 * PI * 299792458.0 /
                                   std::sqrt(n0 * 1e6 * 1.602176634e-19 * 1.602176634e-19 / (8.8541878128e-12 * 9.1093837015e-31)) * 1e6;
        k0_ = lambda_p_um / cfg.get_double("laser.lambda0_um");
    } else {
        throw std::runtime_error("laser: give laser.k0 (= omega0/omega_p) or laser.lambda0_um and units.n0_cm3");
    }
    if (k0_ <= 1) throw std::runtime_error("laser: k0 = omega0/omega_p must be > 1 (underdense plasma)");
    const std::string pol = cfg.get_string("laser.polarization", "linear");
    if (pol != "linear" && pol != "circular") throw std::runtime_error("laser.polarization must be linear or circular");
    circular_ = (pol == "circular");
    // xi discretisation of the march (study option): trapezoid (box scheme, neutral) or backward
    // differences towards the head of the box (BDF2 as in QuickPIC-FD, or backward Euler): A-stable
    // march, dissipative in t (|g| < 1 except at kappa = 0).
    const std::string xs = cfg.get_string("laser.xi_scheme", "trapezoid");
    if (xs == "trapezoid") xi_scheme_ = 0;
    else if (xs == "bdf2") xi_scheme_ = 1;
    else if (xs == "euler") xi_scheme_ = 2;
    else throw std::runtime_error("laser.xi_scheme must be trapezoid, bdf2 or euler");

    // ---- Gaussian pulse:  a^ = a0 / (1 + i s) exp(-r^2 / (w0^2 (1 + i s))) exp(-(xi - xi0)^2 / L0^2),
    //      s = (t - t_focus)/Z_R,  Z_R = k0 w0^2 / 2;  laser.focus = distance to the focal plane
    const double a0 = cfg.get_double("laser.a0");
    const double w0 = cfg.get_double("laser.w0");
    const double L0 = cfg.get_double("laser.L0");
    const double xi0 = cfg.get_double("laser.xi0");
    const double zf = cfg.get_double("laser.focus", 0.0);
    if (w0 <= 0 || L0 <= 0) throw std::runtime_error("laser: w0 and L0 must be positive");
    const double zR = 0.5 * k0_ * w0 * w0;
    const double s = -zf / zR;
    (void)t0;
    // Laguerre-Gaussian vortex LG_0l (laser.l = l >= 1): a = Re[a^(r) e^{i l theta} e^{-i k0 xi}], so |a^|^2 and the
    // plasma response stay axisymmetric and the envelope obeys the same equation with grad_perp^2 -> L_l = L_0 - l^2/r^2.
    //   a^ = a0 N_l (sqrt(2) r / w0)^l / (1 + i s)^(l+1) exp(-r^2 / (w0^2 (1 + i s))) exp(-(xi - xi0)^2 / L0^2),
    // N_l = (e / l)^(l/2): laser.a0 is the peak of |a^| over r at the focus (at r = w0 sqrt(l/2)).
    l_ = cfg.get_int("laser.l", 0);
    if (l_ < 0) throw std::runtime_error("laser.l must be >= 0 (the sense of rotation does not change |a|^2)");
    if (cfg.get_int("laser.p", 0) != 0) throw std::runtime_error("laser.p: only radial index p = 0 is implemented");
    const double lnorm = l_ > 0 ? std::pow(std::exp(1.0) / l_, 0.5 * l_) : 1.0;

    a_ = View3D("laser.a", KL, M, 2);
    an_ = View3D("laser.an", KL, M, 2);
    pond_ = View3D("laser.pond", KL, M, 3);
    guard_ = View3D("laser.guard", 4, M, 2);
    carry_ = View3D("laser.carry", 2, M, 2);
    aa_tmp_ = View1D("laser.aa", M);
    cpr_ = View1D("laser.cpr", M); cpi_ = View1D("laser.cpi", M);
    dpr_ = View1D("laser.dpr", M); dpi_ = View1D("laser.dpi", M);
    solver.ln_coefficients(l_, la_, lb_, lc_);

    auto h = Kokkos::create_mirror_view(a_);
    for (int kl = 0; kl < KL; ++kl) {
        const double xi = box.xi_min + (box.k0 + kl) * box.dxi;
        const double lon = std::exp(-(xi - xi0) * (xi - xi0) / (L0 * L0));
        for (int j = 0; j < M; ++j) {
            const double r = grid.r[j];
            // 1/(1+is) and exp(-r^2/(w0^2 (1+is))) = exp(-r^2 (1 - i s)/(w0^2 (1+s^2)))
            const double den = 1 + s * s;
            const double pre_re = 1 / den, pre_im = -s / den;
            const double ex = std::exp(-r * r / (w0 * w0 * den)), ph = r * r * s / (w0 * w0 * den);
            const double e_re = ex * std::cos(ph), e_im = ex * std::sin(ph);
            const double amp = (j == M - 1) ? 0.0 : a0 * lon;   // a^ = 0 at the wall
            h(kl, j, 0) = amp * (pre_re * e_re - pre_im * e_im);
            h(kl, j, 1) = amp * (pre_re * e_im + pre_im * e_re);
            if (l_ > 0) {   // LG_0l: times N_l (sqrt(2) r / w0)^l / (1 + i s)^l  (Gouy phase (l + 1) atan s)
                const std::complex<double> f = lnorm * std::pow(std::sqrt(2.0) * r / w0, l_) *
                                               std::pow(std::complex<double>(1.0, s), -l_) *
                                               std::complex<double>(h(kl, j, 0), h(kl, j, 1));
                h(kl, j, 0) = (j == 0) ? 0.0 : f.real();
                h(kl, j, 1) = (j == 0) ? 0.0 : f.imag();
            }
        }
    }
    Kokkos::deep_copy(a_, h);
    Kokkos::deep_copy(an_, a_);
    // the head of the box must be free of light
    if (box.k0 == 0) {
        const double xh = box.xi_min;
        if (std::exp(-(xh - xi0) * (xh - xi0) / (L0 * L0)) > 1e-2)
            std::fprintf(stderr, "WARNING: the laser is not negligible at the head of the box (xi = %g): |a| > 1%% of a0 there\n", xh);
    }
}

std::string Laser::description() const {
    char b[320];
    static const char* xsn[] = {"Crank-Nicolson in t, trapezoid in xi", "Crank-Nicolson in t, BDF2 in xi",
                                "Crank-Nicolson in t, backward Euler in xi"};
    char m[64];
    if (l_ > 0) std::snprintf(m, sizeof(m), "LG_0%d vortex, a^ e^{i %d theta}", l_, l_);
    else std::snprintf(m, sizeof(m), "m = 0 envelope");
    std::snprintf(b, sizeof(b), "Laser envelope: k0 = omega0/omega_p = %g, %s polarisation (%s, %s)",
                  k0_, circular_ ? "circular" : "linear", m, xsn[xi_scheme_]);
    return b;
}

void Laser::begin_step() {
    if (advanced_) std::swap(a_, an_);
    advanced_ = false;
}

void Laser::set_guard(const double* p) {
    auto h = Kokkos::create_mirror_view(guard_);
    const int M = grid_.N + 1;
    size_t o = 0;
    for (int l = 0; l < 4; ++l)
        for (int j = 0; j < M; ++j)
            for (int c = 0; c < 2; ++c) h(l, j, c) = p[o++];
    Kokkos::deep_copy(guard_, h);
}

void Laser::get_guard(std::vector<double>& buf) const {
    // [old KL-2, old KL-1, X(KL-1), G(KL-1)] of this rank = guard of the next rank
    auto ha = Kokkos::create_mirror_view_and_copy(HostSpace(), a_);
    auto hc = Kokkos::create_mirror_view_and_copy(HostSpace(), carry_);
    const int M = grid_.N + 1, KL = box_.nloc;
    for (int d = 2; d >= 1; --d)
        for (int j = 0; j < M; ++j)
            for (int c = 0; c < 2; ++c) buf.push_back(ha(KL - d, j, c));
    for (int l = 0; l < 2; ++l)
        for (int j = 0; j < M; ++j)
            for (int c = 0; c < 2; ++c) buf.push_back(hc(l, j, c));
}

void Laser::prepare_slice(int kl, SliceFields& f) {
    const int M = grid_.N + 1;
    auto a = a_; auto g = guard_; auto pond = pond_;
    auto aa = f.aa;
    const Real fac = circular_ ? Real(1) : Real(0.5);
    const Real idxi = Real(1) / box_.dxi;
    const int kglob = box_.k0 + kl;
    Kokkos::parallel_for("laser.aa", Range(0, M), KOKKOS_LAMBDA(int j) {
        auto mag = [&](int l) -> Real {   // |a^|^2 at local slice l (may be -1, -2: guard), old level
            if (l >= 0) return a(l, j, 0) * a(l, j, 0) + a(l, j, 1) * a(l, j, 1);
            const int gi = l + 2;   // -2 -> 0, -1 -> 1
            return g(gi, j, 0) * g(gi, j, 0) + g(gi, j, 1) * g(gi, j, 1);
        };
        const Real m0 = fac * mag(kl);
        aa(j) = m0;
        pond(kl, j, 0) = m0;
        const Real m1 = kglob >= 1 ? fac * mag(kl - 1) : Real(0);
        const Real m2 = kglob >= 2 ? fac * mag(kl - 2) : Real(0);
        pond(kl, j, 2) = (Real(3) * m0 - Real(4) * m1 + m2) * Real(0.5) * idxi;
    });
    solver_.gradient(f.aa, f.daa);
    auto daa = f.daa;
    Kokkos::parallel_for("laser.daa", Range(0, M), KOKKOS_LAMBDA(int j) { pond(kl, j, 1) = daa(j); });
}

void Laser::advance_slice(int kl, const View1D& chi0) {
    if (!(dt_ > 0)) return;
    // Unknown X = a^(n+1) - a^n.  Crank-Nicolson in t of  2 d/dt (i k0 - d/dxi) a^ = -H a^:
    //     2 (i k0 - d/dxi) X / dt = -(1/2) H (X + 2 a^n),   H = L0 - chi,
    // i.e. the ODE in xi   dX/dxi = G,   G = i k0 X + (dt/4) H (X + 2 a^n),
    // integrated with the trapezoidal rule from slice k-1 to k (box scheme: second order and
    // neutrally stable in xi and t). laser.xi_scheme = bdf2 | euler: backward differences instead (study option).
    // The carry (X, G) of the previous slice comes from the guard for the first local slice.
    const int N = grid_.N;
    auto a = a_; auto an = an_; auto g = guard_; auto carry = carry_;
    auto la = la_; auto lb = lb_; auto lc = lc_;
    auto chi = chi0;
    auto cpr = cpr_, cpi = cpi_, dpr = dpr_, dpi = dpi_;
    const Real dxi = box_.dxi, dt = dt_, k0 = k0_;
    const bool first = (kl == 0);
    const int scheme = xi_scheme_;
    const bool axis0 = (l_ > 0);
    Kokkos::parallel_for("laser.advance", Range(0, 1), KOKKOS_LAMBDA(int) {
        // trapezoid: carry = (X_{k-1}, G_{k-1});  bdf2 / euler: carry = (X_{k-1}, X_{k-2})
        auto Xp = [&](int j) -> Cx { return first ? Cx{g(2, j, 0), g(2, j, 1)} : Cx{carry(0, j, 0), carry(0, j, 1)}; };
        auto Gp = [&](int j) -> Cx { return first ? Cx{g(3, j, 0), g(3, j, 1)} : Cx{carry(1, j, 0), carry(1, j, 1)}; };
        auto A = [&](int j) -> Cx { return Cx{a(kl, j, 0), a(kl, j, 1)}; };
        // d/dxi X_k = (cd X_k - c1 X_{k-1} - c2 X_{k-2}) / (2 dxi) [trapezoid: 2 (X_k - X_{k-1}) / dxi - G_{k-1}];
        // times 4/dt:  [cd 2/(dt dxi) - 4 i k0/dt - H] X_k = 2/(dt dxi) (c1 X_{k-1} + c2 X_{k-2}) (+ 4/dt G_{k-1}) + 2 H a^n
        const Real cd = scheme == 0 ? Real(4) : (scheme == 1 ? Real(3) : Real(2));
        const Real c1 = scheme == 0 ? Real(4) : (scheme == 1 ? Real(4) : Real(2));
        const Real c2 = scheme == 1 ? Real(-1) : Real(0);
        const Real s0 = Real(2) / (dt * dxi);
        const Real s1 = cd * s0, s2 = Real(4) / dt;
        const Cx alpha{s1, -s2 * k0};   // trapezoid: 8/(dt dxi) - 4 i k0 / dt
        Cx dprev{0, 0}, cprev{0, 0};
        for (int j = 0; j < N; ++j) {
            if (axis0 && j == 0) {   // vortex (l >= 1): a^ = 0 on the axis
                cpr(0) = 0; cpi(0) = 0; dpr(0) = 0; dpi(0) = 0;
                continue;   // cprev = dprev = 0
            }
            const Real aj = j > 0 ? la(j) : Real(0), bj = lb(j), cj = lc(j);
            // H a^n at node j
            Cx Ha = cscale(bj - chi(j), A(j));
            if (j > 0) Ha = cadd(Ha, cscale(aj, A(j - 1)));
            Ha = cadd(Ha, cscale(cj, A(j + 1)));
            // [8/(dt dxi) - 4 i k0/dt - H] X_k = 8/(dt dxi) X_{k-1} + 4/dt G_{k-1} + 2 H a^n_k
            const Cx hist = scheme == 0 ? cadd(cscale(c1 * s0, Xp(j)), cscale(s2, Gp(j)))
                                        : cadd(cscale(c1 * s0, Xp(j)), cscale(c2 * s0, Gp(j)));
            const Cx rhs = cadd(hist, cscale(Real(2), Ha));
            const Cx diag = csub(alpha, Cx{bj - chi(j), Real(0)});
            const Real low = -aj, up = -cj;
            const Cx m = csub(diag, cscale(low, cprev));
            const Cx cpj = cdiv(Cx{up, Real(0)}, m);
            const Cx dpj = cdiv(csub(rhs, cscale(low, dprev)), m);
            cpr(j) = cpj.re; cpi(j) = cpj.im; dpr(j) = dpj.re; dpi(j) = dpj.im;
            cprev = cpj; dprev = dpj;
        }
        // back substitution: X into an (temporarily), X_N = 0
        an(kl, N, 0) = Real(0); an(kl, N, 1) = Real(0);
        Cx xn{0, 0};
        for (int j = N - 1; j >= 0; --j) {
            const Cx x = csub(Cx{dpr(j), dpi(j)}, cmul(Cx{cpr(j), cpi(j)}, xn));
            an(kl, j, 0) = x.re; an(kl, j, 1) = x.im;
            xn = x;
        }
        // carry for the next slice: X and G = i k0 X + (dt/4) H (X + 2 a^n); then a^(n+1) = a^n + X
        auto Y = [&](int j) -> Cx { return Cx{an(kl, j, 0) + Real(2) * a(kl, j, 0), an(kl, j, 1) + Real(2) * a(kl, j, 1)}; };
        if (scheme != 0) {   // bdf2 / euler: carry = (X_k, X_{k-1})
            for (int j = 0; j <= N; ++j) {
                const Cx xp = Xp(j);
                carry(1, j, 0) = xp.re; carry(1, j, 1) = xp.im;
                carry(0, j, 0) = an(kl, j, 0); carry(0, j, 1) = an(kl, j, 1);
            }
        }
        for (int j = 0; j <= N && scheme == 0; ++j) {
            const Cx X{an(kl, j, 0), an(kl, j, 1)};
            Cx G{0, 0};
            if (j < N) {
                const Real aj = j > 0 ? la(j) : Real(0), bj = lb(j), cj = lc(j);
                Cx HY = cscale(bj - chi(j), Y(j));
                if (j > 0) HY = cadd(HY, cscale(aj, Y(j - 1)));
                HY = cadd(HY, cscale(cj, Y(j + 1)));
                G = cadd(cmul(Cx{Real(0), k0}, X), cscale(Real(0.25) * dt, HY));
            }
            carry(0, j, 0) = X.re; carry(0, j, 1) = X.im;
            carry(1, j, 0) = G.re; carry(1, j, 1) = G.im;
        }
        for (int j = 0; j <= N; ++j) {
            an(kl, j, 0) += a(kl, j, 0);
            an(kl, j, 1) += a(kl, j, 1);
        }
    });
}

std::array<double, 5> Laser::sums() const {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), a_);
    std::array<double, 5> S{0, 0, 0, 0, 0};
    const int M = grid_.N + 1, KL = box_.nloc;
    for (int kl = 0; kl < KL; ++kl) {
        const double xi = box_.xi_min + (box_.k0 + kl) * box_.dxi;
        for (int j = 0; j < M; ++j) {
            const double m2 = h(kl, j, 0) * h(kl, j, 0) + h(kl, j, 1) * h(kl, j, 1);
            const double dv = grid_.V[j] * box_.dxi;
            S[0] += m2 * dv;
            S[1] += m2 * xi * dv;
            S[2] += m2 * grid_.r[j] * grid_.r[j] * dv;
            S[3] = std::max(S[3], std::sqrt(m2));
            S[4] += m2 * xi * xi * dv;
        }
    }
    return S;
}

} // namespace quarz
