// BeamPush.cpp: beam deposition, impact ionization, push and particle exchange (part of the Beam class)
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

void Beam::deposit_impact(const View3D& imp) const {
    if (passive_) return;
    const GridD g = grid_.d;
    const Real z2 = q_ * q_;
    if (analytic_) {   // rigid analytic beam: m = 0 density at every node
        const int K = bg_.nloc, M = grid_.N + 1;
        const Real xmin = bg_.xi_min, dxi = bg_.dxi;
        const int k0 = bg_.k0;
        const Real gam = an_gamma_;
        const Real b2 = Real(1) - Real(1) / (gam * gam), be = std::sqrt(b2);
        const Real L = std::log(b2 * gam * gam) - b2;
        const Real c0 = z2 * be, c1 = z2 * L / be, c2 = z2 / be;
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

} // namespace quarz
