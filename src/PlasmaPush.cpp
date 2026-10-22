// PlasmaPush.cpp: plasma deposition and push (part of the PlasmaSpecies class)
#include "PlasmaSpecies.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include "PlasmaKernels.hpp"

namespace quarz {

using namespace detail;

void PlasmaSpecies::deposit(const SliceSources& s, const SliceFields& f) const {
    const GridD g = grid_.d;
    const bool laser = f.laser;
    auto aa = f.aa;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    auto lev = lev_;
    const bool ionz = ionizable_;
    const Real q0 = q_, qm0 = q_ / m_;
    const bool neg = q_ < 0;
    const bool m1 = m1_;
    const Real chif = frozen_ ? Real(0) : Real(1);   // immobile ions do not respond: no chi
    Kokkos::parallel_for("plasma.deposit", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real zi = ionz ? lev(i) : Real(1);
        if (zi == Real(0)) return;   // neutral atom
        const Real q = q0 * zi, qm = qm0 * zi;
        const Real PX = px(i), PY = py(i), D = dl(i);
        const PGeo p = geometry(g, x(i), y(i));
        const Real A = laser ? qm * qm * lin(aa, p) : Real(0);   // ponderomotive: qhat^2 <a^2>
        const Real gam = (Real(1) + PX * PX + PY * PY + A + D * D) / (Real(2) * D);
        const Real invd = Real(1) / D;
        const Real pz = gam - D;
        const Real ux = PX * invd, uy = PY * invd;
        const Real ur = p.c * ux + p.s * uy, uth = p.c * uy - p.s * ux;
        const Real wq = wi * q;
        const MScalar& dens = neg ? s.ne : s.ni;
        for (int cc = 0; cc < 2; ++cc) {
            const int jj = p.j + cc;
            const Real iv = g.inv_V(jj);
            const Real a = wq * p.W[cc] * iv;
            const Real dr = wq * ur * p.dW[cc] * iv;         // d/dxi of the radial weight
            const Real vjz = a * pz * invd, vchi = chif * a * qm * invd, vrho = a * gam * invd;
            const Real vn = wi * p.W[cc] * iv * gam * invd;
            Kokkos::atomic_add(&s.rhot.a0(jj), a);
            Kokkos::atomic_add(&s.drho.a0(jj), dr);
            Kokkos::atomic_add(&s.jz.a0(jj), vjz);
            Kokkos::atomic_add(&s.chi.a0(jj), vchi);
            Kokkos::atomic_add(&s.rho.a0(jj), vrho);
            Kokkos::atomic_add(&dens.a0(jj), vn);
            // J+ mode 1:  u+ e^{-i th} = u_r + i u_th
            Kokkos::atomic_add(&s.jp.r1(jj), a * ur);
            Kokkos::atomic_add(&s.jp.i1(jj), a * uth);
            if (m1) {
                const Real c = p.c, sn = p.s;
                // scalar mode 1: < f e^{-i th} >
                Kokkos::atomic_add(&s.rhot.ar(jj), a * c);
                Kokkos::atomic_add(&s.rhot.ai(jj), -a * sn);
                // d/dxi [W e^{-i th}] = W' u_r e^{-i th} - i (u_th/r) W e^{-i th}
                const Real th = wq * uth * p.Wor[cc] * iv;
                Kokkos::atomic_add(&s.drho.ar(jj), dr * c - th * sn);
                Kokkos::atomic_add(&s.drho.ai(jj), -dr * sn - th * c);
                Kokkos::atomic_add(&s.jz.ar(jj), vjz * c);
                Kokkos::atomic_add(&s.jz.ai(jj), -vjz * sn);
                Kokkos::atomic_add(&s.chi.ar(jj), vchi * c);
                Kokkos::atomic_add(&s.chi.ai(jj), -vchi * sn);
                Kokkos::atomic_add(&s.rho.ar(jj), vrho * c);
                Kokkos::atomic_add(&s.rho.ai(jj), -vrho * sn);
                Kokkos::atomic_add(&dens.ar(jj), vn * c);
                Kokkos::atomic_add(&dens.ai(jj), -vn * sn);
                // J+ mode 0: u+ ;  mode 2: u+ e^{-2i th}
                Kokkos::atomic_add(&s.jp.r0(jj), a * ux);
                Kokkos::atomic_add(&s.jp.i0(jj), a * uy);
                const Real c2 = c * c - sn * sn, s2 = Real(2) * c * sn;
                Kokkos::atomic_add(&s.jp.r2(jj), a * (ux * c2 + uy * s2));
                Kokkos::atomic_add(&s.jp.i2(jj), a * (uy * c2 - ux * s2));
            }
        }
    });
}

void PlasmaSpecies::deposit_rho(const MScalar& rhot, const MScalar& rho) const {
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto w = w_;
    auto lev = lev_;
    const bool ionz = ionizable_;
    const Real q0 = q_;
    const bool m1 = m1_;
    Kokkos::parallel_for("plasma.deposit_rho", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real q = q0 * (ionz ? lev(i) : Real(1));
        if (q == Real(0)) return;
        const PGeo p = geometry(g, x(i), y(i));
        for (int cc = 0; cc < 2; ++cc) {
            const int jj = p.j + cc;
            const Real a = wi * q * p.W[cc] * g.inv_V(jj);
            Kokkos::atomic_add(&rhot.a0(jj), a);
            Kokkos::atomic_add(&rho.a0(jj), a);
            if (m1) {
                Kokkos::atomic_add(&rhot.ar(jj), a * p.c);
                Kokkos::atomic_add(&rhot.ai(jj), -a * p.s);
                Kokkos::atomic_add(&rho.ar(jj), a * p.c);
                Kokkos::atomic_add(&rho.ai(jj), -a * p.s);
            }
        }
    });
}

void PlasmaSpecies::deposit_S(const SliceFields& f, const MVector& S) const {
    if (frozen_) return;   // J_perp = 0 and stays 0
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    auto lev = lev_;
    const bool ionz = ionizable_;
    const Real q0 = q_, qm0 = q_ / m_;
    const bool m1 = m1_;
    Kokkos::parallel_for("plasma.deposit_S", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real zi = ionz ? lev(i) : Real(1);
        if (zi == Real(0)) return;
        const Real q = q0 * zi, qm = qm0 * zi;
        const Real PX = px(i), PY = py(i), D = dl(i);
        const PGeo p = geometry(g, x(i), y(i));
        const PFields F = gather(f, p, m1, false);
        const Real gam = (Real(1) + PX * PX + PY * PY + qm * qm * F.A + D * D) / (Real(2) * D);
        const Real invd = Real(1) / D;
        const Real fp = -Real(0.5) * qm * qm * F.dA * invd;   // ponderomotive dp_perp/dxi (radial)
        const Real ux = PX * invd, uy = PY * invd;
        const Real ur = p.c * ux + p.s * uy, uth = p.c * uy - p.s * ux;
        const Real dD = qm * ((PX * F.Wx + PY * F.Wy) * invd - F.Ez);
        // du+/dxi without the B_perp part (that part is -i chi B+, treated implicitly)
        const Real ax = (qm * (gam * F.Wx + PY * F.Bz) * invd + fp * p.c) * invd - PX * dD * invd * invd;
        const Real ay = (qm * (gam * F.Wy - PX * F.Bz) * invd + fp * p.s) * invd - PY * dD * invd * invd;
        const Real wq = wi * q;
        // term_k = [a+ W + u+ u_r W' - i k u+ (u_th/r) W] e^{-i k th}
        auto add_mode = [&](int k, const View1D& Sr, const View1D& Si, Real ck, Real sk) {
            for (int cc = 0; cc < 2; ++cc) {
                const int jj = p.j + cc;
                const Real iv = g.inv_V(jj) * wq;
                // A = a+ W + u+ u_r W' ;  B = -i k u+ (u_th/r) W = k (u_th/r) W (uy - i ux)
                Real Ar = ax * p.W[cc] + ux * ur * p.dW[cc];
                Real Ai = ay * p.W[cc] + uy * ur * p.dW[cc];
                if (k != 0) {
                    const Real f2 = k * uth * p.Wor[cc];
                    Ar += f2 * uy;
                    Ai -= f2 * ux;
                }
                // times e^{-i k th} = ck - i sk
                Kokkos::atomic_add(&Sr(jj), iv * (Ar * ck + Ai * sk));
                Kokkos::atomic_add(&Si(jj), iv * (Ai * ck - Ar * sk));
            }
        };
        add_mode(1, S.r1, S.i1, p.c, p.s);
        if (m1) {
            add_mode(0, S.r0, S.i0, Real(1), Real(0));
            add_mode(2, S.r2, S.i2, p.c * p.c - p.s * p.s, Real(2) * p.c * p.s);
        }
    });
}

void PlasmaSpecies::push(const SliceFields& f, Real dxi) {
    if (frozen_) return;
    const int ordh = ab_.order;
    const int k = static_cast<int>(npush_ % ordh);   // history slot of this push (k % ord below)
    const bool first0 = (npush_ == 0);
    // xi of the history points: the first push fills all slots with the current force, at fictitious
    // uniformly spaced points (then any coefficients give the Euler step)
    if (first0)
        for (int m = 0; m < ordh; ++m) xi_slot_[(k - m + ordh) % ordh] = xi_cur_ - m * static_cast<double>(dxi);
    else
        xi_slot_[k] = xi_cur_;
    double tn[5];
    bool uniform = true;
    for (int m = 0; m < ordh; ++m) {
        tn[m] = xi_slot_[(k - m + ordh) % ordh];
        if (m > 0 && std::abs((tn[m - 1] - tn[m]) - static_cast<double>(dxi)) > 1e-9 * dxi) uniform = false;
    }
    const ABCoeffs abv = uniform ? ab_ : make_ab_variable(ordh, tn, dxi);
    const Real xi_old = static_cast<Real>(xi_cur_);
    ++npush_;
    xi_cur_ += dxi;
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    auto hist = hist_;
    auto lost = lost_;
    auto fresh = fresh_;
    auto lev = lev_;
    const bool ionz = ionizable_;
    const Real qm0 = q_ / m_;
    const Real dmin = delta_min_;
    const Real qsamax = max_qsa_;
    const ABCoeffs ab = abv;
    const int ord = ab.order;
    const int slot = k;
    const bool first = first0;
    const Real R = grid_.R;
    const bool m1 = m1_;
    const bool trap = trap_;
    auto tw = tw_; auto txi = txi_; auto tpz = tpz_; auto ntrap = ntrap_;
    Kokkos::parallel_for("plasma.push", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real X = x(i), Y = y(i), PX = px(i), PY = py(i), D = dl(i);
        const Real zi = ionz ? lev(i) : Real(1);
        if (zi == Real(0)) {          // neutral atom: ballistic
            x(i) = X + dxi * PX / D; y(i) = Y + dxi * PY / D;
            return;
        }
        const Real qm = qm0 * zi;
        const PGeo p = geometry(g, X, Y);
        const PFields F = gather(f, p, m1, true);
        const Real gam = (Real(1) + PX * PX + PY * PY + qm * qm * F.A + D * D) / (Real(2) * D);
        const Real invd = Real(1) / D;
        const Real fp = -Real(0.5) * qm * qm * F.dA * invd;   // ponderomotive force (laser), radial
        Real fv[5];
        fv[0] = PX * invd;
        fv[1] = PY * invd;
        fv[2] = qm * (gam * F.Wx * invd + F.By + PY * F.Bz * invd) + fp * p.c;
        fv[3] = qm * (gam * F.Wy * invd - F.Bx - PX * F.Bz * invd) + fp * p.s;
        fv[4] = qm * ((PX * F.Wx + PY * F.Wy) * invd - F.Ez);
        if (first || fresh(i)) {      // start (or restart after birth / ionization): first order
            for (int sl = 0; sl < ord; ++sl)
                for (int v = 0; v < 5; ++v) hist(sl, v, i) = fv[v];
            fresh(i) = 0;
        } else {
            for (int v = 0; v < 5; ++v) hist(slot, v, i) = fv[v];
        }
        Real sum[5] = {0, 0, 0, 0, 0};
        for (int m = 0; m < ord; ++m) {
            const int sl = (slot - m + ord) % ord;
            for (int v = 0; v < 5; ++v) sum[v] += ab.c[m] * hist(sl, v, i);
        }
        Real xn = X + dxi * sum[0], yn = Y + dxi * sum[1];
        Real pxn = PX + dxi * sum[2], pyn = PY + dxi * sum[3];
        const Real dn = D + dxi * sum[4];
        // trapped (v_z -> c): not quasi-static.  The quasi-static weight gamma/Delta = 1/(1 - v_z)
        // of a particle in the density and currents diverges there, so such particles are removed
        // (gamma without the laser term: the laser quiver motion is not a forward drift)
        const Real gn = (Real(1) + pxn * pxn + pyn * pyn + dn * dn) / (Real(2) * dn);
        if (!(dn > dmin) || gn > qsamax * dn) {
            if (trap) {   // record the state at this slice (x, y, p_x, p_y, Delta stay unchanged)
                tw(i) = wi;
                txi(i) = xi_old;
                tpz(i) = gam - D;   // incl. the laser quiver term: the mean p_z
                Kokkos::atomic_add(&ntrap(0), 1);
            }
            w(i) = Real(0);
            Kokkos::atomic_add(&lost(0), 1);
            return;
        }
        const Real rn = Kokkos::sqrt(xn * xn + yn * yn);
        if (rn > R) {                 // specular reflection at the wall, incl. the AB history
            const Real nx = xn / rn, ny = yn / rn;
            const Real rr = Real(2) * R - rn;
            if (rr < Real(0)) { w(i) = Real(0); Kokkos::atomic_add(&lost(0), 1); return; }
            xn = nx * rr; yn = ny * rr;
            const Real pn = pxn * nx + pyn * ny;
            pxn -= Real(2) * pn * nx; pyn -= Real(2) * pn * ny;
            for (int sl = 0; sl < ord; ++sl) {
                const Real a = hist(sl, 0, i) * nx + hist(sl, 1, i) * ny;
                hist(sl, 0, i) -= Real(2) * a * nx; hist(sl, 1, i) -= Real(2) * a * ny;
                const Real b = hist(sl, 2, i) * nx + hist(sl, 3, i) * ny;
                hist(sl, 2, i) -= Real(2) * b * nx; hist(sl, 3, i) -= Real(2) * b * ny;
            }
        }
        x(i) = xn; y(i) = yn; px(i) = pxn; py(i) = pyn; dl(i) = dn;
    });
}

Real PlasmaSpecies::max_cells(Real dxi) const {
    if (frozen_ || Np_ == 0) return 0;
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    Real mx = 0;
    Kokkos::parallel_reduce("plasma.max_cells", Range(0, Np_), KOKKOS_LAMBDA(int i, Real& m) {
        if (w(i) == Real(0)) return;
        const Real X = x(i), Y = y(i), D = dl(i);
        const Real X1 = X + dxi * px(i) / D, Y1 = Y + dxi * py(i) / D;
        // closest approach of the straight path to the axis
        const Real dx = X1 - X, dy = Y1 - Y, l2 = dx * dx + dy * dy;
        Real t = l2 > Real(0) ? -(X * dx + Y * dy) / l2 : Real(0);
        t = t < Real(0) ? Real(0) : (t > Real(1) ? Real(1) : t);
        const Real xm = X + t * dx, ym = Y + t * dy;
        const Real c = Kokkos::sqrt(l2) * g.inv_h(g.locate(Kokkos::sqrt(xm * xm + ym * ym)));
        if (c > m) m = c;
    }, Kokkos::Max<Real>(mx));
    return mx;
}

} // namespace quarz
