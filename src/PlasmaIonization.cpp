// PlasmaIonization.cpp: field ionization kernel and charge-state diagnostics (part of the PlasmaSpecies class)
#include "PlasmaSpecies.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include "PlasmaKernels.hpp"

namespace quarz {

using namespace detail;

// ============================================================================ ionization
namespace {
struct IonizeKernel {
    GridD g;
    SliceFields f;
    bool m1 = false;
    IonParams P;
    View3D imp;
    int kl = 0;
    bool has_imp = false;
    Real dxi = 0;
    uint64_t key = 0;
    View1D x, y, px, py, dl, w, lev, wq;
    IView1D fresh, cnt_e, cnt_c;
    bool create = false;
    int nc0 = 0, ne0 = 0;                         // first free index: ion children, electrons
    int nion = 0, tot_e = 0;                      // particles processed, electrons created (create pass)
    View1D ex, ey, epx, epy, edl, ew;
    IView1D efresh;

    // new macro-electron j: parent velocity + laser drift (field-born electrons in the laser)
    KOKKOS_INLINE_FUNCTION void electron(int j, Real X, Real Y, Real PX, Real PY, Real D, Real wgt, const IonRate& R,
                                         bool laser_born, Real Aamp, Real Ex, Real Ey, Real Ez, Real EL, int l,
                                         uint64_t i, uint64_t d) const {
        Real dpx = 0, dpy = 0;
        if (laser_born && Aamp > Real(0)) {
            const int lb = (R.wl[1] > Real(0) && ion_rand(key, i, d) * (R.wl[0] + R.wl[1]) >= R.wl[0]) ? 1 : 0;
            Real A, B, C, H;
            ion_lobe_coeffs(P, lb, Ex, Ey, Ez, EL, A, B, C, H);
            // birth phase phi (from the field maximum) ~ W(E(phi)): rejection from a uniform proposal
            const Real Wp = adk_rate(P, l, ion_E(A, B, C, Real(0)));
            const Real hw = R.sg[lb] < Real(1e29) ? Kokkos::fmin(H, Real(6) * R.sg[lb]) : H;
            Real ph = Real(0);
            for (int tr = 0; tr < 256; ++tr) {
                const Real p1 = hw * (Real(2) * ion_rand(key, i, d + 1 + 2 * tr) - Real(1));
                const Real wv = adk_rate(P, l, ion_E(A, B, C, p1));
                if (!(Wp > Real(0)) || ion_rand(key, i, d + 2 + 2 * tr) * Wp <= wv) { ph = p1; break; }
            }
            const Real u5 = ion_rand(key, i, d + 999);
            if (!P.circular) {
                // a = |a^| sin(phi) at phi from the field maximum (E ~ cos): drift -a, sign irrelevant
                const Real pd = Aamp * Kokkos::sin(ph);
                if (m1) dpx = pd;   // polarisation along x
                else {              // m = 0 ring: the polarisation direction relative to the ring is random
                    const Real al = Real(2) * Real(PI) * u5;
                    dpx = pd * Kokkos::cos(al); dpy = pd * Kokkos::sin(al);
                }
            } else {
                // |drift| = |a^|, perpendicular to the field at birth
                const Real sp = Kokkos::sqrt(Ex * Ex + Ey * Ey);
                const Real th = (m1 && sp > Real(0)) ? Kokkos::atan2(Ey, Ex) + ph : Real(2) * Real(PI) * u5;
                dpx = Aamp * Kokkos::cos(th + Real(0.5) * Real(PI)); dpy = Aamp * Kokkos::sin(th + Real(0.5) * Real(PI));
            }
        }
        ex(j) = X; ey(j) = Y; epx(j) = PX + dpx; epy(j) = PY + dpy; edl(j) = D; ew(j) = wgt; efresh(j) = 1;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if (!create) { cnt_e(i) = 0; cnt_c(i) = 0; }
        else if ((i + 1 < nion ? cnt_e(i + 1) : tot_e) == cnt_e(i)) return;   // nothing happens to this one
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const int z = static_cast<int>(lev(i) + Real(0.5));
        if (z >= P.zmax) return;
        const Real X = x(i), Y = y(i), PX = px(i), PY = py(i), D = dl(i);
        const PGeo pg = geometry(g, X, Y);
        Real Ex = 0, Ey = 0, Ez = 0, EL = 0, Aamp = 0;
        if (P.field) {
            const PFields F = gather(f, pg, m1, true);
            Ex = (F.Wx + F.By) * P.econv; Ey = (F.Wy - F.Bx) * P.econv; Ez = F.Ez * P.econv;   // E = W + (B_y, -B_x)
            if (P.laser) {
                Aamp = Kokkos::sqrt(Kokkos::fmax(P.circular ? F.A : Real(2) * F.A, Real(0)));   // |a^|
                EL = P.k0 * Aamp * P.econv;
            }
        }
        Real Rimp = 0;
        if (P.impact && has_imp && z == 0) {
            for (int c = 0; c < 3; ++c) {
                const Real s = c == 0 ? P.s0 : (c == 1 ? P.s1 : P.s2);
                if (s != Real(0)) Rimp += s * (pg.W[0] * imp(kl, pg.j, c) + pg.W[1] * imp(kl, pg.j + 1, c));
            }
            Rimp = Kokkos::fmax(Rimp, Real(0));
        }
        const Real dtf = dxi * (Real(1) + PX * PX + PY * PY + D * D) / (Real(2) * D * D);   // dxi gamma/Delta
        const uint64_t ii = static_cast<uint64_t>(i);

        // ---- splitting of the first ionization (quantum wq)
        const Real qq = wq(i);
        if (qq > Real(0) && z == P.z0 && wi > qq) {
            const IonRate R = P.field ? ion_field_rate(P, z, Ex, Ey, Ez, EL) : IonRate();
            const Real a = R.W * dtf + Rimp * dxi;
            if (!(a > Real(0))) return;
            const Real dN = -wi * Kokkos::expm1(-a);
            Real we = 0;
            if (dN >= qq) we = dN;
            else if (ion_rand(key, ii, 0) * qq < dN) we = qq;
            if (we <= Real(0)) return;
            if (!create) { cnt_e(i) = 1; cnt_c(i) = 1; return; }
            const int jc = nc0 + cnt_c(i);
            x(jc) = X; y(jc) = Y; px(jc) = PX; py(jc) = PY; dl(jc) = D; w(jc) = we;
            lev(jc) = Real(z + 1); wq(jc) = Real(0); fresh(jc) = 1;
            w(i) = wi - we;
            const bool lb = P.laser && ion_rand(key, ii, 1) * a < R.W * dtf;
            electron(ne0 + cnt_e(i), X, Y, PX, PY, D, we, R, lb, Aamp, Ex, Ey, Ez, EL, z, ii, 1u << 20);
            return;
        }
        // ---- whole macro-particle, possibly several levels in this slice
        int ne = 0;
        for (int l = z; l < P.zmax; ++l) {
            const IonRate R = P.field ? ion_field_rate(P, l, Ex, Ey, Ez, EL) : IonRate();
            const Real a = R.W * dtf + (l == 0 ? Rimp : Real(0)) * dxi;
            if (!(a > Real(0))) break;
            const uint64_t d = 16 + 16 * static_cast<uint64_t>(l);
            if (ion_rand(key, ii, d) >= -Kokkos::expm1(-a)) break;
            if (create) {
                const bool lb = P.laser && ion_rand(key, ii, d + 1) * a < R.W * dtf;
                electron(ne0 + cnt_e(i) + ne, X, Y, PX, PY, D, wi, R, lb, Aamp, Ex, Ey, Ez, EL, l, ii, (2u + static_cast<uint64_t>(l)) << 20);
            }
            ++ne;
        }
        if (!create) cnt_e(i) = ne;
        else if (ne > 0) { lev(i) = Real(z + ne); fresh(i) = 1; }
    }
};
} // namespace

IonizeResult PlasmaSpecies::ionize(const SliceFields& f, const View3D& imp, int kl, Real dxi, int k, int step,
                                   PlasmaSpecies& prod) {
    IonizeResult res;
    if (!ionizable_ || Np_ == 0) return res;
    uint64_t nh = 1469598103934665603ULL;   // FNV-1a of the species name
    for (char c : name_) { nh ^= static_cast<unsigned char>(c); nh *= 1099511628211ULL; }
    const uint64_t key = ion_mix(seed_ ^ ion_mix(static_cast<uint64_t>(step) ^ ion_mix(static_cast<uint64_t>(k) ^ ion_mix(nh))));
    const int N = Np_;
    int tot_e = 0, tot_c = 0;
    auto make = [&](bool create) {
        IonizeKernel K;
        K.g = grid_.d; K.f = f; K.m1 = m1_; K.P = ion_.P;
        K.imp = imp; K.kl = kl; K.has_imp = imp.extent(0) > 0;
        K.dxi = dxi; K.key = key;
        K.x = x_; K.y = y_; K.px = px_; K.py = py_; K.dl = dl_; K.w = w_; K.lev = lev_; K.wq = wq_;
        K.fresh = fresh_; K.cnt_e = cnt_e_; K.cnt_c = cnt_c_;
        K.create = create; K.nc0 = Np_; K.ne0 = prod.Np_;
        K.nion = N; K.tot_e = tot_e;
        K.ex = prod.x_; K.ey = prod.y_; K.epx = prod.px_; K.epy = prod.py_; K.edl = prod.dl_; K.ew = prod.w_;
        K.efresh = prod.fresh_;
        return K;
    };
    Kokkos::parallel_for("ion.count", Range(0, N), make(false));
    auto ce = cnt_e_, cc = cnt_c_;
    Kokkos::parallel_scan("ion.scan_e", Range(0, N), KOKKOS_LAMBDA(int i, int& u, bool fin) {
        const int v = ce(i); if (fin) ce(i) = u; u += v; }, tot_e);
    if (tot_e == 0) return res;
    Kokkos::parallel_scan("ion.scan_c", Range(0, N), KOKKOS_LAMBDA(int i, int& u, bool fin) {
        const int v = cc(i); if (fin) cc(i) = u; u += v; }, tot_c);
    prod.ensure_capacity(prod.Np_ + tot_e);
    ensure_capacity(Np_ + tot_c);
    Kokkos::parallel_for("ion.create", Range(0, N), make(true));
    // statistics of the new electrons
    const int e0 = prod.Np_;
    auto ew = prod.w_; auto epx = prod.px_; auto epy = prod.py_;
    double sw = 0, swp = 0;
    Kokkos::parallel_reduce("ion.stats", Range(e0, e0 + tot_e), KOKKOS_LAMBDA(int j, double& a, double& b) {
        a += ew(j); b += ew(j) * (epx(j) * epx(j) + epy(j) * epy(j)); }, sw, swp);
    prod.Np_ += tot_e;
    Np_ += tot_c;
    res.born_w = sw; res.born_wp2 = swp; res.born_n = tot_e;
    return res;
}

void PlasmaSpecies::deposit_charge_state(const View3D& out, int kl, int comp) const {
    if (!ionizable_) return;
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_; auto lev = lev_;
    Kokkos::parallel_for("ion.charge_state", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wz = w(i) * lev(i);
        if (wz == Real(0)) return;
        const Real D = dl(i);
        const Real gd = (Real(1) + px(i) * px(i) + py(i) * py(i) + D * D) / (Real(2) * D * D);
        const PGeo p = geometry(g, x(i), y(i));
        for (int cc = 0; cc < 2; ++cc)
            Kokkos::atomic_add(&out(kl, p.j + cc, comp), wz * gd * p.W[cc] * g.inv_V(p.j + cc));
    });
}

void PlasmaSpecies::charge_sums(double& zw, double& wsum) const {
    zw = wsum = 0;
    if (!ionizable_) return;
    auto w = w_; auto lev = lev_;
    Kokkos::parallel_reduce("ion.sums", Range(0, Np_), KOKKOS_LAMBDA(int i, double& a, double& b) {
        a += w(i) * lev(i); b += w(i); }, zw, wsum);
}

} // namespace quarz
