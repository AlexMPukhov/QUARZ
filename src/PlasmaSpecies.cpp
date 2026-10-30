#include "PlasmaSpecies.hpp"
#include "Config.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

namespace quarz {

ABCoeffs make_ab(int order) {
    ABCoeffs ab;
    ab.order = order;
    switch (order) {
        case 1: ab.c[0] = 1; break;
        case 2: ab.c[0] = 1.5; ab.c[1] = -0.5; break;
        case 3: ab.c[0] = 23. / 12; ab.c[1] = -16. / 12; ab.c[2] = 5. / 12; break;
        case 4: ab.c[0] = 55. / 24; ab.c[1] = -59. / 24; ab.c[2] = 37. / 24; ab.c[3] = -9. / 24; break;
        case 5:
            ab.c[0] = 1901. / 720; ab.c[1] = -2774. / 720; ab.c[2] = 2616. / 720;
            ab.c[3] = -1274. / 720; ab.c[4] = 251. / 720; break;
        default: throw std::runtime_error("pusher.ab_order must be 1..5");
    }
    return ab;
}

ABCoeffs make_ab_variable(int order, const double* t, double h) {
    ABCoeffs ab;
    ab.order = order;
    // 3-point Gauss-Legendre on [t0, t0 + h]: exact for the polynomial degree <= 4 needed here
    const double gx[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const double gw[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    for (int m = 0; m < order; ++m) {
        double s = 0;
        for (int q = 0; q < 3; ++q) {
            const double tt = t[0] + 0.5 * h * (1.0 + gx[q]);
            double l = 1;
            for (int n = 0; n < order; ++n)
                if (n != m) l *= (tt - t[n]) / (t[m] - t[n]);
            s += 0.5 * gw[q] * l;   // (1/h) * (h/2) * w
        }
        ab.c[m] = s;
    }
    return ab;
}

namespace {
// geometry of one particle: radius, azimuth, hat-function weights
struct PGeo {
    Real r, c, s;          // radius, cos/sin of azimuth
    int j;                 // cell
    Real W[2], dW[2], Wor[2];   // W, dW/dr, W/r (W/r := 0 on the axis node)
};
KOKKOS_INLINE_FUNCTION PGeo geometry(const GridD& g, Real x, Real y) {
    PGeo p;
    p.r = Kokkos::sqrt(x * x + y * y);
    if (p.r > Real(0)) { p.c = x / p.r; p.s = y / p.r; } else { p.c = Real(1); p.s = Real(0); }
    Real t;
    g.weights(p.r, p.j, t);
    const Real ih = g.inv_h(p.j);
    p.W[0] = Real(1) - t; p.W[1] = t;
    p.dW[0] = -ih; p.dW[1] = ih;
    const Real ir = p.r > Real(0) ? Real(1) / p.r : Real(0);
    p.Wor[0] = (p.j == 0) ? Real(0) : p.W[0] * ir;
    p.Wor[1] = (p.j == 0) ? ih : p.W[1] * ir;      // W_1/r = 1/h_0 exactly in the first cell
    return p;
}
KOKKOS_INLINE_FUNCTION Real lin(const View1D& f, const PGeo& p) { return p.W[0] * f(p.j) + p.W[1] * f(p.j + 1); }

// fields at a particle
struct PFields { Real Wx, Wy, Bx, By, Bz, Ez, A, dA; };   // A = <a^2>, dA = d<a^2>/dr (laser)
KOKKOS_INLINE_FUNCTION PFields gather(const SliceFields& f, const PGeo& p, bool m1, bool needB) {
    PFields o;
    eval_vector(lin(f.wp.r0, p), lin(f.wp.i0, p), lin(f.wp.r1, p), lin(f.wp.i1, p), lin(f.wp.r2, p),
                lin(f.wp.i2, p), p.c, p.s, m1, o.Wx, o.Wy);
    o.Ez = eval_scalar(lin(f.ez.a0, p), lin(f.ez.ar, p), lin(f.ez.ai, p), p.c, p.s, m1);
    o.Bz = eval_scalar(lin(f.bz.a0, p), lin(f.bz.ar, p), lin(f.bz.ai, p), p.c, p.s, m1);
    if (needB)
        eval_vector(lin(f.bp.r0, p), lin(f.bp.i0, p), lin(f.bp.r1, p), lin(f.bp.i1, p), lin(f.bp.r2, p),
                    lin(f.bp.i2, p), p.c, p.s, m1, o.Bx, o.By);
    else
        o.Bx = o.By = Real(0);
    if (f.laser) { o.A = lin(f.aa, p); o.dA = lin(f.daa, p); }
    else o.A = o.dA = Real(0);
    return o;
}
} // namespace

PlasmaSpecies::PlasmaSpecies(const Config& cfg, const std::string& name, const RadialGrid& grid,
                             const DensityProfile& prof, bool mode1)
    : name_(name), grid_(grid), prof_(prof), m1_(mode1) {
    {   // species-specific parsed profile <species>.density(x,y,z) overrides the common plasma profile
        Parser tmp;
        if (read_function(cfg, name + ".density", 3, tmp, {"x", "y", "z"})) prof_ = DensityProfile(cfg, name);
    }
    ionizable_ = is_ionizable(cfg, name);
    if (ionizable_) {
        ion_ = make_ion_species(cfg, name, cfg.get_double("units.n0_cm3", 0.0));
        if (cfg.has(name + ".charge"))
            throw std::runtime_error(name + ".charge: an ionizable species has the charge z e of its charge state");
        q_ = 1;
        if (ion_.mass_me <= 0 && !cfg.has(name + ".mass")) throw std::runtime_error(name + ".mass is required (in electron masses)");
        m_ = cfg.get_double(name + ".mass", ion_.mass_me);
        // immobile ionizable ions are still macro-particles (their charge changes slice by slice), but
        // they are not pushed
        frozen_ = !cfg.get_bool(name + ".mobile", true);
        mobile_ = true;
    } else {
        q_ = cfg.get_double(name + ".charge", -1.0);
        m_ = cfg.get_double(name + ".mass", 1.0);
        mobile_ = cfg.get_bool(name + ".mobile", true);
    }
    seed_ = static_cast<unsigned long long>(cfg.get_int("plasma.seed", 1));
    ppc_ = cfg.get_int(name + ".ppc", 4);
    ntheta_ = cfg.get_int(name + ".ntheta", m1_ ? 8 : 1);
    // thermal spread: a number or a function  <species>.uth(x,y,z)  (z = lab position)
    if (!read_function(cfg, name + ".uth", 3, uth_, {"x", "y", "z"}, true)) uth_ = Parser("0", {});
    density_factor_ = cfg.get_double(name + ".density", 1.0);
    if (cfg.has(name + ".u_drift")) {
        const auto u = cfg.get_list(name + ".u_drift");
        if (u.size() != 3) throw std::runtime_error(name + ".u_drift: three numbers u_x u_y u_z");
        for (int c = 0; c < 3; ++c) udrift_[c] = std::stod(u[c]);
    }
    tail_dump_ = cfg.get_bool(name + ".tail_dump", false);
    // declustering: plasma.declustering = 1 turns it on for the mobile electron species (charge < 0); <species>.declustering
    // overrides per species (e.g. 1 for mobile ions). Parameters plasma.declustering.K / .kappa / .D0 (D0 in local cells)
    {
        const bool on = cfg.get_bool("plasma.declustering", false);
        decl_ = cfg.get_bool(name + ".declustering", on && q_ < 0 && mobile_ && !ionizable_);
        if (decl_ && (!mobile_ || frozen_)) throw std::runtime_error(name + ".declustering: only for mobile species");
        decl_K_ = cfg.get_double("plasma.declustering.K", 0.01);
        decl_kappa_ = cfg.get_double("plasma.declustering.kappa", 0.005);
        decl_D0_ = cfg.get_double("plasma.declustering.D0", 0.01);
        decl_win_ = cfg.get_int("plasma.declustering.window", 4);
        decl_stride_ = cfg.get_int("plasma.declustering.stride", 1);
        if (decl_ && (decl_K_ < 0 || decl_kappa_ < 0 || !(decl_D0_ > 0) || decl_win_ < 1))
            throw std::runtime_error("plasma.declustering: K, kappa >= 0, D0 > 0, window >= 1 required");
    }
    delta_min_ = cfg.get_double("pusher.delta_min", 1e-3);
    max_qsa_ = cfg.get_double("pusher.max_qsa_factor", 35.0);
    ab_ = make_ab(cfg.get_int("pusher.ab_order", 3));
    if (ppc_ < 0 || (ppc_ == 0 && !mobile_)) throw std::runtime_error(name + ".ppc must be >= 1 (0: an initially empty mobile species, e.g. for ionization electrons)");
    if (ntheta_ < 1) throw std::runtime_error(name + ".ntheta must be >= 1");
    if (m1_ && ntheta_ < 4)
        throw std::runtime_error(name + ".ntheta must be >= 4 with azimuthal mode 1 (8 recommended)");

    Nload_ = Np_ = grid.N * ppc_ * ntheta_;
    cap_ = std::max(Nload_, 1);
    x_ = View1D(name + ".x", cap_);
    y_ = View1D(name + ".y", cap_);
    px_ = View1D(name + ".px", cap_);
    py_ = View1D(name + ".py", cap_);
    dl_ = View1D(name + ".delta", cap_);
    w_ = View1D(name + ".w", cap_);
    fresh_ = IView1D(name + ".fresh", cap_);
    hist_ = View3D(name + ".hist", ab_.order, 5, cap_);
    lost_ = IView1D(name + ".lost", 1);
    // electrons receiving delta electrons: fast ones (beyond the quasi-static limit) become beam particles
    trap_ = cfg.get_bool(name + ".trap", mobile_ && !ionizable_ && receives_delta_electrons(cfg, name));
    if (trap_) {
        if (ionizable_ || !mobile_) throw std::runtime_error(name + ".trap: only for mobile species without ionization (e.g. the electrons)");
        trap_beam_ = cfg.get_string(name + ".trap_beam", name + "_trapped");
        trap_nphi_ = cfg.get_int(name + ".trap_nphi", 1);
        if (trap_nphi_ < 1) throw std::runtime_error(name + ".trap_nphi must be >= 1");
        if (m1_ && trap_nphi_ != 1) throw std::runtime_error(name + ".trap_nphi: only with modes = 0 (with modes = 1 the particles keep their azimuth)");
        tw_ = View1D(name + ".trap_w", cap_);
        txi_ = View1D(name + ".trap_xi", cap_);
        tpz_ = View1D(name + ".trap_pz", cap_);
        ntrap_ = IView1D(name + ".ntrap", 1);
    }
    if (decl_) {
        r0_ = View1D(name + ".r0", cap_); dprev_ = View1D(name + ".dprev", cap_);
        dg_ = View1D(name + ".dg", cap_); dD_ = View1D(name + ".dD", cap_); da_ = View1D(name + ".da", cap_);
    }
    if (ionizable_) {
        lev_ = View1D(name + ".z", cap_);
        wq_ = View1D(name + ".wq", cap_);
        if (ion_.quiet) acc_ = View1D(name + ".acc", cap_);
        cnt_e_ = IView1D(name + ".cnt_e", cap_);
        cnt_c_ = IView1D(name + ".cnt_c", cap_);
    }
}

double PlasmaSpecies::max_density(double z0, double z1, double R) const {
    const int NZ = 256, NR = 64;
    double nmax = 0;
    for (int a = 0; a <= NZ; ++a) {
        const double z = z0 + (z1 - z0) * a / NZ;
        const double fz = prof_.parsed() ? density_factor_ : prof_.longitudinal(z) * density_factor_;
        for (int b = 0; b <= NR; ++b) {
            const double r = R * b / NR;
            const double n = fz * (prof_.parsed() ? prof_.eval(r, 0.0, z) : prof_.radial(r));
            nmax = std::max(nmax, n);
        }
    }
    return nmax;
}

void PlasmaSpecies::load(Real z, unsigned long long seed) {
    auto hx = Kokkos::create_mirror_view(x_);
    auto hy = Kokkos::create_mirror_view(y_);
    auto hpx = Kokkos::create_mirror_view(px_);
    auto hpy = Kokkos::create_mirror_view(py_);
    auto hdl = Kokkos::create_mirror_view(dl_);
    auto hw = Kokkos::create_mirror_view(w_);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const bool parsed = prof_.parsed();
    const double fz = parsed ? density_factor_ : prof_.longitudinal(z) * density_factor_;
    int i = 0;
    for (int j = 0; j < grid_.N; ++j) {
        const double h = grid_.h[j];
        for (int m = 0; m < ppc_; ++m) {
            // sub-ring [ra, rb]; particles at its volume centroid -> exact deposit of a uniform plasma
            const double ra = grid_.r[j] + m * h / ppc_, rb = grid_.r[j] + (m + 1) * h / ppc_;
            const double r = (2.0 / 3.0) * (rb * rb * rb - ra * ra * ra) / (rb * rb - ra * ra);
            const double vring = fz * PI * (rb * rb - ra * ra);
            // built-in profiles: n(r) f(z);  parsed: ring average (ntheta = 1) or the value at the particle
            const double wring = parsed ? (ntheta_ == 1 ? prof_.eval_avg(r, z) * vring : vring)
                                        : prof_.radial(r) * vring;
            const double off = ((j * ppc_ + m) % 2) * 0.5;   // stagger neighbouring rings in azimuth
            for (int l = 0; l < ntheta_; ++l, ++i) {
                const double th = 2 * PI * (l + off) / ntheta_;
                double px = 0, py = 0, pz = 0;
                double ut;
                if (uth_.is_constant()) ut = uth_(0, 0, z);
                else if (ntheta_ == 1) {   // axisymmetric run: rms average over the ring
                    ut = 0;
                    for (int q = 0; q < 16; ++q) {
                        const double v = uth_(r * std::cos(2 * PI * q / 16), r * std::sin(2 * PI * q / 16), z);
                        ut += v * v;
                    }
                    ut = std::sqrt(ut / 16);
                } else ut = uth_(r * std::cos(th), r * std::sin(th), z);
                if (ut > 0) { px = ut * gauss(rng); py = ut * gauss(rng); pz = ut * gauss(rng); }
                px += udrift_[0]; py += udrift_[1]; pz += udrift_[2];
                const double gam = std::sqrt(1 + px * px + py * py + pz * pz);
                const double X = (ntheta_ == 1) ? r : r * std::cos(th), Y = (ntheta_ == 1) ? 0.0 : r * std::sin(th);
                const double W = (parsed && ntheta_ > 1) ? prof_.eval(X, Y, z) * wring / ntheta_ : wring / ntheta_;
                // rings without plasma (outside the column, e.g. plasma.rmax or a parsed profile < grid
                // radius) are not loaded: they carry no charge and would only cost time. The random
                // numbers above are drawn anyway, so the other particles do not change. Ionizable
                // species keep all rings (their random numbers are keyed on the particle index).
                if (W == 0 && !ionizable_) { --i; continue; }
                hx(i) = X;
                hy(i) = Y;
                hpx(i) = px;
                hpy(i) = py;
                hdl(i) = gam - pz;
                hw(i) = W;
            }
        }
    }
    Kokkos::deep_copy(x_, hx); Kokkos::deep_copy(y_, hy);
    Kokkos::deep_copy(px_, hpx); Kokkos::deep_copy(py_, hpy);
    Kokkos::deep_copy(dl_, hdl); Kokkos::deep_copy(w_, hw);
    Kokkos::deep_copy(hist_, 0.0);
    Kokkos::deep_copy(lost_, 0);
    Kokkos::deep_copy(fresh_, 0);
    if (trap_) { Kokkos::deep_copy(tw_, 0.0); Kokkos::deep_copy(ntrap_, 0); }
    Np_ = i;   // rings with plasma (<= Nload_)
    nfam_ = i;
    if (decl_) {
        auto x = x_, y = y_, r0 = r0_, dp = dprev_;
        Kokkos::parallel_for("decl.init", Range(0, nfam_), KOKKOS_LAMBDA(int q) {
            r0(q) = Kokkos::sqrt(x(q) * x(q) + y(q) * y(q)); dp(q) = Real(0); });
    }
    npush_ = 0;
    xi_cur_ = 0;
    if (ionizable_) {
        Kokkos::deep_copy(lev_, Real(ion_.P.z0));
        auto wq = wq_; auto w = w_;
        const Real f = ion_.nsplit > 1 ? Real(1) / ion_.nsplit : Real(0);
        Kokkos::parallel_for("ion.init_wq", Range(0, cap_), KOKKOS_LAMBDA(int i) { wq(i) = w(i) * f; });
        if (ion_.quiet) {
            // initial phases of the accumulators interleaved along the particle index (neighbours in r): golden-ratio
            // sequence with a random offset, so that the emissions of the atoms in a region follow each other evenly
            // (random phases would leave the shot noise of the emitted quanta). The offset changes from step to step
            // (z of the box front), not between ranks.
            auto acc = acc_;
            const uint64_t key = ion_mix(seed_ ^ 0x9e3779b97f4a7c15ULL ^ ion_mix(static_cast<uint64_t>(std::llround(double(z) * 1024.0))));
            const Real s0 = ion_rand(key, 0, 7);
            Kokkos::parallel_for("ion.init_acc", Range(0, cap_), KOKKOS_LAMBDA(int i) {
                const Real v = s0 + Real(i) * Real(0.61803398874989485);
                acc(i) = v - Kokkos::floor(v);
            });
        }
    }
}

void PlasmaSpecies::ensure_capacity(int n) {
    if (n <= cap_) return;
    const int nc = std::max(n, cap_ + cap_ / 2);
    for (View1D* v : {&x_, &y_, &px_, &py_, &dl_, &w_}) Kokkos::resize(*v, nc);
    Kokkos::resize(fresh_, nc);
    Kokkos::resize(hist_, ab_.order, 5, nc);
    if (trap_) for (View1D* v : {&tw_, &txi_, &tpz_}) Kokkos::resize(*v, nc);   // new entries are 0
    if (ionizable_) {
        Kokkos::resize(lev_, nc);
        Kokkos::resize(wq_, nc);
        if (ion_.quiet) Kokkos::resize(acc_, nc);
        Kokkos::resize(cnt_e_, nc);
        Kokkos::resize(cnt_c_, nc);
    }
    if (decl_)
        for (View1D* v : {&r0_, &dprev_, &dg_, &dD_, &da_}) Kokkos::resize(*v, nc);
    cap_ = nc;
}

// [Np][x y px py Delta w fresh (z wq)][hist slabs], each array the first Np entries
void PlasmaSpecies::pack_state(std::vector<double>& buf) const {
    buf.push_back(static_cast<double>(Np_));
    std::vector<View1D> vs = {x_, y_, px_, py_, dl_, w_};
    if (ionizable_) { vs.push_back(lev_); vs.push_back(wq_); if (ion_.quiet) vs.push_back(acc_); }
    if (decl_) { vs.push_back(r0_); vs.push_back(dprev_); }
    for (const View1D& v : vs) {
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), v);
        buf.insert(buf.end(), h.data(), h.data() + Np_);
    }
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace(), fresh_);
    for (int i = 0; i < Np_; ++i) buf.push_back(static_cast<double>(hf(i)));
    auto hh = Kokkos::create_mirror_view_and_copy(HostSpace(), hist_);
    for (int sl = 0; sl < ab_.order; ++sl)
        for (int v = 0; v < 5; ++v) buf.insert(buf.end(), &hh(sl, v, 0), &hh(sl, v, 0) + Np_);
    buf.push_back(static_cast<double>(npush_));
    buf.push_back(xi_cur_);
    for (int sl = 0; sl < ab_.order; ++sl) buf.push_back(xi_slot_[sl]);
}

size_t PlasmaSpecies::unpack_state(const double* p) {
    size_t off = 0;
    Np_ = static_cast<int>(p[off++]);
    ensure_capacity(Np_);
    std::vector<View1D> vs = {x_, y_, px_, py_, dl_, w_};
    if (ionizable_) { vs.push_back(lev_); vs.push_back(wq_); if (ion_.quiet) vs.push_back(acc_); }
    if (decl_) { vs.push_back(r0_); vs.push_back(dprev_); }
    for (View1D v : vs) {
        auto h = Kokkos::create_mirror_view(v);
        for (int i = 0; i < Np_; ++i) h(i) = p[off + i];
        Kokkos::deep_copy(v, h);
        off += Np_;
    }
    auto hf = Kokkos::create_mirror_view(fresh_);
    for (int i = 0; i < Np_; ++i) hf(i) = static_cast<int>(p[off + i]);
    Kokkos::deep_copy(fresh_, hf);
    off += Np_;
    auto hh = Kokkos::create_mirror_view(hist_);
    for (int sl = 0; sl < ab_.order; ++sl)
        for (int v = 0; v < 5; ++v) {
            for (int i = 0; i < Np_; ++i) hh(sl, v, i) = p[off + i];
            off += Np_;
        }
    Kokkos::deep_copy(hist_, hh);
    npush_ = static_cast<long>(p[off++]);
    xi_cur_ = p[off++];
    for (int sl = 0; sl < ab_.order; ++sl) xi_slot_[sl] = p[off++];
    return off;
}

void PlasmaSpecies::write_state(const std::string& fn) const {
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), x_);
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), y_);
    auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), px_);
    auto hpy = Kokkos::create_mirror_view_and_copy(HostSpace(), py_);
    auto hd = Kokkos::create_mirror_view_and_copy(HostSpace(), dl_);
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), w_);
    std::ofstream o(fn);
    o << "# x y px py Delta w (particles at the end of the box)\n" << std::setprecision(17);
    for (int i = 0; i < Np_; ++i)
        if (hw(i) != 0) o << hx(i) << " " << hy(i) << " " << hpx(i) << " " << hpy(i) << " " << hd(i) << " " << hw(i) << "\n";
}

int PlasmaSpecies::lost_count() const {
    auto h = Kokkos::create_mirror_view(lost_);
    Kokkos::deep_copy(h, lost_);
    return h(0);
}

void PlasmaSpecies::reset_lost() { Kokkos::deep_copy(lost_, 0); }

int PlasmaSpecies::collect_trapped(std::vector<double>& out) {
    if (!trap_) return 0;
    auto hn = Kokkos::create_mirror_view_and_copy(HostSpace(), ntrap_);
    if (hn(0) == 0) return 0;
    auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), tw_);
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), x_);
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), y_);
    auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), px_);
    auto hpy = Kokkos::create_mirror_view_and_copy(HostSpace(), py_);
    auto hpz = Kokkos::create_mirror_view_and_copy(HostSpace(), tpz_);
    auto hxi = Kokkos::create_mirror_view_and_copy(HostSpace(), txi_);
    int n = 0;
    for (int i = 0; i < Np_; ++i) {
        if (!(hw(i) > 0)) continue;
        out.insert(out.end(), {hx(i), hy(i), hpx(i), hpy(i), hpz(i), hxi(i), hw(i)});
        ++n;
    }
    Kokkos::deep_copy(tw_, 0.0);
    Kokkos::deep_copy(ntrap_, 0);
    return n;
}

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

// ============================================================================ electron impact
void PlasmaSpecies::deposit_eimpact(const View3D& eimp, const EITargets& eit) const {
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    const EITargets T = eit;
    const int NT = eit.T;
    const Real norm = eit.norm;
    Kokkos::parallel_for("plasma.eimpact_deposit", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const Real PX = px(i), PY = py(i), D = dl(i);
        const Real gam = (Real(1) + PX * PX + PY * PY + D * D) / (Real(2) * D);
        const Real E = (gam - Real(1)) * EI_ME_EV;
        const Real f = wi * Kokkos::sqrt(Kokkos::fmax(gam * gam - Real(1), Real(0))) / D * norm;   // w |u|/Delta
        if (!(f > Real(0))) return;
        PGeo p;
        bool have = false;
        for (int t = 0; t < NT; ++t) {
            const Real sg = ei_sigma(T, t, E);
            if (!(sg > Real(0))) continue;
            if (!have) { p = geometry(g, x(i), y(i)); have = true; }
            const Real r = f * sg;
            int b = static_cast<int>(Kokkos::floor(EI_BPO * Kokkos::log2(E / T.par(t, 0))));
            b = b < 0 ? 0 : (b >= EI_NB ? EI_NB - 1 : b);
            for (int cc = 0; cc < 2; ++cc) {
                const int jj = p.j + cc;
                const Real v = r * p.W[cc] * g.inv_V(jj);
                Kokkos::atomic_add(&eimp(t, 2 * b, jj), v);
                Kokkos::atomic_add(&eimp(t, 2 * b + 1, jj), v * E);
            }
        }
    });
}

void PlasmaSpecies::deposit_targets(const View2D& ntg) const {
    if (!ionizable_ || !ion_.P.eimp) return;
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_; auto lev = lev_;
    const int z0 = ion_.P.z0, zmax = ion_.P.zmax, t0 = ion_.P.et0;
    Kokkos::parallel_for("plasma.eimpact_targets", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        const Real wi = w(i);
        if (wi == Real(0)) return;
        const int l = static_cast<int>(lev(i) + Real(0.5));
        if (l < z0 || l >= zmax) return;
        const Real PX = px(i), PY = py(i), D = dl(i);
        const Real gd = (Real(1) + PX * PX + PY * PY + D * D) / (Real(2) * D * D);   // gamma/Delta
        const PGeo p = geometry(g, x(i), y(i));
        for (int cc = 0; cc < 2; ++cc) {
            const int jj = p.j + cc;
            Kokkos::atomic_add(&ntg(t0 + l - z0, jj), wi * gd * p.W[cc] * g.inv_V(jj));
        }
    });
}

// energy loss of the projectiles over a step h (time in the slice tau = h gamma/Delta):
//   continuous: dE = sum_t n_t [sigma_t (Ip_t + <W>_t) + sum_j sigma_j W_j] v tau  (expected loss);
//   discrete:   collisions drawn one by one with the rate nu = sum n sigma v (null-free: the energy and the rate
//               are updated after every collision): ionization loses Ip + W, W from 1/(J^2 + W^2) on
//               [0, (E - Ip)/2] (the ion side makes the ions and the secondary electrons from the rates; both
//               sides agree on average), excitation of channel j loses W_j.
// The direction is kept; random numbers keyed on (seed, step, slice, sub-slice, species, particle).
void PlasmaSpecies::eloss(const View2D& ntg, const EITargets& eit, Real h, bool discrete, int k, int step, int sub) {
    const GridD g = grid_.d;
    auto x = x_; auto y = y_; auto px = px_; auto py = py_; auto dl = dl_; auto w = w_;
    const EITargets T = eit;
    const int NT = eit.T;
    const Real norm = eit.norm;
    uint64_t nh = 1469598103934665603ULL;
    for (char c : name_) { nh ^= static_cast<unsigned char>(c); nh *= 1099511628211ULL; }
    const uint64_t key = ion_mix(seed_ ^ ion_mix(0xE1055ULL + static_cast<uint64_t>(step) ^
                                 ion_mix(static_cast<uint64_t>(k) * 64 + static_cast<uint64_t>(sub) ^ ion_mix(nh))));
    Kokkos::parallel_for("plasma.eimpact_loss", Range(0, Np_), KOKKOS_LAMBDA(int i) {
        if (w(i) == Real(0)) return;
        const Real PX = px(i), PY = py(i), D = dl(i);
        const Real gam = (Real(1) + PX * PX + PY * PY + D * D) / (Real(2) * D);
        const Real E0 = (gam - Real(1)) * EI_ME_EV;
        Real E = E0;
        PGeo p;
        bool have = false;
        const Real tau = h * gam / D;   // time in the slice
        if (!discrete) {
            Real L = 0;
            for (int t = 0; t < NT; ++t) {
                const Real sg = ei_sigma(T, t, E), sx = ei_exc_loss(T, t, E);   // ionization, excitation
                if (!(sg > Real(0)) && !(sx > Real(0))) continue;
                if (!have) { p = geometry(g, x(i), y(i)); have = true; }
                const Real n = p.W[0] * ntg(t, p.j) + p.W[1] * ntg(t, p.j + 1);
                L += n * (sg * (T.par(t, 0) + ei_mean_w(T, t, E)) + sx);
            }
            if (!(L > Real(0))) return;
            E = Kokkos::fmax(E - L * norm * Kokkos::sqrt(gam * gam - Real(1)) / gam * tau, Real(0));
        } else {
            Real left = tau;
            const uint64_t ii = static_cast<uint64_t>(i);
            for (int ev = 0; ev < 256; ++ev) {
                // total rate nu = sum_t n_t (sigma_ion + sum_j sigma_j) v at the present energy
                const Real gm = Real(1) + E / EI_ME_EV, v = Kokkos::sqrt(Real(1) - Real(1) / (gm * gm));
                Real nu = 0;
                for (int t = 0; t < NT; ++t) {
                    Real st = ei_sigma(T, t, E);
                    const int nx = static_cast<int>(T.par(t, 15)), o = static_cast<int>(T.par(t, 14));
                    for (int j = o; j < o + nx; ++j) st += ei_exc_sigma(T, j, E);
                    if (!(st > Real(0))) continue;
                    if (!have) { p = geometry(g, x(i), y(i)); have = true; }
                    nu += (p.W[0] * ntg(t, p.j) + p.W[1] * ntg(t, p.j + 1)) * st;
                }
                nu *= norm * v;
                if (!(nu > Real(0))) break;
                const Real s = -Kokkos::log(Real(1) - ion_rand(key, ii, 3 * ev)) / nu;
                if (s > left) break;
                left -= s;
                // which collision
                Real u = ion_rand(key, ii, 3 * ev + 1) * nu / (norm * v), c = 0, dE = 0;
                for (int t = 0; t < NT && dE == Real(0); ++t) {
                    const Real n = p.W[0] * ntg(t, p.j) + p.W[1] * ntg(t, p.j + 1);
                    const Real si = n * ei_sigma(T, t, E);
                    if (u < c + si) {
                        const Real Ip = T.par(t, 0), J = T.par(t, 11), wm = Real(0.5) * (E - Ip);
                        dE = Ip + J * Kokkos::tan(ion_rand(key, ii, 3 * ev + 2) * Kokkos::atan(wm / J));
                        break;
                    }
                    c += si;
                    const int nx = static_cast<int>(T.par(t, 15)), o = static_cast<int>(T.par(t, 14));
                    for (int j = o; j < o + nx; ++j) {
                        const Real sj = n * ei_exc_sigma(T, j, E);
                        if (u < c + sj) { dE = T.exc(j, 0); break; }
                        c += sj;
                    }
                }
                if (dE == Real(0)) break;   // round-off at the end of the list
                E = Kokkos::fmax(E - dE, Real(0));
            }
            if (E == E0) return;
        }
        const Real u0 = Kokkos::sqrt(gam * gam - Real(1));
        const Real E1 = E / EI_ME_EV;
        const Real k1 = Kokkos::sqrt(E1 * (E1 + Real(2))) / u0;
        const Real uz = gam - D;
        const Real qx = PX * k1, qy = PY * k1, qz = uz * k1;
        const Real q2 = Real(1) + qx * qx + qy * qy, g1 = Kokkos::sqrt(q2 + qz * qz);
        px(i) = qx; py(i) = qy; dl(i) = qz > Real(0) ? q2 / (g1 + qz) : g1 - qz;
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
    View1D eflag;                                 // born_dump: 0 field, 1 beam impact, 2 electron impact
    View3D eimp;                                  // electron impact: rates per target, energy bin, node
    EITargets eit;
    bool has_e = false;
    bool quiet = false;                           // ionization.quiet: accumulator emission, stratified draws
    View1D acc;                                   //   accumulated expected ionization of every particle
    Real qs[7] = {0, 0, 0, 0, 0, 0, 0};           //   random offsets of the R_7 sequence in this slice

    // quiet draws: point j of the R_7 low-discrepancy sequence (Roberts), dimension dim, shifted by qs: the
    // draws of the electrons born in a slice (index j in the slice) fill [0, 1) evenly instead of at random
    KOKKOS_INLINE_FUNCTION Real qr(int dim, uint64_t j) const {
        const Real A[7] = {0.91159235348205492, 0.83100061892695187, 0.75753380995266428, 0.69056202865697702,
                           0.62951106492875597, 0.57385747322139935, 0.52312408457716075};
        const Real v = qs[dim] + Real(j) * A[dim];
        return v - Kokkos::floor(v);
    }

    // electron-impact rate of level l at the particle (per unit xi)
    KOKKOS_INLINE_FUNCTION Real erate(const PGeo& pg, int l) const {
        const int t = P.et0 + l - P.z0;
        Real r = 0;
        for (int b = 0; b < EI_NB; ++b) r += pg.W[0] * eimp(t, 2 * b, pg.j) + pg.W[1] * eimp(t, 2 * b, pg.j + 1);
        return Kokkos::fmax(r, Real(0));
    }
    // electron impact: projectile energy (eV) chosen ~ rate among the bins (u in [0, 1)), its rate-weighted mean
    KOKKOS_INLINE_FUNCTION Real eproj(const PGeo& pg, int l, Real u) const {
        const int t = P.et0 + l - P.z0;
        Real tot = 0;
        for (int b = 0; b < EI_NB; ++b) tot += pg.W[0] * eimp(t, 2 * b, pg.j) + pg.W[1] * eimp(t, 2 * b, pg.j + 1);
        const Real x = u * tot;
        Real c = 0;
        int b = 0;
        for (; b < EI_NB; ++b) {
            const Real r = pg.W[0] * eimp(t, 2 * b, pg.j) + pg.W[1] * eimp(t, 2 * b, pg.j + 1);
            if (r > Real(0) && (c + r > x || b == EI_NB - 1)) {
                return (pg.W[0] * eimp(t, 2 * b + 1, pg.j) + pg.W[1] * eimp(t, 2 * b + 1, pg.j + 1)) / r;
            }
            c += r;
        }
        return Real(0);
    }
    // electron impact on level l: ln W_max = ln((E - Ip)/2) and J of the ejected electron (m c^2)
    KOKKOS_INLINE_FUNCTION void ekin(const PGeo& pg, int l, Real u, Real& lw, Real& Je) const {
        const int t = P.et0 + l - P.z0;
        const Real Ip = eit.par(t, 0);
        const Real E = eproj(pg, l, u);
        lw = Kokkos::log(Kokkos::fmax(Real(0.5) * (E - Ip), Real(1e-6)) / EI_ME_EV);
        Je = eit.par(t, 11) / EI_ME_EV;
    }

    // new macro-electron j: parent velocity + laser drift (field-born electrons in the laser), or
    // + the delta-electron momentum (impact-born, P.delta; lnwm = ln W_max, kap: see Ionization.hpp)
    KOKKOS_INLINE_FUNCTION void electron(int j, Real X, Real Y, Real PX, Real PY, Real D, Real wgt, const IonRate& R,
                                         bool laser_born, Real Aamp, Real Ex, Real Ey, Real Ez, Real EL, int l,
                                         uint64_t i, uint64_t d, int kind, Real lnwm, Real kap, Real pin, Real Je,
                                         uint64_t qj) const {
        // kind: 0 field-born, 1 beam impact (delta electron if P.delta), 2 electron impact (always energy,
        // isotropic; lnwm = ln W_max, Je = J, both in m c^2)
        if (eflag.extent(0) > 0) eflag(j - ne0) = Real(kind);
        if ((kind == 1 && P.delta) || kind == 2) {
            // draws: random (counter-based), or with ionization.quiet the point qj of the R_7 sequence
            auto U = [&](int dim, uint64_t dd) { return quiet ? qr(dim, qj) : ion_rand(key, i, dd); };
            const bool ib = kind == 1;
            const Real J = ib ? P.dJ : Je, wm = Kokkos::exp(lnwm);
            const Real u0 = U(2, d + 4096);
            Real W = -1;
            if (ib && P.nsh > 0 && u0 < pin) {
                // inner subshell (~ n_i/B_i), energy transfer E ~ 1/E^2 on [B_i, W_max], ejected W = E - B_i
                const Real us = u0 / pin;
                int sh = 0;
                while (sh < P.nsh - 1 && us > P.shP[sh]) ++sh;
                const Real B = P.shB[sh];
                if (wm > B) {
                    const Real ue = U(3, d + 4099);
                    W = Real(1) / (Real(1) / B - ue * (Real(1) / B - Real(1) / wm)) - B;
                }
            }
            // otherwise W = J tan(u atan(W_max/J)): exact sampling of dsigma/dW ~ 1/(J^2 + W^2) on [0, W_max]
            if (W < Real(0))
                W = J * Kokkos::tan((ib && P.nsh > 0 ? U(4, d + 4100) : u0) * Kokkos::atan(wm / J));
            const Real u = Kokkos::sqrt(W * (W + Real(2)));
            Real c = (!ib || P.delta_iso) ? Real(2) * U(5, d + 4097) - Real(1) : kap * Kokkos::sqrt(W / (W + Real(2)));
            c = Kokkos::fmin(c, Real(1));
            const Real s = Kokkos::sqrt(Kokkos::fmax(Real(1) - c * c, Real(0)));
            const Real ph = Real(2) * Real(PI) * U(6, d + 4098);
            // parent velocity + delta momentum (the parent moves non-relativistically)
            const Real uzp = (Real(1) + PX * PX + PY * PY - D * D) / (Real(2) * D);
            const Real qx = PX + u * s * Kokkos::cos(ph), qy = PY + u * s * Kokkos::sin(ph), qz = uzp + u * c;
            const Real q2 = Real(1) + qx * qx + qy * qy, gm = Kokkos::sqrt(q2 + qz * qz);
            ex(j) = X; ey(j) = Y; epx(j) = qx; epy(j) = qy; edl(j) = qz > Real(0) ? q2 / (gm + qz) : gm - qz;
            ew(j) = wgt; efresh(j) = 1;
            return;
        }
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
        Real lnwm = 0, kap = 1, pin = 0;   // delta electrons: ln W_max, kappa of the beams at this atom,
        if (P.delta && Rimp > Real(0)) {   // probability of an inner-shell ionization
            const Real n2 = pg.W[0] * imp(kl, pg.j, 2) + pg.W[1] * imp(kl, pg.j + 1, 2);
            if (n2 > Real(0)) {
                lnwm = (pg.W[0] * imp(kl, pg.j, 3) + pg.W[1] * imp(kl, pg.j + 1, 3)) / n2;
                kap = (pg.W[0] * imp(kl, pg.j, 4) + pg.W[1] * imp(kl, pg.j + 1, 4)) / n2;
                pin = Kokkos::fmin(P.s_in * n2 / Rimp, Real(1));
            }
        }
        const Real dtf = dxi * (Real(1) + PX * PX + PY * PY + D * D) / (Real(2) * D * D);   // dxi gamma/Delta
        const uint64_t ii = static_cast<uint64_t>(i);

        // ---- splitting of the first ionization (quantum wq)
        const Real qq = wq(i);
        if (qq > Real(0) && z == P.z0 && wi > qq) {
            const IonRate R = P.field ? ion_field_rate(P, z, Ex, Ey, Ez, EL) : IonRate();
            const Real Re = has_e ? erate(pg, z) : Real(0);
            const Real a = R.W * dtf + (Rimp + Re) * dxi;
            if (!(a > Real(0))) return;
            const Real dN = -wi * Kokkos::expm1(-a);
            Real we = 0;
            if (dN >= qq) we = dN;
            else if (quiet) {
                // a quantum whenever the accumulated dN/qq passes 1 (decided in the count pass; the create pass
                // only sees the particles that emit)
                if (create) we = qq;
                else {
                    const Real an = acc(i) + dN / qq;
                    if (an >= Real(1)) { acc(i) = an - Real(1); we = qq; } else acc(i) = an;
                }
            }
            else if (ion_rand(key, ii, 0) * qq < dN) we = qq;
            if (we <= Real(0)) return;
            if (!create) { cnt_e(i) = 1; cnt_c(i) = 1; return; }
            const int jc = nc0 + cnt_c(i);
            x(jc) = X; y(jc) = Y; px(jc) = PX; py(jc) = PY; dl(jc) = D; w(jc) = we;
            lev(jc) = Real(z + 1); wq(jc) = Real(0); fresh(jc) = 1;
            if (quiet) { const Real v = qs[0] + Real(jc) * Real(0.61803398874989485); acc(jc) = v - Kokkos::floor(v); }
            w(i) = wi - we;
            const uint64_t qj = static_cast<uint64_t>(cnt_e(i));   // index of the electron in this slice
            const Real v = (quiet ? qr(0, qj) : ion_rand(key, ii, 1)) * a;   // which process: field, beam impact, electron impact
            const int kind = v < R.W * dtf ? 0 : (v < R.W * dtf + Rimp * dxi ? 1 : 2);
            Real lw = lnwm, Je = 0;
            if (kind == 2) ekin(pg, z, quiet ? qr(1, qj) : ion_rand(key, ii, 2), lw, Je);
            electron(ne0 + cnt_e(i), X, Y, PX, PY, D, we, R, P.laser && kind == 0, Aamp, Ex, Ey, Ez, EL, z, ii, 1u << 20,
                     kind, lw, kap, pin, Je, qj);
            return;
        }
        // ---- whole macro-particle, possibly several levels in this slice
        int ne = 0;
        // quiet: every particle ionizes when its accumulated optical depth sum(a) passes a threshold tau = -ln(1 - u),
        // u from an interleaved sequence (exact Markov chain per level: the fraction ionized at depth A is 1 - e^-A),
        // a new threshold for the next level; within a slice the depth left over goes on with the next level's rate.
        // acc >= 0: a fresh uniform u (set at loading / birth, or the quantum path's fraction); acc < 0: -1 - (depth
        // still missing). The count pass advances it, the create pass makes the counted electrons.
        const int ne_create = create ? (i + 1 < nion ? cnt_e(i + 1) : tot_e) - cnt_e(i) : 0;
        Real rem = 0, left = 1;   // depth missing to the threshold, fraction of the slice left
        if (quiet && !create) {
            const Real a0 = acc(i);
            rem = a0 >= Real(0) ? -Kokkos::log1p(-Kokkos::fmin(a0, Real(1) - Real(1e-12))) : Real(-1) - a0;
        }
        for (int l = z; l < P.zmax; ++l) {
            const IonRate R = P.field ? ion_field_rate(P, l, Ex, Ey, Ez, EL) : IonRate();
            const Real Ri = l == 0 ? Rimp : Real(0), Re = has_e ? erate(pg, l) : Real(0);
            const Real a = R.W * dtf + (Ri + Re) * dxi;
            if (!(a > Real(0))) break;
            const uint64_t d = 16 + 16 * static_cast<uint64_t>(l);
            if (quiet) {
                if (create) { if (ne >= ne_create) break; }
                else {
                    if (rem > a * left) { rem -= a * left; break; }
                    left -= rem / a;
                    const Real v = qs[0] + Real(i) * Real(0.61803398874989485) + Real(l + 1) * Real(0.75487766624669276);
                    rem = -Kokkos::log1p(-(v - Kokkos::floor(v)));   // threshold of the next level
                }
            } else if (ion_rand(key, ii, d) >= -Kokkos::expm1(-a)) break;
            if (create) {
                const uint64_t qj = static_cast<uint64_t>(cnt_e(i) + ne);
                const Real v = (quiet ? qr(0, qj) : ion_rand(key, ii, d + 1)) * a;   // which process: field, beam impact, electron impact
                const int kind = v < R.W * dtf ? 0 : (v < R.W * dtf + Ri * dxi ? 1 : 2);
                Real lw = lnwm, Je = 0;
                if (kind == 2) ekin(pg, l, quiet ? qr(1, qj) : ion_rand(key, ii, d + 2), lw, Je);
                electron(ne0 + cnt_e(i) + ne, X, Y, PX, PY, D, wi, R, P.laser && kind == 0, Aamp, Ex, Ey, Ez, EL, l, ii,
                         (2u + static_cast<uint64_t>(l)) << 20, kind, lw, kap, pin, Je, qj);
            }
            ++ne;
        }
        if (!create) { cnt_e(i) = ne; if (quiet) acc(i) = Real(-1) - rem; }
        else if (ne > 0) { lev(i) = Real(z + ne); fresh(i) = 1; }
    }
};
} // namespace

IonizeResult PlasmaSpecies::ionize(const SliceFields& f, const View3D& imp, int kl, Real dxi, int k, int step,
                                   PlasmaSpecies& prod, const View3D& eimp, const EITargets& eit) {
    IonizeResult res;
    if (!ionizable_ || Np_ == 0) return res;
    uint64_t nh = 1469598103934665603ULL;   // FNV-1a of the species name
    for (char c : name_) { nh ^= static_cast<unsigned char>(c); nh *= 1099511628211ULL; }
    const uint64_t key = ion_mix(seed_ ^ ion_mix(static_cast<uint64_t>(step) ^ ion_mix(static_cast<uint64_t>(k) ^ ion_mix(nh))));
    const int N = Np_;
    int tot_e = 0, tot_c = 0;
    View1D eflag;
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
        K.eflag = eflag;
        K.eimp = eimp; K.eit = eit; K.has_e = ion_.P.eimp && eimp.extent(0) > 0;
        K.quiet = ion_.quiet; K.acc = acc_;
        for (int dd = 0; dd < 7; ++dd) K.qs[dd] = ion_rand(key, 0x51ce0ffULL, 9000 + dd);
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
    if (ion_.born_dump) eflag = View1D("ion.born_flag", tot_e);
    Kokkos::parallel_for("ion.create", Range(0, N), make(true));
    if (ion_.born_dump) {   // records: slice, x, y, px, py, Delta, w, impact-born
        const std::pair<int, int> rg(prod.Np_, prod.Np_ + tot_e);
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace(), eflag);
        View1D vv[6] = {prod.x_, prod.y_, prod.px_, prod.py_, prod.dl_, prod.w_};
        std::vector<Kokkos::View<Real*, HostSpace>> hv;
        for (auto& v : vv) hv.push_back(Kokkos::create_mirror_view_and_copy(HostSpace(), Kokkos::subview(v, rg)));
        for (int j = 0; j < tot_e; ++j) {
            born_rec_.push_back(k);
            for (auto& h : hv) born_rec_.push_back(h(j));
            born_rec_.push_back(hf(j));
        }
    }
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

// Declustering (Kargapolov, Okhotnikov, Shalimova, Sosedkin & Lotov, Phys. Plasmas 32, 023905 (2025), Sec. IV),
// on the lattice of the rings loaded at the box head (index order = initial radius; neighbours i +- ntheta,
// i.e. the same azimuth column with modes = 1). Per ring: radial displacement d = r - r0, inhomogeneity
// g = d - (d_- + d_+)/2 (one neighbour at the ends: g = d - d_nb), noise part D = g - <g> (<g> over 2 window + 1
// rings), F = 1 - D^2/D0^2 for |D| < D0 (D0 in units of the local cell), else 0, and a radial kick
//   dp_i = -{K h / w_i [c_i a_i - (a_- + a_+)/2] + kappa h F_i (D_i - D_i^prev)} / s^2,   a_i = w_i F_i D_i,
// s = h_min/ppc the smallest ring spacing (the paper's Eq. 32 measures D in lattice spacings: its stability bound
// K + kappa < h^2/(dxi^2 N) is then that of an explicit spring), c_i = (number of neighbours)/2. With the weights w
// (rings: w ~ r) the restoring part is a sum of pair forces:
// sum_i w_i dp_i = 0 up to round-off (Newton's third law; for equal weights it is the paper's Eq. 32).
// A particle that restarted its push history (fresh) starts without damping (D^prev = D).
std::string PlasmaSpecies::decl_description(double dxi) const {
    double hmin = 1e300;
    for (int j = 0; j < grid_.N; ++j) hmin = std::min(hmin, double(grid_.h[j]));
    // explicit spring on the lattice (spacing a = h/ppc, neighbours at 1 .. S steps: largest eigenvalue ~ (2S)^2)
    const double a = hmin / std::max(ppc_, 1), bound = a * a / (dxi * dxi * decl_stride_ * decl_stride_);
    std::ostringstream o;
    o << "Declustering of '" << name_ << "': K = " << decl_K_ << ", kappa = " << decl_kappa_ << ", D0 = " << decl_D0_
      << " cells, <g> over " << 2 * decl_win_ + 1 << " rings; explicit-spring bound (h_min/ppc)^2/(dxi^2 S^2) = " << bound
      << "; stable in practice for K + kappa < ~bound/4 and K, kappa >~ bound/25 (K-kappa scan, ppc 4)"
      << (decl_K_ + decl_kappa_ < 0.25 * bound ? "" : "  ** K + kappa > bound/4: unstable in the scan **");
    return o.str();
}

void PlasmaSpecies::decluster(Real h) {
    if (!decl_ || nfam_ == 0) return;
    const int n = nfam_, s = ntheta_, W = decl_win_, ns = decl_stride_;
    const GridD g = grid_.d;
    const Real K = decl_K_, kap = decl_kappa_, D0 = decl_D0_;
    // spring constant in physical units: 1/a^2 with the smallest ring spacing a = h_min/ppc (on a stretched grid
    // the same restoring frequency everywhere; the clustering rate does not depend on h; stability is set by h_min)
    double hmin = 1e300;
    for (int j = 0; j < grid_.N; ++j) hmin = std::min(hmin, double(grid_.h[j]));
    const Real ia2 = Real(std::max(ppc_, 1) * std::max(ppc_, 1) / (hmin * hmin));
    auto x = x_, y = y_, px = px_, py = py_, w = w_, r0 = r0_, dp = dprev_, G = dg_, Dv = dD_, A = da_;
    auto fresh = fresh_;
    Kokkos::parallel_for("decl.g", Range(0, n), KOKKOS_LAMBDA(int i) {
        auto disp = [&](int q) { return Kokkos::sqrt(x(q) * x(q) + y(q) * y(q)) - r0(q); };
        const Real d = disp(i);
        Real gs = 0;
        for (int t = 1; t <= ns; ++t) {   // neighbours at 1 .. ns lattice steps
            const bool lo = i - t * s >= 0, hi = i + t * s < n;
            gs += (lo && hi) ? d - Real(0.5) * (disp(i - t * s) + disp(i + t * s))
                             : (lo ? d - disp(i - t * s) : (hi ? d - disp(i + t * s) : Real(0)));
        }
        G(i) = gs;
    });
    Real dmax = 0;
    Kokkos::parallel_reduce("decl.D", Range(0, n), KOKKOS_LAMBDA(int i, Real& mx) {
        Real sg = 0; int c = 0;
        for (int q = -W; q <= W; ++q) {
            const int k = i + q * s;
            if (k >= 0 && k < n) { sg += G(k); ++c; }
        }
        const Real D = G(i) - sg / Real(c);
        const Real hl = g.h(g.locate(r0(i)));
        const Real d0 = D0 * hl, F = Kokkos::fabs(D) < d0 ? Real(1) - D * D / (d0 * d0) : Real(0);
        Dv(i) = D;
        A(i) = w(i) * F * D * ia2;
        mx = Kokkos::fmax(mx, Kokkos::fabs(D) / hl);
    }, Kokkos::Max<Real>(dmax));
    decl_dmax_ = dmax;
    Real sum_wk = 0, sum_wa = 0;   // Newton's third law: sum w dp of the restoring part vs sum w |dp|
    Kokkos::parallel_reduce("decl.kick", Range(0, n), KOKKOS_LAMBDA(int i, Real& swk, Real& swa) {
        const Real wi = w(i), D = Dv(i);
        if (wi == Real(0)) { dp(i) = D; return; }
        Real ci = 0, nb = 0;
        for (int t = 1; t <= ns; ++t) {
            const bool lo = i - t * s >= 0, hi = i + t * s < n;
            ci += Real(0.5) * Real((lo ? 1 : 0) + (hi ? 1 : 0));
            nb += Real(0.5) * ((lo ? A(i - t * s) : Real(0)) + (hi ? A(i + t * s) : Real(0)));
        }
        const Real hl = g.h(g.locate(r0(i)));
        const Real d0 = D0 * hl, F = Kokkos::fabs(D) < d0 ? Real(1) - D * D / (d0 * d0) : Real(0);
        const Real Dp = fresh(i) ? D : dp(i);
        const Real kr = -K * h * (ci * A(i) - nb) / wi;   // restoring part (pair forces)
        const Real kick = kr - kap * h * F * (D - Dp) * ia2;
        swk += wi * kr; swa += wi * Kokkos::fabs(kr);
        dp(i) = D;
        const Real X = x(i), Y = y(i), r = Kokkos::sqrt(X * X + Y * Y);
        if (r > Real(0)) { px(i) += kick * X / r; py(i) += kick * Y / r; }
        else px(i) += kick;
    }, sum_wk, sum_wa);
    decl_newton_ = sum_wa > 0 ? std::fabs(sum_wk) / sum_wa : 0.0;
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
