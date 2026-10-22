#include "PlasmaSpecies.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include "PlasmaKernels.hpp"

namespace quarz {

using namespace detail;

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
    trap_ = cfg.get_bool(name + ".trap", false);
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
    if (ionizable_) {
        lev_ = View1D(name + ".z", cap_);
        wq_ = View1D(name + ".wq", cap_);
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
    npush_ = 0;
    xi_cur_ = 0;
    if (ionizable_) {
        Kokkos::deep_copy(lev_, Real(ion_.P.z0));
        auto wq = wq_; auto w = w_;
        const Real f = ion_.nsplit > 1 ? Real(1) / ion_.nsplit : Real(0);
        Kokkos::parallel_for("ion.init_wq", Range(0, cap_), KOKKOS_LAMBDA(int i) { wq(i) = w(i) * f; });
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
        Kokkos::resize(cnt_e_, nc);
        Kokkos::resize(cnt_c_, nc);
    }
    cap_ = nc;
}

// [Np][x y px py Delta w fresh (z wq)][hist slabs], each array the first Np entries
void PlasmaSpecies::pack_state(std::vector<double>& buf) const {
    buf.push_back(static_cast<double>(Np_));
    std::vector<View1D> vs = {x_, y_, px_, py_, dl_, w_};
    if (ionizable_) { vs.push_back(lev_); vs.push_back(wq_); }
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
    if (ionizable_) { vs.push_back(lev_); vs.push_back(wq_); }
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

} // namespace quarz
