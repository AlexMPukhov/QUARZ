// BeamDiag.cpp: beam moments, slices, packing, dumps and the local particle set (part of the Beam class)
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
        const double xx = m(7) - m(5) * m(5), yy = m(8) - m(6) * m(6);
        const double pxx = m(11) - m(9) * m(9), pyy = m(12) - m(10) * m(10);
        const double xpx = m(13) - m(5) * m(9), ypy = m(14) - m(6) * m(10);
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
