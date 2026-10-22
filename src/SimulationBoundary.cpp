// SimulationBoundary.cpp: two-stage runs: the boundary file between the stages
#include "Simulation.hpp"
#include "Beam.hpp"
#include "Config.hpp"
#include "Laser.hpp"
#include "OpenPMDWriter.hpp"
#include "RadialGrid.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include "SimulationIOInternal.hpp"

namespace quarz {

namespace fs = std::filesystem;
using namespace detail;

// ============================================================================ two-stage runs
// File: "QZBOUND1", int32 version 1, int32 M, int32 m1, int32 nmobile, then per mobile species
// int32 ionizable, int32 ab_order; float64 dxi, xi_head (stage-1 xi.min), xi_next (first slice of
// stage 2), float64 r[M]; records: float64 t, int64 step, int64 len, float64 max |rho_b| of the
// last slice, float64 data[len] (mobile species pack_state ..., B+ 6 x M).
namespace {
constexpr char BMAGIC[8] = {'Q', 'Z', 'B', 'O', 'U', 'N', 'D', '1'};
template <class T> void bput(std::FILE* f, const T& v) { std::fwrite(&v, sizeof(T), 1, f); }
template <class T> T bget(std::FILE* f) {
    T v{};
    if (std::fread(&v, sizeof(T), 1, f) != 1) throw std::runtime_error("boundary file: unexpected end");
    return v;
}
} // namespace

void Simulation::open_boundary_write() {
    if (laser_) throw std::runtime_error("boundary.write: not available with a laser");
    const int M = grid_->N + 1;
    const bool append = restarted_ && fs::exists(bwrite_path_);
    if (append) {
        // keep the records of the steps before the restart
        std::FILE* f = std::fopen(bwrite_path_.c_str(), "rb");
        char mg[8];
        if (!f || std::fread(mg, 1, 8, f) != 8 || std::memcmp(mg, BMAGIC, 8) != 0)
            throw std::runtime_error("boundary.write: " + bwrite_path_ + " is not a boundary file");
        bget<int32_t>(f); bget<int32_t>(f); bget<int32_t>(f);
        const int ns = bget<int32_t>(f);
        for (int i = 0; i < 2 * ns; ++i) bget<int32_t>(f);
        for (int i = 0; i < 3 + M; ++i) bget<double>(f);
        long long keep = std::ftell(f);
        while (true) {
            const long long pos = std::ftell(f);
            double t; int64_t st, len; double rb;
            if (std::fread(&t, 8, 1, f) != 1 || std::fread(&st, 8, 1, f) != 1 || std::fread(&len, 8, 1, f) != 1 ||
                std::fread(&rb, 8, 1, f) != 1) break;
            if (st >= start_step_) break;
            if (std::fseek(f, static_cast<long>(8 * len), SEEK_CUR) != 0) break;
            keep = pos + 32 + 8 * len;
        }
        std::fclose(f);
        fs::resize_file(bwrite_path_, static_cast<uintmax_t>(keep));
        bwf_ = std::fopen(bwrite_path_.c_str(), "ab");
    } else {
        bwf_ = std::fopen(bwrite_path_.c_str(), "wb");
        if (!bwf_) throw std::runtime_error("boundary.write: cannot open " + bwrite_path_);
        std::fwrite(BMAGIC, 1, 8, bwf_);
        bput<int32_t>(bwf_, 1);
        bput<int32_t>(bwf_, M);
        bput<int32_t>(bwf_, m1_ ? 1 : 0);
        int ns = 0;
        for (auto& s : species_) if (s->mobile()) ++ns;
        bput<int32_t>(bwf_, ns);
        for (auto& s : species_)
            if (s->mobile()) { bput<int32_t>(bwf_, s->ionizable() ? 1 : 0); bput<int32_t>(bwf_, s->ab_order()); }
        bput<double>(bwf_, box_.dxi);
        bput<double>(bwf_, xi_head_);
        bput<double>(bwf_, box_.xi_min + box_.nxi * box_.dxi);
        for (int j = 0; j < M; ++j) bput<double>(bwf_, grid_->r[j]);
    }
    if (!bwf_) throw std::runtime_error("boundary.write: cannot open " + bwrite_path_);
    std::cout << "Boundary: the plasma state leaving the box (xi = " << box_.xi_min + box_.nxi * box_.dxi
              << ") is written every step to " << bwrite_path_ << std::endl;
}

void Simulation::write_boundary(int n) {
    std::vector<double> d;
    for (auto& sp : species_)
        if (sp->mobile()) sp->pack_state(d);
    const int M = grid_->N + 1;
    for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), v);
        d.insert(d.end(), h.data(), h.data() + M);
    }
    // beam charge in the last slice: stage 2 has no beam upstream, so it should be ~0 there
    std::vector<double> g;
    get_last_rhot(g);
    double rb = 0;
    for (int j = 0; j < M; ++j) rb = std::max(rb, std::abs(g[static_cast<size_t>(B_NC1) * j + B_RT0]));
    bput<double>(bwf_, t_);
    bput<int64_t>(bwf_, n);
    bput<int64_t>(bwf_, static_cast<int64_t>(d.size()));
    bput<double>(bwf_, rb);
    std::fwrite(d.data(), 8, d.size(), bwf_);
    std::fflush(bwf_);
}

void Simulation::open_boundary_read() {
    if (laser_) throw std::runtime_error("boundary.read: not available with a laser");
    brf_ = std::fopen(bread_path_.c_str(), "rb");
    char mg[8];
    if (!brf_ || std::fread(mg, 1, 8, brf_) != 8 || std::memcmp(mg, BMAGIC, 8) != 0)
        throw std::runtime_error("boundary.read: " + bread_path_ + " is not a boundary file");
    bget<int32_t>(brf_);
    const int M = bget<int32_t>(brf_), m1 = bget<int32_t>(brf_), ns = bget<int32_t>(brf_);
    std::vector<std::pair<int, int>> sp;
    for (int i = 0; i < ns; ++i) { const int a = bget<int32_t>(brf_); const int b = bget<int32_t>(brf_); sp.emplace_back(a, b); }
    const double dxi = bget<double>(brf_), head = bget<double>(brf_), next = bget<double>(brf_);
    std::vector<double> r(M);
    for (int j = 0; j < M; ++j) r[j] = bget<double>(brf_);
    // the two stages must fit together
    std::vector<std::pair<int, int>> mine;
    for (auto& s : species_) if (s->mobile()) mine.emplace_back(s->ionizable() ? 1 : 0, s->ab_order());
    bool gridok = M == grid_->N + 1;
    for (int j = 0; gridok && j < M; ++j) gridok = std::abs(r[j] - grid_->r[j]) <= 1e-12 * (1 + std::abs(r[j]));
    if (!gridok) throw std::runtime_error("boundary.read: the radial grid differs from the stage-1 grid");
    if (m1 != (m1_ ? 1 : 0)) throw std::runtime_error("boundary.read: 'modes' differs from stage 1");
    if (mine != sp) throw std::runtime_error("boundary.read: the mobile plasma species (or pusher.ab_order) differ from stage 1");
    if (std::abs(dxi - box_.dxi) > 1e-12 * dxi) throw std::runtime_error("boundary.read: xi.step differs from stage 1");
    if (std::abs(box_.xi_min - next) > 1e-9 * (1 + std::abs(next)))
        throw std::runtime_error("boundary.read: xi.min must be " + std::to_string(next) + " (the slice after the stage-1 box)");
    xi_head_ = head;
    // index of the records
    while (true) {
        BRec R;
        if (std::fread(&R.t, 8, 1, brf_) != 1) break;
        int64_t st, len;
        if (std::fread(&st, 8, 1, brf_) != 1 || std::fread(&len, 8, 1, brf_) != 1 || std::fread(&R.rhob, 8, 1, brf_) != 1) break;
        R.step = st; R.len = len; R.off = std::ftell(brf_);
        if (std::fseek(brf_, static_cast<long>(8 * len), SEEK_CUR) != 0) break;
        bindex_.push_back(R);
    }
    if (bindex_.empty()) throw std::runtime_error("boundary.read: no records in " + bread_path_);
    double rbmax = 0;
    for (const auto& R : bindex_) rbmax = std::max(rbmax, R.rhob);
    if (comm_.root()) {
        std::cout << "Boundary: plasma entering at xi = " << box_.xi_min << " from " << bread_path_ << " ("
                  << bindex_.size() << " steps, t = " << bindex_.front().t << " ... " << bindex_.back().t
                  << "); density profile at the stage-1 head xi = " << head << "\n";
        if (rbmax > 1e-6)
            std::cout << "WARNING: the stage-1 beams reach its last slice (max |rho_b| = " << rbmax
                      << "); stage 2 does not see them\n";
    }
}

std::vector<char> Simulation::boundary_mask(const std::vector<double>& d) const {
    // 1 = interpolate in t (particle coordinates, momenta, weights, AB history, B+); 0 = take from
    // the nearer record (counts, flags, charge states, xi bookkeeping)
    std::vector<char> m(d.size(), 0);
    size_t o = 0;
    for (const auto& s : species_) {
        if (!s->mobile()) continue;
        const size_t np = static_cast<size_t>(d[o]);
        ++o;
        std::fill(m.begin() + o, m.begin() + o + 6 * np, 1);
        o += 6 * np;
        if (s->ionizable()) o += 2 * np;   // charge state, split quantum
        o += np;                           // fresh flags
        const size_t nh = static_cast<size_t>(s->ab_order()) * 5 * np;
        std::fill(m.begin() + o, m.begin() + o + nh, 1);
        o += nh;
        o += 2 + s->ab_order();
    }
    std::fill(m.begin() + o, m.end(), 1);   // B+
    return m;
}

void Simulation::boundary_state(double t) {
    auto read = [&](size_t i) {
        std::vector<double> d(static_cast<size_t>(bindex_[i].len));
        std::fseek(brf_, static_cast<long>(bindex_[i].off), SEEK_SET);
        if (std::fread(d.data(), 8, d.size(), brf_) != d.size()) throw std::runtime_error("boundary.read: short record");
        return d;
    };
    const double eps = 1e-9 * std::max(1.0, std::abs(t));
    if (t < bindex_.front().t - eps || t > bindex_.back().t + eps)
        throw std::runtime_error("boundary.read: t = " + std::to_string(t) + " outside the stage-1 records (" +
                                 std::to_string(bindex_.front().t) + " ... " + std::to_string(bindex_.back().t) + ")");
    size_t i = 0;
    while (i + 1 < bindex_.size() && bindex_[i + 1].t <= t + eps) ++i;
    if (std::abs(bindex_[i].t - t) <= eps || i + 1 == bindex_.size()) { bbuf_ = read(i); return; }
    const double w = (t - bindex_[i].t) / (bindex_[i + 1].t - bindex_[i].t);
    std::vector<double> a = read(i), b = read(i + 1);
    const std::vector<char> ma = boundary_mask(a), mb = boundary_mask(b);
    if (a.size() != b.size() || ma != mb) {   // different particle numbers (ionization): nearer record
        if (!bwarned_ && comm_.root())
            std::cout << "boundary.read: particle numbers change between records; using the nearer record (no interpolation)\n";
        bwarned_ = true;
        bbuf_ = w < 0.5 ? a : b;
        return;
    }
    bbuf_.resize(a.size());
    for (size_t k = 0; k < a.size(); ++k) bbuf_[k] = ma[k] ? a[k] + w * (b[k] - a[k]) : (w < 0.5 ? a[k] : b[k]);
}

} // namespace quarz
