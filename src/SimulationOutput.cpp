// SimulationOutput.cpp: field, axis, beam, laser and ionization output (part of the Simulation class)
#ifndef QUARZ_VERSION
#define QUARZ_VERSION "unknown"
#endif
#include "Simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "SimulationInternal.hpp"

namespace quarz {

using namespace detail;

// ---------------------------------------------------------------------------
// Output format (version 2):
//   int32 version(=2), int32 M, int32 K, int32 mode1, float64 t, float64 r[M], float64 xi[K],
//   int32 ncomp, then ncomp x { char name[16], float64 data[K][M] }
// mode 0: psi ez er eth br bth bz ne ni rhob
// mode 1 (if present): <name>_c, <name>_s  (cos and sin amplitudes: f = f0 + f_c cos + f_s sin)
std::shared_ptr<const FieldTable> Simulation::field_table(const OutputGroup* g) const {
    auto T = std::make_shared<FieldTable>();
    T->KL = box_.nloc;
    T->M = grid_->N + 1;
    // host backends: no copy (the mirror is the array itself); GPU: one host copy
    T->F = Kokkos::create_mirror_view_and_copy(HostSpace(), fld);
    T->D = Kokkos::create_mirror_view_and_copy(HostSpace(), diag);
    T->B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const FieldTable* t = T.get();   // the lambdas below are owned by *t
    auto add = [&](const std::string& name, std::function<double(int, int)> fun) {
        if (!g || g->wants(name)) T->comps.emplace_back(name, fun);
    };
    // W+ = E+ + i B+ ;  polar m=0: V_r0 = Re v1, V_th0 = Im v1
    add("psi", [t](int k, int j) { return t->D(k, j, D_PSI0); });
    add("ez", [t](int k, int j) { return t->F(k, j, F_EZ0); });
    add("er", [t](int k, int j) { return t->F(k, j, F_EP1R); });
    add("eth", [t](int k, int j) { return t->F(k, j, F_EP1I); });
    add("br", [t](int k, int j) { return t->F(k, j, F_BP1R); });
    add("bth", [t](int k, int j) { return t->F(k, j, F_BP1I); });
    add("bz", [t](int k, int j) { return t->F(k, j, F_BZ0); });
    add("ne", [t](int k, int j) { return t->D(k, j, D_NE0); });
    add("ni", [t](int k, int j) { return t->D(k, j, D_NI0); });
    add("rhob", [t](int k, int j) { return t->B(k, j, B_RHO0); });
    if (m1_) {
        // scalars f1 (complex, f = f0 + 2 Re[f1 e^{i th}]):  f_c = 2 Re f1,  f_s = -2 Im f1
        auto scalD = [&](const std::string& n, int cr, int ci) {
            add(n + "_c", [t, cr](int k, int j) { return 2.0 * t->D(k, j, cr); });
            add(n + "_s", [t, ci](int k, int j) { return -2.0 * t->D(k, j, ci); });
        };
        auto scalF = [&](const std::string& n, int cr, int ci) {
            add(n + "_c", [t, cr](int k, int j) { return 2.0 * t->F(k, j, cr); });
            add(n + "_s", [t, ci](int k, int j) { return -2.0 * t->F(k, j, ci); });
        };
        scalD("psi", D_PSI1R, D_PSI1I);
        scalF("ez", F_EZ1R, F_EZ1I);
        scalF("bz", F_BZ1R, F_BZ1I);
        scalD("ne", D_NE1R, D_NE1I);
        scalD("ni", D_NI1R, D_NI1I);
        add("rhob_c", [t](int k, int j) { return 2.0 * t->B(k, j, B_RHO1R); });
        add("rhob_s", [t](int k, int j) { return -2.0 * t->B(k, j, B_RHO1I); });
        // vectors from V+ modes v0, v2:  V_r1c = Re v0 + Re v2,  V_r1s = Im v0 - Im v2,
        //                                V_th1c = Im v0 + Im v2, V_th1s = Re v2 - Re v0
        auto vec = [&](const std::string& nr, const std::string& nt, int c0r, int c0i, int c2r, int c2i) {
            add(nr + "_c", [t, c0r, c2r](int k, int j) { return t->F(k, j, c0r) + t->F(k, j, c2r); });
            add(nr + "_s", [t, c0i, c2i](int k, int j) { return t->F(k, j, c0i) - t->F(k, j, c2i); });
            add(nt + "_c", [t, c0i, c2i](int k, int j) { return t->F(k, j, c0i) + t->F(k, j, c2i); });
            add(nt + "_s", [t, c0r, c2r](int k, int j) { return t->F(k, j, c2r) - t->F(k, j, c0r); });
        };
        vec("er", "eth", F_EP0R, F_EP0I, F_EP2R, F_EP2I);
        vec("br", "bth", F_BP0R, F_BP0I, F_BP2R, F_BP2I);
    }
    for (size_t s = 0; s < ion_sp_.size(); ++s) {   // ion charge density z n of the ionizable species (m = 0)
        const int c = diag_base_ + static_cast<int>(s);
        add("nz_" + species_[ion_sp_[s]]->name(), [t, c](int k, int j) { return t->D(k, j, c); });
        if (m1_) {
            add("nz_" + species_[ion_sp_[s]]->name() + "_c", [](int, int) { return 0.0; });
            add("nz_" + species_[ion_sp_[s]]->name() + "_s", [](int, int) { return 0.0; });
        }
    }
    if (laser_ && (!g || g->wants("a_re"))) {   // laser envelope a^ (m = 0 only)
        const auto& env = laser_->envelope();
        T->L = Kokkos::create_mirror_view_and_copy(HostSpace(), env);
        add("a_re", [t](int k, int j) { return t->L(k, j, 0); });
        add("a_im", [t](int k, int j) { return t->L(k, j, 1); });
        if (m1_)
            for (const char* n : {"a_re_c", "a_re_s", "a_im_c", "a_im_s"}) add(n, [](int, int) { return 0.0; });
    }
    return T;
}

void Simulation::write_fields(const std::string& fn, const OutputGroup* g) const {
    // written: radial nodes 0..M-1 (r <= rmax of the group), slices k = 0, s, 2s, ... (s = xi_stride)
    const int M = g ? g->Mout : grid_->N + 1;
    const int s = g ? g->xi_stride : 1;
    const int K = (box_.nxi - 1) / s + 1;
    const auto T = field_table(g);
    const auto& out = T->comps;
    // layout: header | ncomp x { name[16], data[K][M] };  every rank writes its own rows
    const int KL0 = box_.nloc, k00 = box_.k0;
    std::vector<int> rows;   // local slices that are written
    for (int kl = 0; kl < KL0; ++kl)
        if ((k00 + kl) % s == 0) rows.push_back(kl);
    const int KL = static_cast<int>(rows.size());
    const int k0 = (k00 + s - 1) / s;   // output row of the first written local slice
    const long long hbytes = 4 * 4 + 8 + 8LL * M + 8LL * K + 4;
    const long long cbytes = 16 + 8LL * K * M;
    const long long total = hbytes + static_cast<long long>(out.size()) * cbytes;
    if (comm_.root()) {
        std::vector<char> h(static_cast<size_t>(hbytes));
        char* p = h.data();
        const int32_t hdr[4] = {2, M, K, m1_ ? 1 : 0};
        std::memcpy(p, hdr, sizeof(hdr)); p += sizeof(hdr);
        std::memcpy(p, &t_, 8); p += 8;
        std::memcpy(p, grid_->r.data(), 8 * static_cast<size_t>(M)); p += 8 * static_cast<size_t>(M);
        for (int k = 0; k < K; ++k) { const double x = box_.xi_min + k * s * box_.dxi; std::memcpy(p, &x, 8); p += 8; }
        const int32_t nc = static_cast<int32_t>(out.size());
        std::memcpy(p, &nc, 4);
        Comm::write_at(fn, 0, h.data(), h.size(), total);
        for (size_t c = 0; c < out.size(); ++c) {
            char name[16];
            std::memset(name, 0, sizeof(name));
            std::strncpy(name, out[c].first.c_str(), 15);
            Comm::write_at(fn, hbytes + static_cast<long long>(c) * cbytes, name, 16);
        }
    }
    std::vector<double> v(static_cast<size_t>(KL) * M);
    for (size_t c = 0; c < out.size(); ++c) {
        for (int k = 0; k < KL; ++k)
            for (int j = 0; j < M; ++j) v[static_cast<size_t>(k) * M + j] = out[c].second(rows[k], j);
        Comm::write_at(fn, hbytes + static_cast<long long>(c) * cbytes + 16 + 8LL * M * k0, v.data(), 8 * v.size());
    }
}

void Simulation::write_axis(const std::string& fn) const {
    // fixed-width text so that every rank can write its own lines at the right offset
    const int K = box_.nxi, k0 = box_.k0, KL = box_.nloc;
    auto F = Kokkos::create_mirror_view_and_copy(HostSpace(), fld);
    auto D = Kokkos::create_mirror_view_and_copy(HostSpace(), diag);
    auto B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const double r1 = grid_->r[1];
    const int ncol = m1_ ? 8 : 6;
    const int lw = 18 * ncol;            // "%+.10e " = 18 characters per column, last blank -> '\n'
    const int hw = 256;
    if (comm_.root()) {
        std::string h = "# t = " + std::to_string(t_);
        h.resize(hw / 2 - 1, ' ');
        h += '\n';
        std::string h2 = "# xi  Ez(0)  psi(0)  ne(0)  rho_beam(0)  (Er-Bth)/r|r1";
        if (m1_) h2 += "  Wx(0)  Wy(0)  [W = E_perp + z x B_perp on the axis, from m=1]";
        h2.resize(hw / 2 - 1, ' ');
        h2 += '\n';
        h += h2;
        Comm::write_at(fn, 0, h.data(), h.size(), hw + static_cast<long long>(lw) * K);
    }
    std::string buf;
    buf.reserve(static_cast<size_t>(lw) * KL);
    char cell[32];
    for (int k = 0; k < KL; ++k) {
        double v[8] = {box_.xi_min + (k0 + k) * box_.dxi, F(k, 0, F_EZ0), D(k, 0, D_PSI0), D(k, 0, D_NE0),
                       B(k, 0, B_RHO0), (F(k, 1, F_EP1R) - F(k, 1, F_BP1I)) / r1, 0, 0};
        if (m1_) { v[6] = F(k, 0, F_EP0R) - F(k, 0, F_BP0I); v[7] = F(k, 0, F_EP0I) + F(k, 0, F_BP0R); }
        for (int c = 0; c < ncol; ++c) {
            std::snprintf(cell, sizeof(cell), "%+.10e%c", v[c], c == ncol - 1 ? '\n' : ' ');
            buf += cell;
        }
    }
    Comm::write_at(fn, hw + static_cast<long long>(lw) * k0, buf.data(), buf.size());
}

// plasma particles that reached the quasi-static limit in this sweep become beam particles:
// state of the slice where the limit was reached, weight = plasma weight (particles per unit
// length) x the time this step represents; with modes = 0 at a random azimuth (nphi copies)
void Simulation::convert_trapped(int n) {
    trapped_step_ = 0;
    const double dtw = have_prev_ ? 0.5 * (dt_prev_ + cur_dt_) : cur_dt_;
    const double dtp = have_prev_ ? dt_prev_ : cur_dt_;
    for (size_t s = 0; s < species_.size(); ++s) {
        if (trap_beam_[s] < 0) continue;
        std::vector<double> rec;
        const int nt = species_[s]->collect_trapped(rec);
        if (nt == 0) continue;
        trapped_step_ += nt;
        const int nphi = species_[s]->trap_nphi();
        std::vector<double> p;
        p.reserve(static_cast<size_t>(7 * nt * nphi));
        for (int i = 0; i < nt; ++i) {
            const double* r = &rec[7 * static_cast<size_t>(i)];
            const double xi = xi_head_ + r[5], w = r[6] * dtw / nphi;
            if (m1_) { p.insert(p.end(), {r[0], r[1], r[2], r[3], r[4], xi, w}); continue; }
            // random azimuth from a hash of the state (independent of the particle index and the rank)
            union { double d; uint64_t u; } b0{r[0]}, b1{r[1]}, b2{r[2]}, b3{r[3]}, b4{r[4]}, b5{xi};
            const uint64_t key = ion_mix(0x7a3dULL ^ ion_mix(static_cast<uint64_t>(n) + 1000003ULL * (s + 1)));
            const uint64_t id = ion_mix(b0.u ^ ion_mix(b1.u ^ ion_mix(b2.u ^ ion_mix(b3.u ^ ion_mix(b4.u ^ ion_mix(b5.u))))));
            const double ph0 = 2 * PI * ion_rand(key, id, 0);
            for (int j = 0; j < nphi; ++j) {
                const double ph = ph0 + 2 * PI * j / nphi, c = std::cos(ph), sn = std::sin(ph);
                p.insert(p.end(), {c * r[0] - sn * r[1], sn * r[0] + c * r[1], c * r[2] - sn * r[3], sn * r[2] + c * r[3],
                                   r[4], xi, w});
            }
        }
        beams_[trap_beam_[s]]->add_new(p, fld, static_cast<Real>(dtp), laser_ ? laser_->ponderomotive() : View3D());
    }
}

void Simulation::write_beam_output(int n) {
    const bool par = comm_.size() > 1;
    const std::string rk = ".part" + std::to_string(comm_.rank());
    char step[16];
    std::snprintf(step, sizeof(step), "%06d", n);
    for (const auto& b : beams_) {
        const auto S = b->sums();
        if (!par) {
            const BeamDiag d = b->from_sums(S, t_);
            beamlog_ << b->name() << " " << n << " " << std::setprecision(10) << d.t << " " << d.alive << " "
                     << d.npart << " " << d.charge << " " << d.gamma_mean << " " << d.gamma_rms << " " << d.xi_mean
                     << " " << d.xi_rms << " " << d.r_rms << " " << d.emit_nx << " " << d.x_mean << " " << d.y_mean
                     << " " << d.emit_ny << " " << d.sigma_x << " " << d.sigma_y << "\n";
        } else {
            beamlog_ << b->name() << " " << n << " " << std::setprecision(17) << t_;
            for (double x : S) beamlog_ << " " << x;
            beamlog_ << "\n";
        }
        if (b->spin()) {
            const auto P4 = b->spin_sums();
            if (!par) {
                const double w = P4[0] > 0 ? P4[0] : 1;
                const double mx = P4[1] / w, my = P4[2] / w, mz = P4[3] / w;
                spinlog_ << b->name() << " " << n << " " << std::setprecision(10) << t_ << " " << std::setprecision(15)
                         << mx << " " << my << " " << mz << " " << std::sqrt(mx * mx + my * my + mz * mz) << "\n";
            } else {
                spinlog_ << b->name() << " " << n << " " << std::setprecision(17) << t_;
                for (double x : P4) spinlog_ << " " << x;
                spinlog_ << "\n";
            }
        }
        if (slice_bins_ > 0 && out_every_ > 0 && n % out_every_ == 0) {
            const auto ss = b->slice_sums(slice_lo_, slice_hi_, slice_bins_);
            const std::string fn = outdir_ + "/slices_" + b->name() + "_" + step + ".txt";
            if (!par) Beam::write_slices_file(fn, t_, b->charge(), slice_lo_, slice_hi_, slice_bins_, ss);
            else {
                std::ofstream o(fn + rk, std::ios::binary);
                o.write(reinterpret_cast<const char*>(&t_), 8);
                o.write(reinterpret_cast<const char*>(ss.data()), 8 * ss.size());
            }
        }
    }
    beamlog_.flush();
    if (spinlog_.is_open()) spinlog_.flush();
}

// combine the per-rank partial outputs of a parallel run (rank 0, after the run)
void Simulation::merge_partial_outputs() {
    const int P = comm_.size();
    namespace fs = std::filesystem;
    // laser.txt: sums over the ranks of every step
    if (laser_) {
        std::map<int, std::pair<double, std::array<double, 5>>> acc;
        for (int r = 0; r < P; ++r) {
            const std::string fn = outdir_ + "/laser.part" + std::to_string(r);
            std::ifstream in(fn);
            int n;
            double t;
            std::array<double, 5> S;
            while (in >> n >> t >> S[0] >> S[1] >> S[2] >> S[3] >> S[4]) {
                auto it = acc.find(n);
                if (it == acc.end()) { acc[n] = {t, S}; continue; }
                auto& A = it->second.second;
                A[0] += S[0]; A[1] += S[1]; A[2] += S[2]; A[3] = std::max(A[3], S[3]); A[4] += S[4];
            }
            in.close();
            fs::remove(fn);
        }
        const bool hdr = !restarted_ || !fs::exists(outdir_ + "/laser.txt");
        std::ofstream o(outdir_ + "/laser.txt", restarted_ ? std::ios::app : std::ios::trunc);
        if (hdr) o << "# step t a_max w_rms xi_centroid L_rms energy (int |a|^2 dV)\n";
        for (const auto& e : acc) {
            const auto& S = e.second.second;
            const double E = S[0] > 0 ? S[0] : 1;
            const double xc = S[1] / E, L = std::sqrt(std::max(0.0, S[4] / E - xc * xc)), w = std::sqrt(2 * S[2] / E);
            o << e.first << " " << std::setprecision(10) << e.second.first << " " << S[3] << " " << w << " " << xc << " "
              << L << " " << S[0] << "\n";
        }
    }
    // ionization.txt: sums over the ranks
    if (!ion_sp_.empty()) {
        const size_t ns = ion_sp_.size();
        std::map<int, std::pair<double, std::vector<std::array<double, 4>>>> acc;
        for (int r = 0; r < P; ++r) {
            const std::string fn = outdir_ + "/ionization.part" + std::to_string(r);
            std::ifstream in(fn);
            int n;
            double t;
            while (in >> n >> t) {
                std::vector<std::array<double, 4>> A(ns);
                for (auto& a : A) for (double& x : a) in >> x;
                auto it = acc.find(n);
                if (it == acc.end()) { acc[n] = {t, A}; continue; }
                for (size_t s = 0; s < ns; ++s) for (int c = 0; c < 4; ++c) it->second.second[s][c] += A[s][c];
            }
            in.close();
            fs::remove(fn);
        }
        const bool hdr = !restarted_ || !fs::exists(outdir_ + "/ionization.txt");
        std::ofstream o(outdir_ + "/ionization.txt", restarted_ ? std::ios::app : std::ios::trunc);
        if (hdr) {
            o << "# step t";
            for (int i : ion_sp_) {
                const std::string& nm = species_[i]->name();
                o << "  " << nm << ":born_weight " << nm << ":born_per_m " << nm << ":born_p2 " << nm << ":z_tail";
            }
            o << ION_LOG_NOTE;
        }
        const double n0 = cfg_.get_double("units.n0_cm3") * 1e6;
        const double kpi = 2.99792458e8 / (5.64146e4 * std::sqrt(n0 * 1e-6));
        for (const auto& e : acc) ion_line(o, e.first, e.second.first, e.second.second, n0 * kpi * kpi);
    }
    // beams.txt: sum the partial moments of every (beam, step)
    {
        std::map<std::pair<std::string, int>, std::pair<double, std::array<double, 17>>> acc;
        std::vector<std::pair<std::string, int>> order;
        for (int r = 0; r < P; ++r) {
            std::ifstream in(outdir_ + "/beams.part" + std::to_string(r));
            std::string name;
            int n;
            double t;
            while (in >> name >> n >> t) {
                std::array<double, 17> S;
                for (auto& x : S) in >> x;
                auto key = std::make_pair(name, n);
                auto it = acc.find(key);
                if (it == acc.end()) { acc[key] = {t, S}; order.push_back(key); }
                else for (int c = 0; c < 17; ++c) it->second.second[c] += S[c];
            }
            fs::remove(outdir_ + "/beams.part" + std::to_string(r));
        }
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second < b.second : a.first < b.first; });
        const bool hdr = !restarted_ || !fs::exists(outdir_ + "/beams.txt");
        std::ofstream o(outdir_ + "/beams.txt", restarted_ ? std::ios::app : std::ios::trunc);
        if (hdr)
            o << "# beam step t alive npart charge gamma_mean gamma_rms xi_mean xi_rms r_rms emit_nx"
                 " x_mean y_mean emit_ny sigma_x sigma_y\n";
        std::map<std::string, const Beam*> beam;
        for (const auto& b : beams_) beam[b->name()] = b.get();
        for (const auto& key : order) {
            const auto& e = acc[key];
            const BeamDiag d = beam[key.first]->from_sums(e.second, e.first);
            o << key.first << " " << key.second << " " << std::setprecision(10) << d.t << " " << d.alive << " "
              << d.npart << " " << d.charge << " " << d.gamma_mean << " " << d.gamma_rms << " " << d.xi_mean << " "
              << d.xi_rms << " " << d.r_rms << " " << d.emit_nx << " " << d.x_mean << " " << d.y_mean << " "
              << d.emit_ny << " " << d.sigma_x << " " << d.sigma_y << "\n";
        }
    }
    // spin.txt: sum the partial spin sums
    if (fs::exists(outdir_ + "/spin.part0")) {
        std::map<std::pair<std::string, int>, std::pair<double, std::array<double, 4>>> acc;
        for (int r = 0; r < P; ++r) {
            std::ifstream in(outdir_ + "/spin.part" + std::to_string(r));
            std::string name;
            int n;
            double t;
            while (in >> name >> n >> t) {
                std::array<double, 4> S;
                for (auto& x : S) in >> x;
                auto key = std::make_pair(name, n);
                auto it = acc.find(key);
                if (it == acc.end()) acc[key] = {t, S};
                else for (int c = 0; c < 4; ++c) it->second.second[c] += S[c];
            }
            fs::remove(outdir_ + "/spin.part" + std::to_string(r));
        }
        std::vector<std::pair<std::string, int>> order;
        for (const auto& e : acc) order.push_back(e.first);
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second < b.second : a.first < b.first; });
        const bool hdr = !restarted_ || !fs::exists(outdir_ + "/spin.txt");
        std::ofstream o(outdir_ + "/spin.txt", restarted_ ? std::ios::app : std::ios::trunc);
        if (hdr) o << "# beam step t <s_x> <s_y> <s_z> P = |<s>| (weighted means over live particles)\n";
        for (const auto& key : order) {
            const auto& e = acc[key];
            const double w = e.second[0] > 0 ? e.second[0] : 1;
            const double mx = e.second[1] / w, my = e.second[2] / w, mz = e.second[3] / w;
            o << key.first << " " << key.second << " " << std::setprecision(10) << e.first << " " << std::setprecision(15)
              << mx << " " << my << " " << mz << " " << std::sqrt(mx * mx + my * my + mz * mz) << "\n";
        }
    }
    // slices and particle dumps (main output directory and the diagnostic groups)
    std::vector<std::string> dirs;
    for (const auto& g : groups_)
        if (std::find(dirs.begin(), dirs.end(), g.dir) == dirs.end()) dirs.push_back(g.dir);
    for (const auto& d : dirs)
    for (const auto& entry : fs::directory_iterator(d)) {
        const std::string fn = entry.path().string();
        if (fn.size() < 7 || fn.substr(fn.size() - 6) != ".part0") continue;
        const std::string base = fn.substr(0, fn.size() - 6);
        const std::string leaf = entry.path().filename().string();
        if (leaf.rfind("slices_", 0) == 0) {
            std::vector<double> sum;
            double t = 0;
            for (int r = 0; r < P; ++r) {
                std::ifstream in(base + ".part" + std::to_string(r), std::ios::binary);
                in.seekg(0, std::ios::end);
                const size_t nb = static_cast<size_t>(in.tellg());
                in.seekg(0);
                std::vector<double> v(nb / 8);
                in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(nb));
                t = v[0];
                if (sum.empty()) sum.assign(v.begin() + 1, v.end());
                else for (size_t i = 1; i < v.size(); ++i) sum[i - 1] += v[i];
                fs::remove(base + ".part" + std::to_string(r));
            }
            std::string bname = leaf.substr(7);
            bname = bname.substr(0, bname.rfind('_'));
            Real q = -1;
            for (const auto& b : beams_) if (b->name() == bname) q = b->charge();
            Beam::write_slices_file(base, t, q, slice_lo_, slice_hi_, slice_bins_, sum);
        } else if (leaf.rfind("beam_", 0) == 0) {
            std::string bname = leaf.substr(5);
            bname = bname.substr(0, bname.rfind('_'));
            size_t nc = 7;
            for (const auto& b : beams_) if (b->name() == bname) nc = static_cast<size_t>(b->ncomp());
            std::vector<double> all;
            for (int r = 0; r < P; ++r) {
                std::ifstream in(base + ".part" + std::to_string(r), std::ios::binary);
                int32_t n = 0;
                in.read(reinterpret_cast<char*>(&n), 4);
                std::vector<double> v(nc * static_cast<size_t>(n));
                in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(8 * v.size()));
                all.insert(all.end(), v.begin(), v.end());
                fs::remove(base + ".part" + std::to_string(r));
            }
            std::ofstream o(base, std::ios::binary);
            const int32_t n = static_cast<int32_t>(all.size() / nc);
            o.write(reinterpret_cast<const char*>(&n), 4);
            o.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(8 * all.size()));
        }
    }
}

void Simulation::write_laser_diag(int n) {
    const auto S = laser_->sums();
    if (comm_.size() == 1) {
        const double e = S[0] > 0 ? S[0] : 1;
        const double xc = S[1] / e, L = std::sqrt(std::max(0.0, S[4] / e - xc * xc)), w = std::sqrt(2 * S[2] / e);
        laserlog_ << n << " " << std::setprecision(10) << t_ << " " << S[3] << " " << w << " " << xc << " " << L << " "
                  << S[0] << "\n";
    } else {
        laserlog_ << n << " " << std::setprecision(17) << t_;
        for (double x : S) laserlog_ << " " << x;
        laserlog_ << "\n";
    }
    laserlog_.flush();
}

void Simulation::write_ion_diag(int n) {
    if (comm_.size() == 1) {
        const double n0 = cfg_.get_double("units.n0_cm3") * 1e6;                 // m^-3
        const double kpi = 2.99792458e8 / (5.64146e4 * std::sqrt(n0 * 1e-6));    // c/omega_p, m
        ion_line(ionlog_, n, t_, ion_acc_, n0 * kpi * kpi);
    } else {
        ionlog_ << n << " " << std::setprecision(17) << t_;
        for (const auto& a : ion_acc_) for (double x : a) ionlog_ << " " << x;
        ionlog_ << "\n";
    }
    ionlog_.flush();
}

} // namespace quarz
