// SimulationCheckpoint.cpp: checkpoints and restarts (layout in SimulationIO.cpp)
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

// ============================================================================ checkpoints

void Simulation::write_checkpoint(int N) {
    const std::string dir = chk_dir_ + "/chk_" + step_name(N);
    std::error_code ec;
    fs::create_directories(dir, ec);   // every rank (they reach this point at different times)
    if (!fs::is_directory(dir)) throw std::runtime_error("cannot create checkpoint directory " + dir);
    const int P = comm_.size(), rank = comm_.rank();
    if (comm_.root()) {
        std::ofstream m(dir + "/checkpoint.txt");
        m << "step " << N << "\nt " << std::setprecision(17) << t_ << "\nranks " << P << "\nversion " << CHK_VERSION << "\n";
        m << "# restart with:  quarz <deck> restart.from=" << dir << "   (or restart.from=latest)\n# deck:\n";
        for (const auto& k : cfg_.keys()) m << "#   " << k << " = " << cfg_.get_string(k) << "\n";
    }
    const std::string fn = dir + "/rank_" + std::to_string(rank) + ".bin";
    {
        Writer W;
        W.o.open(fn + ".tmp", std::ios::binary | std::ios::trunc);
        if (!W.o) throw std::runtime_error("cannot write checkpoint file " + fn);
        W.put(CHK_MAGIC);
        W.put(CHK_VERSION);
        for (int v : {P, rank, N, box_.nxi, box_.k0, box_.nloc, grid_->N + 1, m1_ ? 1 : 0,
                      static_cast<int>(beams_.size()), laser_ ? 1 : 0})
            W.put(static_cast<int32_t>(v));
        W.put(static_cast<double>(t_));
        W.put(static_cast<double>(box_.dxi));
        W.put(static_cast<double>(box_.xi_min));
        W.put(static_cast<int64_t>(lost_total_));
        for (size_t b = 0; b < beams_.size(); ++b) {
            W.put(beams_[b]->name());
            W.put(static_cast<int32_t>(beams_[b]->started() ? 1 : 0));
            W.put(beams_[b]->packed_all());
            W.put(outbox_[b]);
        }
        if (laser_) {
            auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), laser_->current());
            std::vector<double> v(h.data(), h.data() + h.size());
            W.put(v);
        }
        W.o.close();
        if (!W.o) throw std::runtime_error("error writing checkpoint file " + fn);
    }
    if (adaptive_) write_adapt_state(dir);   // before the rank file: complete = all rank files
    fs::rename(fn + ".tmp", fn);
    if (comm_.root()) {
        std::cout << "checkpoint " << dir << " (step " << N << ", t = " << t_ << ")" << std::endl;
        prune_checkpoints();
    }
}

// adaptive time step: dt of the last step, t history and this rank's gamma records
void Simulation::write_adapt_state(const std::string& dir) const {
    const std::string fn = dir + "/adapt_rank_" + std::to_string(comm_.rank()) + ".txt";
    std::ofstream o(fn + ".tmp");
    o << std::setprecision(17) << "dt_prev " << dt_prev_ << "\ngmin_init " << gmin_init_ << "\n";
    for (const auto& [k, t] : thist_) o << "t " << k << " " << t << "\n";
    for (const auto& [m, g] : lrec_) o << "rec " << m << " " << g << "\n";
    o.close();
    if (!o) throw std::runtime_error("cannot write " + fn);
    fs::rename(fn + ".tmp", fn);
}

void Simulation::read_adapt_state(const std::string& dir) {
    int ranks = 0;
    {
        std::ifstream meta(dir + "/checkpoint.txt");
        std::string key;
        while (meta >> key) if (key == "ranks") meta >> ranks;
    }
    auto read = [&](int r, bool own) {
        const std::string fn = dir + "/adapt_rank_" + std::to_string(r) + ".txt";
        std::ifstream in(fn);
        if (!in) throw std::runtime_error("restart with time.adaptive: " + fn + " missing (checkpoint written without time.adaptive?)");
        std::map<int, double> rec;
        std::string key;
        while (in >> key) {
            if (key == "dt_prev") { double v; in >> v; if (own) { dt_prev_ = v; have_prev_ = true; } }
            else if (key == "gmin_init") { double v; in >> v; if (own) gmin_init_ = v; }
            else if (key == "t") { int k; double v; in >> k >> v; if (own) thist_[k] = v; }
            else if (key == "rec") { int m; double v; in >> m >> v; rec[m] = v; }
        }
        return rec;
    };
    const int me = comm_.rank() < ranks ? comm_.rank() : 0;
    read(me, true);
    grec_.clear();
    if (comm_.root()) {
        grec_[thist_.begin()->first - 1] = gmin_init_;
        for (int r = 0; r < ranks; ++r) {
            const auto rec = read(r, false);
            for (const auto& [m, g] : rec) {
                if (m >= start_step_) continue;
                auto it = grec_.find(m);
                grec_[m] = (it == grec_.end()) ? g : std::min(it->second, g);
            }
        }
    } else grec_[start_step_ - 1] = gmin_init_;
    thist_[start_step_] = t_;
}

void Simulation::prune_checkpoints() {
    if (chk_keep_ <= 0) return;
    const auto done = complete_checkpoints(chk_dir_);
    if (static_cast<int>(done.size()) <= chk_keep_) return;
    auto it = done.end();
    for (int i = 0; i < chk_keep_; ++i) --it;
    const int oldest_kept = it->first;
    // delete every older checkpoint, complete or not (all ranks have finished it: a newer one is complete)
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(chk_dir_, ec)) {
        const std::string leaf = e.path().filename().string();
        if (leaf.rfind("chk_", 0) != 0) continue;
        const int s = std::atoi(leaf.c_str() + 4);
        if (s < oldest_kept) fs::remove_all(e.path(), ec);
    }
}

std::string Simulation::resolve_restart(const std::string& spec) const {
    if (spec == "latest") {
        const auto done = complete_checkpoints(chk_dir_);
        if (done.empty()) throw std::runtime_error("restart.from = latest: no complete checkpoint in " + chk_dir_);
        return done.rbegin()->second;
    }
    if (!fs::exists(spec + "/checkpoint.txt")) throw std::runtime_error("restart.from: " + spec + " is not a checkpoint directory");
    return spec;
}

void Simulation::read_checkpoint(const std::string& dir) {
    const int P = comm_.size(), rank = comm_.rank();
    int ranks = 0;
    {
        std::ifstream meta(dir + "/checkpoint.txt");
        std::string key;
        while (meta >> key) if (key == "ranks") meta >> ranks;
    }
    if (ranks < 1) throw std::runtime_error(dir + "/checkpoint.txt: number of ranks missing");
    const bool same = (ranks == P);
    std::vector<ChkFile> files;
    if (same) files.push_back(read_chk(dir + "/rank_" + std::to_string(rank) + ".bin"));
    else for (int r = 0; r < ranks; ++r) files.push_back(read_chk(dir + "/rank_" + std::to_string(r) + ".bin"));
    const ChkFile& c0 = files[0];
    // the box and the grid must be those of the checkpoint
    if (c0.M != grid_->N + 1 || c0.K != box_.nxi || std::abs(c0.dxi - box_.dxi) > 1e-12 * box_.dxi ||
        std::abs(c0.xi_min - box_.xi_min) > 1e-12 || c0.m1 != (m1_ ? 1 : 0))
        throw std::runtime_error("restart: the radial grid, the xi box or the modes differ from the checkpoint " + dir);
    if (c0.names.size() != beams_.size()) throw std::runtime_error("restart: the beams differ from the checkpoint");
    for (size_t b = 0; b < beams_.size(); ++b)
        if (c0.names[b] != beams_[b]->name()) throw std::runtime_error("restart: beam '" + c0.names[b] + "' expected");
    if (c0.laser != static_cast<bool>(laser_)) throw std::runtime_error("restart: laser present in only one of deck and checkpoint");

    t_ = c0.t;
    start_step_ = c0.step;
    lost_total_ = c0.lost;
    for (size_t b = 0; b < beams_.size(); ++b) {
        beams_[b]->set_started(c0.started[b] != 0);
        if (same) {
            beams_[b]->set_particles(c0.particles[b]);   // memory order, removed particles included
            outbox_[b] = c0.outbox[b];
        } else {
            std::vector<double> all;
            for (const auto& f : files) {
                all.insert(all.end(), f.particles[b].begin(), f.particles[b].end());
                all.insert(all.end(), f.outbox[b].begin(), f.outbox[b].end());
            }
            beams_[b]->set_particles(all);
            beams_[b]->restrict_to_local();   // keeps live particles of the local slices (all for P = 1)
            outbox_[b].clear();
        }
    }
    if (laser_) {
        const int M = grid_->N + 1, KL = box_.nloc, k0 = box_.k0;
        View3D a("restart.a", KL, M, 2);
        auto h = Kokkos::create_mirror_view(a);
        for (const auto& f : files)
            for (int kf = 0; kf < f.nloc; ++kf) {
                const int kl = f.k0 + kf - k0;
                if (kl < 0 || kl >= KL) continue;
                for (int j = 0; j < M; ++j)
                    for (int c = 0; c < 2; ++c) h(kl, j, c) = f.env[(static_cast<size_t>(kf) * M + j) * 2 + c];
            }
        Kokkos::deep_copy(a, h);
        laser_->restore(a);
    }
    if (comm_.root())
        std::cout << "Restart from " << dir << ": step " << start_step_ << ", t = " << t_
                  << (same ? "" : "  (checkpoint of " + std::to_string(ranks) + " ranks redistributed)") << std::endl;
}

void Simulation::truncate_logs(int step) {
    // keep comments and the lines of steps < step; column of the step: beams.txt 1, others 0
    const std::pair<const char*, int> logs[] = {{"beams.txt", 1}, {"laser.txt", 0}, {"ionization.txt", 0}, {"timestep.txt", 0}, {"spin.txt", 1}};
    for (const auto& l : logs) {
        const std::string fn = outdir_ + "/" + l.first;
        std::ifstream in(fn);
        if (!in) continue;
        std::vector<std::string> keep;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') { keep.push_back(line); continue; }
            std::istringstream s(line);
            std::string tok;
            for (int c = 0; c <= l.second; ++c) s >> tok;
            if (std::atoi(tok.c_str()) < step) keep.push_back(line);
        }
        in.close();
        std::ofstream out(fn, std::ios::trunc);
        for (const auto& k : keep) out << k << "\n";
    }
}

} // namespace quarz
