// Output groups (main output + named diagnostics with their own period and selection) and
// checkpoints / restart.
//
// Checkpoint layout:  <checkpoint.dir>/chk_<step>/checkpoint.txt   (rank 0: step, t, ranks, deck)
//                                                 rank_<r>.bin      (one per rank)
// A checkpoint chk_N holds the state at the beginning of step N: beam particles (all, in memory
// order, including removed ones), particles in transit to the downstream rank, the leapfrog
// start flag of every beam, and the laser envelope.  The plasma needs no state: it is loaded
// anew at the head of the box in every step.  Every rank writes its own file when it has
// finished step N - 1 (no synchronisation: the xi pipeline keeps running); a checkpoint is
// complete when all rank files exist.  A restart with the same number of ranks continues
// bit-identically; with a different number, the particles and envelope slices are
// redistributed (the result then differs at round-off level, as for any change of P).
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

namespace quarz {

namespace fs = std::filesystem;

namespace {
constexpr uint64_t CHK_MAGIC = 0x314B484352415551ULL;   // "QUARCHK1"
constexpr int32_t CHK_VERSION = 1;

std::string step_name(int n) {
    char b[16];
    std::snprintf(b, sizeof(b), "%06d", n);
    return b;
}

// complete checkpoints in dir: step -> path
std::map<int, std::string> complete_checkpoints(const std::string& dir) {
    std::map<int, std::string> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string leaf = e.path().filename().string();
        if (leaf.rfind("chk_", 0) != 0 || !e.is_directory()) continue;
        std::ifstream meta(e.path() / "checkpoint.txt");
        std::string key;
        int step = -1, ranks = -1;
        while (meta >> key) {
            if (key == "step") meta >> step;
            else if (key == "ranks") meta >> ranks;
        }
        if (step < 0 || ranks < 1) continue;
        int n = 0;
        for (int r = 0; r < ranks; ++r)
            if (fs::exists(e.path() / ("rank_" + std::to_string(r) + ".bin"), ec)) ++n;
        if (n == ranks) out[step] = e.path().string();
    }
    return out;
}

struct Writer {
    std::ofstream o;
    template <class T> void put(const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    void put(const std::vector<double>& v) {
        put(static_cast<int64_t>(v.size()));
        o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(8 * v.size()));
    }
    void put(const std::string& s) {
        put(static_cast<int32_t>(s.size()));
        o.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
};
struct Reader {
    std::ifstream i;
    std::string fn;
    template <class T> T get() {
        T v;
        i.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!i) throw std::runtime_error("checkpoint file " + fn + " is truncated");
        return v;
    }
    std::vector<double> vec() {
        const int64_t n = get<int64_t>();
        std::vector<double> v(static_cast<size_t>(n));
        i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(8 * n));
        if (!i) throw std::runtime_error("checkpoint file " + fn + " is truncated");
        return v;
    }
    std::string str() {
        const int32_t n = get<int32_t>();
        std::string s(static_cast<size_t>(n), ' ');
        i.read(&s[0], n);
        return s;
    }
};

// one rank's checkpoint file
struct ChkFile {
    int P = 0, rank = 0, step = 0, K = 0, k0 = 0, nloc = 0, M = 0, m1 = 0;
    double t = 0, dxi = 0, xi_min = 0;
    int64_t lost = 0;
    std::vector<std::string> names;
    std::vector<int> started;
    std::vector<std::vector<double>> particles, outbox;
    bool laser = false;
    std::vector<double> env;   // (nloc, M, 2)
};

ChkFile read_chk(const std::string& fn) {
    Reader R;
    R.fn = fn;
    R.i.open(fn, std::ios::binary);
    if (!R.i) throw std::runtime_error("cannot open checkpoint file " + fn);
    ChkFile c;
    if (R.get<uint64_t>() != CHK_MAGIC) throw std::runtime_error(fn + " is not a QUARZ checkpoint");
    if (R.get<int32_t>() != CHK_VERSION) throw std::runtime_error(fn + ": unsupported checkpoint version");
    c.P = R.get<int32_t>(); c.rank = R.get<int32_t>(); c.step = R.get<int32_t>();
    c.K = R.get<int32_t>(); c.k0 = R.get<int32_t>(); c.nloc = R.get<int32_t>(); c.M = R.get<int32_t>();
    c.m1 = R.get<int32_t>();
    const int nb = R.get<int32_t>();
    c.laser = R.get<int32_t>() != 0;
    c.t = R.get<double>(); c.dxi = R.get<double>(); c.xi_min = R.get<double>();
    c.lost = R.get<int64_t>();
    for (int b = 0; b < nb; ++b) {
        c.names.push_back(R.str());
        c.started.push_back(R.get<int32_t>());
        c.particles.push_back(R.vec());
        c.outbox.push_back(R.vec());
    }
    if (c.laser) c.env = R.vec();
    return c;
}
} // namespace

// ============================================================================ output groups

bool OutputGroup::wants(const std::string& comp) const {
    if (fields.empty()) return true;
    std::string base = comp;
    if (comp == "a_re" || comp == "a_im") base = "a";
    else if (comp.size() > 2 && (comp.compare(comp.size() - 2, 2, "_c") == 0 || comp.compare(comp.size() - 2, 2, "_s") == 0))
        base = comp.substr(0, comp.size() - 2);
    return std::find(fields.begin(), fields.end(), base) != fields.end();
}

void Simulation::parse_group(OutputGroup& g, const std::string& pre, bool main) {
    const Config& cfg = cfg_;
    const std::string fmt0 = cfg.get_string("output.format", "native");
    if (main) {
        g.every = cfg.get_int("output.every", 1);
        g.beam_every = cfg.get_int("output.beam_every", 0);
        g.axis = cfg.get_bool("output.axis", true);
    } else {
        g.every = cfg.get_int(pre + "every", 0);
        g.beam_every = cfg.get_int(pre + "beam_every", g.every);
        g.axis = cfg.get_bool(pre + "axis", false);
    }
    if (g.every < 0 || g.beam_every < 0) throw std::runtime_error(pre + "every / beam_every must be >= 0");
    const std::string fmt = main ? fmt0 : cfg.get_string(pre + "format", fmt0);
    if (fmt != "native" && fmt != "openpmd" && fmt != "both")
        throw std::runtime_error(pre + "format must be native, openpmd or both");
    g.native = (fmt != "openpmd");
    g.field_files = cfg.get_bool(pre + "field_files", true);

    // fields: base names; "all" (default) or "none"
    std::vector<std::string> allowed = {"psi", "ez", "er", "eth", "br", "bth", "bz", "ne", "ni", "rhob"};
    for (int i : ion_sp_) allowed.push_back("nz_" + species_[i]->name());
    if (laser_) allowed.push_back("a");
    const auto fl = cfg.get_list(pre + "fields");
    g.fields.clear();
    bool none = false;
    for (const auto& f : fl) {
        if (f == "all") { g.fields.clear(); none = false; break; }
        if (f == "none") { none = true; continue; }
        if (std::find(allowed.begin(), allowed.end(), f) == allowed.end()) {
            std::string a;
            for (const auto& x : allowed) a += " " + x;
            throw std::runtime_error(pre + "fields: unknown field '" + f + "' (available:" + a + ", all, none)");
        }
        g.fields.push_back(f);
    }
    if (none) g.fields = {"<none>"};
    // beams: "all" (default), "none" or names
    g.beams.clear();
    const auto bl = cfg.get_list(pre + "beams");
    if (bl.empty() || (bl.size() == 1 && bl[0] == "all")) {
        for (size_t b = 0; b < beams_.size(); ++b) g.beams.push_back(static_cast<int>(b));
    } else if (!(bl.size() == 1 && bl[0] == "none")) {
        for (const auto& n : bl) {
            int idx = -1;
            for (size_t b = 0; b < beams_.size(); ++b) if (beams_[b]->name() == n) idx = static_cast<int>(b);
            if (idx < 0) throw std::runtime_error(pre + "beams: unknown beam '" + n + "'");
            g.beams.push_back(idx);
        }
    }
    // extent and sampling
    const double R = grid_->R;
    const double rmax = cfg.get_double(pre + "rmax", R);
    if (rmax <= 0) throw std::runtime_error(pre + "rmax must be > 0");
    g.Mout = grid_->N + 1;
    for (int j = 0; j <= grid_->N; ++j)
        if (grid_->r[j] >= rmax * (1 - 1e-12)) { g.Mout = j + 1; break; }
    g.xi_stride = cfg.get_int(pre + "xi_stride", 1);
    g.particle_stride = cfg.get_int(pre + "particle_stride", 1);
    if (g.xi_stride < 1 || g.particle_stride < 1) throw std::runtime_error(pre + "xi_stride and particle_stride must be >= 1");
    if (fmt != "native") {
        if (restarted_) {
            const std::string pat = cfg.get_string(pre + "openpmd_file", cfg.get_string("output.openpmd_file", "openpmd/data_%06T"));
            if (pat.find('%') == std::string::npos)
                throw std::runtime_error(pre + "openpmd_file: a restart needs a file-based series (pattern with %T); "
                                               "a single-file series would be overwritten");
        }
        g.opmd = std::make_unique<OpenPMDWriter>(cfg, g.dir, *grid_, box_, m1_, dt_, pre);
    }
}

void Simulation::setup_output() {
    const Config& cfg = cfg_;
    groups_.clear();
    groups_.emplace_back();
    groups_[0].dir = outdir_;
    std::vector<std::string> names = cfg.get_list("diag.names");
    if (comm_.root())
        for (const auto& n : names) fs::create_directories(outdir_ + "/" + n);
    comm_.barrier();
    parse_group(groups_[0], "output.", true);
    for (const auto& n : names) {
        if (n.empty() || n.find('/') != std::string::npos) throw std::runtime_error("diag.names: bad name '" + n + "'");
        groups_.emplace_back();
        groups_.back().name = n;
        groups_.back().dir = outdir_ + "/" + n;
        parse_group(groups_.back(), "diag." + n + ".", false);
    }
    out_every_ = groups_[0].every;   // beam slice diagnostics go with the main field output

    chk_every_ = cfg.get_int("checkpoint.every", 0);
    chk_keep_ = cfg.get_int("checkpoint.keep", 2);
    chk_at_end_ = cfg.get_bool("checkpoint.at_end", chk_every_ > 0);
    if (chk_every_ < 0) throw std::runtime_error("checkpoint.every must be >= 0");

    if (comm_.root()) {
        auto list = [&](const OutputGroup& g) {
            std::ostringstream s;
            if (g.fields.empty()) s << "all fields";
            else if (g.fields[0] == "<none>") s << "no fields";
            else { s << "fields"; for (const auto& f : g.fields) s << " " << f; }
            s << " every " << g.every << ", ";
            if (g.beams.empty()) s << "no beams";
            else {
                s << "beams";
                for (int b : g.beams) s << " " << beams_[b]->name();
                s << " every " << g.beam_every;
            }
            if (g.Mout < grid_->N + 1) s << ", r <= " << grid_->r[g.Mout - 1] << " (" << g.Mout << " nodes)";
            if (g.xi_stride > 1) s << ", every " << g.xi_stride << ". slice";
            if (g.particle_stride > 1) s << ", every " << g.particle_stride << ". particle";
            s << (g.native ? (g.opmd ? ", native + openPMD" : ", native") : ", openPMD");
            return s.str();
        };
        std::cout << "Output: " << list(groups_[0]) << "\n";
        for (size_t i = 1; i < groups_.size(); ++i)
            std::cout << "Diagnostic '" << groups_[i].name << "' (" << groups_[i].dir << "): " << list(groups_[i]) << "\n";
        for (const auto& g : groups_)
            if (g.opmd) std::cout << g.opmd->description() << "\n";
        if (chk_every_ > 0 || chk_at_end_) {
            std::cout << "Checkpoints in " << chk_dir_ << ":";
            if (chk_every_ > 0) std::cout << " every " << chk_every_ << " steps,";
            if (chk_at_end_) std::cout << " at the end,";
            std::cout << " keep " << (chk_keep_ > 0 ? std::to_string(chk_keep_) : std::string("all")) << "\n";
        }
        if (comm_.size() > 1)
            for (const auto& g : groups_)
                if (g.opmd) { std::cout << "  note: openPMD writes are collective; each output step drains the xi pipeline\n"; break; }
    }
}

void Simulation::write_group_outputs(int n) {
    const bool par = comm_.size() > 1;
    const std::string step = step_name(n);
    for (auto& g : groups_) {
        const bool any_fields = g.fields.empty() || g.fields[0] != "<none>";
        const bool field_out = g.every > 0 && n % g.every == 0;
        const bool beam_out = g.beam_every > 0 && n % g.beam_every == 0 && !g.beams.empty();
        if (field_out) {
            if (g.native && g.field_files && any_fields) write_fields(g.dir + "/fields_" + step + ".bin", &g);
            if (g.axis) write_axis(g.dir + "/axis_" + step + ".txt");
        }
        if (beam_out && g.native)
            for (int b : g.beams) {
                const std::string fn = g.dir + "/beam_" + beams_[b]->name() + "_" + step + ".bin";
                beams_[b]->dump(par ? fn + ".part" + std::to_string(comm_.rank()) : fn, g.particle_stride);
            }
        if (g.opmd && ((field_out && any_fields) || beam_out)) {
            const auto table = (field_out && any_fields) ? field_table(&g) : nullptr;
            std::vector<const Beam*> bl;
            if (beam_out) for (int b : g.beams) bl.push_back(beams_[b].get());
            g.opmd->set_dt(cur_dt_);
            g.opmd->write(n, t_, table.get(), bl, g.xi_stride, g.particle_stride);
        }
    }
}

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

namespace quarz {

// ============================================================================ two-stage runs
// File: "QZBOUND1", int32 version 1, int32 M, int32 m1, int32 nmobile, then per mobile species
// int32 ionizable, int32 ab_order; float64 dxi, xi_head (stage-1 head), xi_next (first slice of
// stage 2), both in the internal xi = t - z (Types.hpp), float64 r[M]; records: float64 t, int64 step, int64 len, float64 max |rho_b| of the
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
    std::cout << "Boundary: the plasma state leaving the box (xi = " << xi_to_user(box_.xi_min + box_.nxi * box_.dxi)
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
        throw std::runtime_error("boundary.read: xi.max must be " + std::to_string(xi_to_user(next)) +
                                 " (the slice behind the stage-1 box)");
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
        std::cout << "Boundary: plasma entering at xi = " << xi_to_user(box_.xi_min) << " from " << bread_path_ << " ("
                  << bindex_.size() << " steps, t = " << bindex_.front().t << " ... " << bindex_.back().t
                  << "); density profile at the stage-1 head xi = " << xi_to_user(head) << "\n";
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
