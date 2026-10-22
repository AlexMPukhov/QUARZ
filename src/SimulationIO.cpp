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
#include "SimulationIOInternal.hpp"

namespace quarz {

namespace fs = std::filesystem;
using namespace detail;

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

} // namespace quarz
