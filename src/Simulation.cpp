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

Simulation::Simulation(const Config& cfg) : cfg_(cfg), comm_(Comm::world()) {
    const int modes = cfg.get_int("modes", 0);
    if (modes != 0 && modes != 1) throw std::runtime_error("modes must be 0 (axisymmetric) or 1 (m = 0 and 1)");
    m1_ = (modes == 1);
    picard_ = cfg.get_int("solver.picard", m1_ ? 3 : 1);
    max_cells_ = cfg.get_double("pusher.max_cells_per_step", 0.0);
    substep_max_ = cfg.get_int("pusher.substep_max", 64);
    if (max_cells_ < 0 || substep_max_ < 1)
        throw std::runtime_error("pusher.max_cells_per_step must be >= 0 and pusher.substep_max >= 1");

    grid_ = std::make_unique<RadialGrid>(cfg);
    const std::string tm = cfg.get_string("solver.tridiag", "auto");
    TridiagMethod method = TridiagMethod::Auto;
    if (tm == "thomas") method = TridiagMethod::Thomas;
    else if (tm == "pcr") method = TridiagMethod::PCR;
    else if (tm == "partition") method = TridiagMethod::Partition;
    else if (tm != "auto") throw std::runtime_error("solver.tridiag must be auto|thomas|pcr|partition");
    solver_ = std::make_unique<FieldSolver>(*grid_, method);
    // radial smoothing of the plasma sources (regularizes the axis caustic at the bubble back)
    smooth_a_ = cfg.get_double("plasma.smooth_length", 0.0);
    if (smooth_a_ < 0) throw std::runtime_error("plasma.smooth_length must be >= 0");
    if (smooth_a_ > 0) solver_->set_filter(std::vector<Real>(grid_->N, smooth_a_ * smooth_a_));

    box_.xi_min = cfg.get_double("xi.min", 0.0);
    const Real xi_max = cfg.get_double("xi.max");
    box_.dxi = cfg.get_double("xi.step");
    box_.nxi = static_cast<int>(std::lround((xi_max - box_.xi_min) / box_.dxi)) + 1;
    xi_head_ = box_.xi_min;
    bwrite_path_ = cfg.get_string("boundary.write", "");
    bread_path_ = cfg.get_string("boundary.read", "");
    dt_ = cfg.get_double("time.dt", 0.0);
    nsteps_ = cfg.get_int("time.steps", 0);
    t_ = cfg.get_double("time.start", 0.0);
    // ---- end of the run and adaptive time step
    t_end_ = cfg.get_double("time.t_end", -1.0);
    if (t_end_ >= 0 && !cfg.has("time.steps")) nsteps_ = 1 << 28;   // the run ends at t_end
    adaptive_ = cfg.get_bool("time.adaptive", false);
    if (adaptive_) {
        nbeta_ = cfg.get_double("time.nt_per_betatron", 20.0);
        dt_max_ = cfg.get_double("time.dt_max", dt_ > 0 ? dt_ : 1e300);
        dt_min_ = cfg.get_double("time.dt_min", 0.0);
        gthr_ = cfg.get_double("time.adaptive_gamma_min", 2.0);
        adens_ = cfg.get_double("time.adaptive_density", 0.0);
        const int P0 = comm_.size();
        lag_ = cfg.get_int("time.adaptive_lag", P0 > 1 ? P0 + 1 : 1);
        if (nbeta_ <= 0 || dt_max_ <= 0 || dt_min_ < 0 || dt_min_ > dt_max_)
            throw std::runtime_error("time.adaptive: need time.nt_per_betatron > 0 and 0 <= time.dt_min <= time.dt_max");
        if (lag_ < P0) throw std::runtime_error("time.adaptive_lag must be >= the number of MPI ranks");
        if (cfg.has("laser.a0")) throw std::runtime_error("time.adaptive is not available with a laser (fixed dt of the envelope solver)");
    } else if (dt_ <= 0 && nsteps_ > 0) throw std::runtime_error("time.dt must be > 0 (or set time.adaptive = 1)");
    out_every_ = cfg.get_int("output.every", 1);
    slice_bins_ = cfg.get_int("output.beam_slices", 0);
    {
        const auto rg = cfg.get_list("output.beam_slices_range");
        slice_lo_ = rg.size() == 2 ? std::stod(rg[0]) : box_.xi_min;
        slice_hi_ = rg.size() == 2 ? std::stod(rg[1]) : xi_max;
    }
    outdir_ = cfg.get_string("output.dir", "out");
    seed_ = cfg.get_int("plasma.seed", 1);

    // ---- decomposition along xi: contiguous blocks of slices, rank 0 at the head
    const int P = comm_.size(), r = comm_.rank(), K = box_.nxi;
    if (P > K) throw std::runtime_error("more MPI ranks than xi slices");
    box_.k0 = static_cast<int>((static_cast<long long>(K) * r) / P);
    box_.nloc = static_cast<int>((static_cast<long long>(K) * (r + 1)) / P) - box_.k0;
    if (box_.nloc < 2 && P > 1) throw std::runtime_error("each rank needs at least 2 xi slices");
    const std::string shape = cfg.get_string("beams.xi_shape", P > 1 ? "ngp" : "linear");
    if (shape == "ngp") box_.ngp = true;
    else if (shape == "linear") {
        if (P > 1) throw std::runtime_error("beams.xi_shape = linear is not possible with the xi decomposition; use ngp");
        box_.ngp = false;
    } else throw std::runtime_error("beams.xi_shape must be linear or ngp");

    profile_ = DensityProfile(cfg, "plasma");
    auto names = cfg.get_list("plasma.species");
    if (names.empty()) names.push_back("electrons");
    bool has_positive = false;
    for (const auto& n : names) {
        species_.push_back(std::make_unique<PlasmaSpecies>(cfg, n, *grid_, profile_, m1_));
        const auto& sp = *species_.back();   // (neutral gas does not count as a positive species)
        if (sp.charge() > 0 && (!sp.ionizable() || sp.ion_info().P.z0 > 0)) has_positive = true;
    }
    neutralize_ = cfg.get_bool("plasma.neutralize", !has_positive);
    bool any_impact = false;
    for (size_t i = 0; i < species_.size(); ++i) {
        if (!species_[i]->ionizable()) continue;
        const std::string pn = species_[i]->ion_info().product;
        int pi = -1;
        for (size_t j = 0; j < species_.size(); ++j) if (species_[j]->name() == pn) pi = static_cast<int>(j);
        if (pi < 0) throw std::runtime_error(species_[i]->name() + ".ionization.product: no plasma species '" + pn +
                                             "' (add it to plasma.species; '" + pn + ".ppc = 0' gives an initially empty species)");
        if (species_[pi]->ionizable() || species_[pi]->charge() >= 0 || !species_[pi]->mobile())
            throw std::runtime_error(species_[i]->name() + ".ionization.product '" + pn + "' must be a mobile negative species");
        ion_sp_.push_back(static_cast<int>(i));
        ion_prod_.push_back(pi);
        any_impact = any_impact || species_[i]->ion_info().P.impact;
    }
    ion_acc_.assign(ion_sp_.size(), {0, 0, 0, 0});

    std::vector<long> nglobal;
    for (const auto& n : cfg.get_list("beams")) {
        beams_.push_back(std::make_unique<Beam>(cfg, n, *grid_, box_, m1_));
        nglobal.push_back(beams_.back()->num_global());   // the generators keep only the local particles
    }
    // beams of trapped plasma particles (<species>.trap = 1): initially empty
    trap_beam_.assign(species_.size(), -1);
    for (size_t i = 0; i < species_.size(); ++i) {
        if (!species_[i]->trap()) continue;
        const std::string bn = species_[i]->trap_beam();
        for (const auto& b : beams_)
            if (b->name() == bn)
                throw std::runtime_error(species_[i]->name() + ".trap_beam '" + bn + "' is created automatically: do not list it in 'beams'");
        const double qm[2] = {species_[i]->charge(), species_[i]->mass()};
        beams_.push_back(std::make_unique<Beam>(cfg, bn, *grid_, box_, m1_, qm));
        nglobal.push_back(0);
        trap_beam_[i] = static_cast<int>(beams_.size()) - 1;
    }
    outbox_.resize(beams_.size());

    const int M = grid_->N + 1;
    const int KL = box_.nloc;
    src_ = SliceSources(M);
    fld_ = SliceFields(M);
    bg_rhot_ = MScalar("bg.rhot", M);
    bg_rho_ = MScalar("bg.rho", M);
    gz_ = MVector("gz", M);
    fld = View3D("fld", KL, M, m1_ ? int(F_NC1) : int(F_NC0));
    bsrc = View3D("bsrc", KL, M, m1_ ? int(B_NC1) : int(B_NC0));
    diag_base_ = m1_ ? int(D_NC1) : int(D_NC0);
    diag = View3D("diag", KL, M, diag_base_ + static_cast<int>(ion_sp_.size()));
    if (any_impact) imp_ = View3D("impact", KL, M, 3);
    bguard = View2D("bguard", M, B_NC1);

    if (comm_.root()) {
        std::cout << "QUARZ " QUARZ_VERSION " - Quasistatic Arbitrary-resolution RZ code, Kokkos execution space: " << ExecSpace::name() << "\n";
        if (P > 1)
            std::cout << "MPI: " << P << " ranks, xi decomposition with time pipelining, " << K / P
                      << "-" << (K + P - 1) / P << " slices per rank\n";
        std::cout << "Azimuthal modes: " << (m1_ ? "m = 0 and 1" : "m = 0 (axisymmetric)");
        if (m1_) std::cout << ",  Picard iterations for the chi_1 coupling: " << picard_;
        std::cout << "\n";
        grid_->print_summary(std::cout);
        std::cout << "Box: xi in [" << box_.xi_min << ", " << box_.xi_min + (K - 1) * box_.dxi << "], dxi = "
                  << box_.dxi << ", " << K << " slices;  beam shape in xi: " << (box_.ngp ? "nearest slice" : "linear")
                  << "\n";
        for (const auto& s : species_)
            std::cout << "Plasma species '" << s->name() << "': q = " << s->charge() << ", m = " << s->mass()
                      << (s->frozen() ? ", immobile" : (s->mobile() ? ", mobile" : ", immobile")) << ", "
                      << "up to " << s->num_particles() << " macro-particles per slice (rings without plasma are skipped)\n";
        for (int i : ion_sp_) std::cout << "Species '" << species_[i]->name() << "'" << species_[i]->ion_info().description() << "\n";
        std::cout << "Neutralising immobile background: " << (neutralize_ ? "yes" : "no") << "\n";
        if (smooth_a_ > 0) {
            std::cout << "Radial smoothing of the plasma sources: a = " << smooth_a_ << "\n";
        }
        for (size_t b = 0; b < beams_.size(); ++b) {
            std::cout << "Beam '" << beams_[b]->name() << "': " << nglobal[b] << " macro-particles"
                      << (beams_[b]->rigid() ? " (rigid)" : "") << "\n";
            if (beams_[b]->scattering()) std::cout << "  " << beams_[b]->scattering_description() << "\n";
        }
        std::filesystem::create_directories(outdir_);
        grid_->write(outdir_ + "/grid.txt");
    }
    comm_.barrier();   // output directory exists
    // ---- restart: the checkpoint gives the first step; the logs keep their lines of earlier steps
    {
        const std::string rs = cfg.get_string("restart.from", "");
        chk_dir_ = cfg.get_string("checkpoint.dir", outdir_ + "/checkpoints");
        if (!rs.empty()) {
            restart_path_ = resolve_restart(rs);
            restarted_ = true;
            std::ifstream meta(restart_path_ + "/checkpoint.txt");
            std::string key;
            while (meta >> key)
                if (key == "step") meta >> start_step_;
            if (comm_.root()) truncate_logs(start_step_);
            comm_.barrier();
        }
    }
    const auto log_mode = restarted_ ? std::ios::app : std::ios::trunc;
    // a header is written unless the run continues an existing log (restart into the same directory)
    auto fresh = [&](const std::string& f) { return !restarted_ || !std::filesystem::exists(outdir_ + "/" + f); };
    const bool fresh_beams = fresh("beams.txt"), fresh_laser = fresh("laser.txt"), fresh_ion = fresh("ionization.txt");
    if (P == 1) beamlog_.open(outdir_ + "/beams.txt", std::ios::out | log_mode);
    else beamlog_.open(outdir_ + "/beams.part" + std::to_string(r));
    if (P == 1 && fresh_beams)
        beamlog_ << "# beam step t alive npart charge gamma_mean gamma_rms xi_mean xi_rms r_rms emit_nx"
                    " x_mean y_mean emit_ny sigma_x sigma_y\n";
    bool any_spin = false;
    for (const auto& b : beams_) any_spin = any_spin || b->spin();
    if (any_spin) {
        const bool fresh_spin = fresh("spin.txt");
        if (P == 1) spinlog_.open(outdir_ + "/spin.txt", std::ios::out | log_mode);
        else spinlog_.open(outdir_ + "/spin.part" + std::to_string(r));
        if (P == 1 && fresh_spin) spinlog_ << "# beam step t <s_x> <s_y> <s_z> P = |<s>| (weighted means over live particles)\n";
    }

    // ---- laser envelope
    if (cfg.has("laser.a0")) {
        laser_ = std::make_unique<Laser>(cfg, *grid_, *solver_, box_, t_, dt_);
        fld_.laser = true;
        if (comm_.root()) std::cout << laser_->description() << "\n";
        if (P == 1) laserlog_.open(outdir_ + "/laser.txt", std::ios::out | log_mode);
        else laserlog_.open(outdir_ + "/laser.part" + std::to_string(r));
        if (P == 1 && fresh_laser) laserlog_ << "# step t a_max w_rms xi_centroid L_rms energy (int |a|^2 dV)\n";
        for (int i : ion_sp_) species_[i]->set_laser(laser_->k0(), laser_->circular());
    }
    if (!ion_sp_.empty()) {
        if (P == 1) ionlog_.open(outdir_ + "/ionization.txt", std::ios::out | log_mode);
        else ionlog_.open(outdir_ + "/ionization.part" + std::to_string(r));
        if (P == 1 && fresh_ion) {
            ionlog_ << "# step t";
            for (int i : ion_sp_) {
                const std::string& nm = species_[i]->name();
                ionlog_ << "  " << nm << ":born_weight " << nm << ":born_per_m " << nm << ":born_p2 " << nm << ":z_tail";
            }
            ionlog_ << ION_LOG_NOTE;
        }
    }

    // ---- output groups (main output and diag.names), checkpoints, restart
    setup_output();
    if (restarted_) read_checkpoint(restart_path_);
    if (!bwrite_path_.empty() && comm_.rank() == P - 1) open_boundary_write();
    if (!bread_path_.empty()) open_boundary_read();   // every rank: xi_head_
    // ---- time step bookkeeping
    if (adaptive_ && restarted_) read_adapt_state(restart_path_);
    else {
        thist_[start_step_] = t_;
        if (adaptive_) {
            double g = local_gamma_eff();
            gmin_init_ = comm_.allreduce_min(g);
            grec_[start_step_ - 1] = gmin_init_;   // gamma_eff at t_start
        }
    }
    next_rec_ = start_step_;
    if (adaptive_ && comm_.root()) {
        dtlog_.open(outdir_ + "/timestep.txt", std::ios::out | (restarted_ ? std::ios::app : std::ios::trunc));
        if (!restarted_ || dtlog_.tellp() == 0)
            dtlog_ << "# step t dt gamma_eff n_max   (adaptive: dt = 2 pi / (" << nbeta_
                   << " omega_beta), omega_beta^2 = n_max / (2 gamma_eff), gamma_eff = gamma m/|q| of the slowest beam particle)\n";
        std::cout << "Adaptive time step: " << nbeta_ << " steps per betatron period, dt in [" << dt_min_ << ", " << dt_max_
                  << "], gamma >= " << gthr_ << ", lag " << lag_ << " steps\n";
    }
}

// ---------------------------------------------------------------------------
// time step
double Simulation::local_gamma_eff() const {
    double g = 1e300;
    for (const auto& b : beams_) g = std::min(g, b->min_gamma_eff(gthr_));
    return std::min(g, 1e300);
}

double Simulation::plasma_density_max(double z0, double z1) const {
    if (adens_ > 0) return adens_;
    double ne = 0, np = 0;
    for (const auto& s : species_) {
        const double n = std::abs(s->charge()) * s->max_density(z0, z1, grid_->R);
        if (s->charge() < 0) ne += n; else np += n;
    }
    return ne > 0 ? ne : np;   // electrons; a pure ion channel: its charge density
}

void Simulation::collect_records(int m) {
    const int P = comm_.size();
    for (; next_rec_ <= m; ++next_rec_) {
        const int k = next_rec_;
        double g = lrec_.count(k) ? lrec_[k] : 1e300;
        for (int r = 1; r < P; ++r) {
            const auto v = comm_.recv(r, 31000 + k % 1500);
            if (v.size() != 2 || static_cast<int>(v[0]) != k) throw std::runtime_error("adaptive time step: record out of order");
            g = std::min(g, v[1]);
        }
        grec_[k] = g;
    }
}

double Simulation::choose_dt(int n, bool& last) {
    last = (n >= nsteps_) || (t_end_ >= 0 && t_ >= t_end_ - 1e-9 * std::max(1.0, std::abs(t_end_)));
    double dt = dt_;
    if (adaptive_) {
        const int first = grec_.begin()->first;          // start_step - 1 (gamma at t_start)
        const int m = std::max(n - lag_, first);
        if (m >= next_rec_) collect_records(m);
        const double G = grec_.at(m);
        double gam = G;
        if (G < 1e299) {
            // extrapolate a decreasing gamma_eff from t_{m+1} to t_{n+1}
            if (grec_.count(m - 1) && thist_.count(m + 1) && thist_.count(m)) {
                const double rate = (G - grec_.at(m - 1)) / (thist_.at(m + 1) - thist_.at(m));
                const double tau = t_ + (have_prev_ ? dt_prev_ : 0.0) - thist_.at(m + 1);
                if (rate < 0 && tau > 0) gam = std::max(0.5 * G, G + rate * tau);
            }
            // the plasma of the whole box has the density at the box head, z = t - xi_min (see
            // step_fields); the maximum over the head positions of this step, [t_n, t_n + dt]:
            // first with dt = dt_{n-1}, then once more with the dt found
            const double zh = t_ - xi_head_, cnst = 6.283185307179586 / nbeta_;
            auto dt_of = [&](double nm) { return nm > 0 ? cnst * std::sqrt(2 * gam / nm) : dt_max_; };
            double nmax = plasma_density_max(zh, zh + std::min(have_prev_ ? dt_prev_ : 0.0, dt_max_));
            dt = std::min(dt_max_, dt_of(nmax));
            const double n2 = plasma_density_max(zh, zh + dt);
            if (n2 > nmax) { nmax = n2; dt = std::min(dt, dt_of(nmax)); }
            last_dens_ = nmax;
        } else dt = dt_max_;
        last_gamma_ = gam;
        dt = std::min(dt_max_, std::max(dt_min_, dt));
        if (!(dt > 0) || dt >= 1e299) throw std::runtime_error("adaptive time step: no finite dt (no beam and no time.dt_max?)");
    }
    if (t_end_ >= 0 && !last) {   // land exactly on t_end, without a tiny last step
        const double rem = t_end_ - t_;
        if (dt >= rem * (1 - 1e-12)) dt = rem;
        else if (dt > 0.5 * rem) dt = 0.5 * rem;
    }
    return dt;
}

Simulation::~Simulation() = default;

void Simulation::run() {
    Kokkos::Timer total;
    const int P = comm_.size(), rank = comm_.rank();
    if (start_step_ > nsteps_ && comm_.root())
        std::cout << "restart at step " << start_step_ << " > time.steps = " << nsteps_ << ": nothing to do\n";
    int n_last = start_step_ - 1;
    for (int n = start_step_; n <= nsteps_; ++n) {
        Kokkos::Timer timer;
        if (rank == 0) cur_dt_ = choose_dt(n, cur_last_);
        // ---- message from the upstream rank: beam particles, plasma state, B+, guard
        std::vector<double> msg;
        size_t off = 0;
        if (rank > 0) {
            msg = comm_.recv(rank - 1, n % 30000);
            if (static_cast<int>(msg[0]) != n) throw std::runtime_error("pipeline message out of order");
            cur_dt_ = msg[1];
            cur_last_ = msg[2] != 0;
            off = 3;
            for (auto& b : beams_) {
                const size_t np = static_cast<size_t>(msg[off]);
                ++off;
                b->append(msg.data() + off, np);
                off += static_cast<size_t>(b->ncomp()) * np;
            }
        }
        step_ = n;
        step_fields(msg, off);
        if (bwf_) write_boundary(n);
        convert_trapped(n);
        const double tf = timer.seconds();
        if (rank < P - 1) comm_.isend(rank + 1, n % 30000, make_message(n));

        long lost = 0;
        for (auto& s : species_) if (s->mobile()) { lost += s->lost_count(); s->reset_lost(); }
        const long trapped = trapped_step_;
        lost -= trapped;
        const double nsub = nsub_step_;
        nsub_step_ = 0;
        lost_total_ += lost;

        write_beam_output(n);
        if (laser_) write_laser_diag(n);
        if (!ion_sp_.empty()) write_ion_diag(n);
        write_group_outputs(n);

        if (comm_.root()) {
            std::cout << "step " << std::setw(6) << n << "  t = " << std::setw(10) << t_;
            if (adaptive_ && !cur_last_) std::cout << "  dt = " << std::setw(10) << cur_dt_;
            std::cout << "   sweep " << std::fixed << std::setprecision(3) << tf << " s" << std::defaultfloat;
            if (P > 1) std::cout << " (rank 0 of " << P << ", pipelined)";
            if (lost) std::cout << "   (" << lost << " plasma particles removed as trapped)";
            if (trapped) std::cout << "   (" << trapped << " trapped plasma particles -> beam)";
            if (nsub > 0) std::cout << "   (" << static_cast<long>(nsub) << " extra sub-slices)";
            std::cout << std::endl;
        } else if (lost || trapped) {
            std::cout << "rank " << rank << " step " << n << ": ";
            if (lost) std::cout << lost << " plasma particles removed as trapped ";
            if (trapped) std::cout << trapped << " trapped plasma particles -> beam";
            std::cout << std::endl;
        }

        if (adaptive_ && comm_.root() && !cur_last_)
            dtlog_ << n << " " << std::setprecision(12) << t_ << " " << cur_dt_ << " " << last_gamma_ << " " << last_dens_ << std::endl;

        // the last step pushes the beams only for a final checkpoint (the state of step n + 1)
        const bool last = cur_last_;
        const bool chk = (chk_every_ > 0 && (n + 1) % chk_every_ == 0 && !last) || (last && chk_at_end_);
        if (!have_prev_) { dt_prev_ = cur_dt_; have_prev_ = true; }
        if (!last || chk) {
            const Real dkick = Real(0.5) * (dt_prev_ + cur_dt_), ddrift = cur_dt_;
            for (size_t b = 0; b < beams_.size(); ++b) {
                beams_[b]->push(fld, dkick, ddrift, laser_ ? laser_->ponderomotive() : View3D(), diag, n);
                if (P > 1) {
                    auto out = beams_[b]->extract_outgoing();
                    if (rank < P - 1) outbox_[b].insert(outbox_[b].end(), out.begin(), out.end());
                }
            }
            t_ += cur_dt_;
            dt_prev_ = cur_dt_;
            thist_[n + 1] = t_;
            if (adaptive_ && !last) {   // record for the adaptive time step of later steps
                const double g = local_gamma_eff();
                lrec_[n] = g;
                if (rank > 0) comm_.isend_small(0, 31000 + n % 1500, std::vector<double>{double(n), g});
            }
        }
        if (chk) write_checkpoint(n + 1);
        n_last = n;
        if (last) break;
    }
    // rank 0 collects the records it did not need, so that all small sends complete
    if (adaptive_ && rank == 0) {
        if (n_last - 1 >= next_rec_) collect_records(n_last - 1);
        // the global gamma_eff after every push (for checking the step choice afterwards)
        std::ofstream g(outdir_ + "/gamma_records.txt", std::ios::out | (restarted_ ? std::ios::app : std::ios::trunc));
        if (!restarted_) g << "# step m, t_{m+1}, min gamma_eff over all beams after the push of step m\n";
        for (const auto& [m, v] : grec_)
            if (m >= start_step_ && thist_.count(m + 1)) g << m << " " << std::setprecision(12) << thist_.at(m + 1) << " " << v << "\n";
    }
    comm_.wait_send();
    if (bwf_) { std::fclose(bwf_); bwf_ = nullptr; }
    if (brf_) { std::fclose(brf_); brf_ = nullptr; }
    Kokkos::fence();
    for (auto& g : groups_)
        if (g.opmd) g.opmd->close();
    beamlog_.close();
    if (spinlog_.is_open()) spinlog_.close();
    laserlog_.close();
    ionlog_.close();
    const double tmax = comm_.allreduce_max(total.seconds());
    if (P > 1 && comm_.root()) merge_partial_outputs();
    comm_.barrier();
    if (comm_.root() && (chk_every_ > 0 || chk_at_end_)) prune_checkpoints();
    if (comm_.root()) std::cout << "Total run time " << tmax << " s\n";
}

} // namespace quarz
