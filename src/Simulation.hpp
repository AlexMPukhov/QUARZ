// Time loop: for every time step the plasma is swept through the box slice
// by slice (xi = xi_min ... xi_max), the fields are stored on the (xi, r)
// grid in azimuthal modes (m = 0, optionally m = 1), then the beams are
// advanced with a large time step dt.
//
// Parallelisation (MPI): domain decomposition along xi with time pipelining.
// Rank r owns the slices [k0, k0 + nloc).  For time step n it
//   1. receives from rank r-1 the plasma state entering its first slice (particles,
//      Adams-Bashforth history, B+ warm start), the beam rho - J_z of the slice just
//      upstream, and beam particles that slipped into its slices;
//   2. deposits its beam particles, sweeps its slices, and sends the same kind of
//      message to rank r+1;
//   3. writes its part of the output and pushes its beam particles.
// While rank r works on step n, rank r-1 already works on step n+1: with P ranks and
// many time steps the speed-up approaches P.  Beam particles are owned by nearest
// slice and use nearest-slice (NGP) deposit/gather in xi, which keeps every rank's
// work local to its slices; serial runs can use the same scheme (beams.xi_shape = ngp)
// and then give the same results as parallel runs.
#pragma once

#include "Beam.hpp"
#include "Config.hpp"
#include "FieldSolver.hpp"
#include "FieldStore.hpp"
#include "FieldTable.hpp"
#include "Laser.hpp"
#include "OpenPMDWriter.hpp"
#include "Parallel.hpp"
#include "PlasmaSpecies.hpp"
#include "Profiles.hpp"
#include "RadialGrid.hpp"
#include "SliceData.hpp"

#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace quarz {

// diagnostics-only arrays stored per slice
enum DiagComp : int { D_PSI0 = 0, D_NE0, D_NI0, D_PSI1R, D_PSI1I, D_NE1R, D_NE1I, D_NI1R, D_NI1I, D_NC0 = 3, D_NC1 = 9 };

// One output group: the main output (keys output.*) or a named diagnostic (keys diag.<name>.*),
// each with its own period, selection of fields and beams, and extent.
struct OutputGroup {
    std::string name;                 // "" = main output
    std::string dir;                  // directory of its files
    int every = 0;                    // field output period (steps; 0 = never)
    int beam_every = 0;               // beam particle output period
    bool axis = false;                // axis_*.txt with the field output
    bool native = true;               // native files (fields_*.bin, beam_*.bin)
    bool field_files = true;          // fields_*.bin (native)
    std::vector<std::string> fields;  // base names of the selected field components (empty: all)
    std::vector<int> beams;           // indices of the selected beams
    int Mout = 0;                     // radial nodes written (native): r <= rmax
    int xi_stride = 1;                // every xi_stride-th slice
    int particle_stride = 1;          // every particle_stride-th beam particle
    std::unique_ptr<OpenPMDWriter> opmd;
    bool wants(const std::string& comp) const;   // component selected (ez, ez_c, ez_s -> "ez"; a_re/a_im -> "a")
};

class Simulation {
public:
    explicit Simulation(const Config& cfg);
    ~Simulation();

    void run();

    bool mode1() const { return m1_; }
    const RadialGrid& grid() const { return *grid_; }
    const BeamGrid& box() const { return box_; }
    Real time() const { return t_; }

    View3D fld;    // (local xi, r, FieldComp): fields for the beams
    View3D bsrc;   // (local xi, r, BeamComp): beam sources
    View3D diag;   // (local xi, r, DiagComp)
    View2D bguard; // (r, B_NC1): beam sources of the slice just upstream of this rank (ngp xi-derivatives)

    // native field file of an output group (nullptr: all fields, all slices, all nodes)
    void write_fields(const std::string& filename, const OutputGroup* g = nullptr) const;
    void write_axis(const std::string& filename) const;
    // host copy of the local fields, by name (only the components selected by g, if given)
    std::shared_ptr<const FieldTable> field_table(const OutputGroup* g = nullptr) const;

    // slice-level kernels (public: they contain device lambdas, CUDA restriction)
    void zero_sources();
    void combine_sources(int kl, Real ds = 0);   // ds > 0: sub-slice at xi_k + ds (beam rho - J_z extrapolated)
    void solve_slice_fields();                  // psi, E_z, W+, B_z, S, B+ from the sources in src_
    void filter_plasma_sources();               // radial smoothing of the plasma deposit (plasma.smooth_length)
    int  subslices(Real dxi) const;             // adaptive sub-slicing: number of sub-slices for the next step
    void compute_wplus();
    void compute_bz();
    void solve_bplus(int k);
    void store_slice(int kl);
    void set_guard(const double* p);
    void get_last_rhot(std::vector<double>& buf) const;
    void step_fields(const std::vector<double>& msg, size_t& off);   // one quasi-static sweep of the local slices
    std::vector<double> make_message(int n);                         // message for the downstream rank

private:
    void write_beam_output(int n);
    void merge_partial_outputs();

    const Config& cfg_;
    Comm& comm_;
    bool m1_ = false;
    int picard_ = 1;
    Real smooth_a_ = 0;      // plasma.smooth_length
    Real max_cells_ = 0;     // pusher.max_cells_per_step (0: no sub-slicing)
    int  substep_max_ = 64;  // pusher.substep_max
    double nsub_step_ = 0;   // extra sub-slices in this time step (diagnostic)
    std::unique_ptr<RadialGrid> grid_;
    std::unique_ptr<FieldSolver> solver_;
    DensityProfile profile_;
    std::vector<std::unique_ptr<PlasmaSpecies>> species_;
    std::vector<std::unique_ptr<Beam>> beams_;
    std::vector<std::vector<double>> outbox_;   // per beam: particles for the downstream rank
    SliceSources src_;
    SliceFields fld_;
    MScalar bg_rhot_, bg_rho_;
    MVector gz_;                 // i (d_x + i d_y) J_z  work array
    bool neutralize_ = true;
    BeamGrid box_;
    Real dt_ = 0, t_ = 0;
    int nsteps_ = 0;
    int out_every_ = 1, beam_dump_every_ = 0, slice_bins_ = 0;
    double slice_lo_ = 0, slice_hi_ = 0;
    std::string outdir_;
    unsigned long long seed_ = 1;
    std::ofstream beamlog_;
    std::ofstream spinlog_;   // spin.txt: mean spin (polarization) of the beams with <beam>.spin
    long lost_total_ = 0;
    std::vector<int> trap_beam_;   // per plasma species: index of the beam its trapped particles join (-1: none)
    long trapped_step_ = 0;        // particles converted in this step (this rank)
    void convert_trapped(int n);
    // ---- output groups (SimulationIO.cpp)
    std::vector<OutputGroup> groups_;        // [0] = main output (output.*), then diag.names
    void setup_output();
    void parse_group(OutputGroup& g, const std::string& prefix, bool main);
    void write_group_outputs(int n);
    // ---- checkpoints and restart (SimulationIO.cpp)
    int chk_every_ = 0, chk_keep_ = 2;       // checkpoint.every, checkpoint.keep
    bool chk_at_end_ = true;                 // checkpoint.at_end
    std::string chk_dir_;                    // checkpoint.dir
    int start_step_ = 0;                     // first step of this run (> 0 after a restart)
    bool restarted_ = false;
    std::string restart_path_;               // checkpoint directory of the restart
    void write_checkpoint(int next_step);    // state at the beginning of step next_step
    void read_checkpoint(const std::string& dir);
    void prune_checkpoints();
    std::string resolve_restart(const std::string& spec) const;   // "latest" -> newest complete checkpoint
    void truncate_logs(int step);            // drop log lines of steps >= step (restart)
    std::unique_ptr<Laser> laser_;          // laser envelope (laser.a0 given)
    std::ofstream laserlog_;
    void write_laser_diag(int n);
    // ---- ionization
    std::vector<int> ion_sp_, ion_prod_;     // ionizable species and their electron product species (indices)
    View3D imp_;                             // (local xi, r, 3): beam impact-ionization sources
    EITargets eit_;                          // electron-impact ionization: targets (species, level)
    View3D eimp_;                            //   (target, 2 x energy bin, r): rates of the projectiles, this slice
    View2D ntg_;                             //   (target, r): densities of the target levels, this slice
    std::vector<int> ei_proj_;               //   projectile species (mobile electrons)
    bool ei_discrete_ = true;                //   ionization.electron_loss: discrete collisions or continuous
    int diag_base_ = 0;                      // first diag component of the ion charge densities
    int step_ = 0;
    std::ofstream ionlog_;
    std::vector<std::array<double, 4>> ion_acc_;   // per ionizable species: born w, born w p^2, tail z w, tail w
    void write_ion_diag(int n);
    // ---- time step: fixed or adaptive (time.adaptive), end by step count or time.t_end
    // Rank 0 chooses dt_n (and whether step n is the last) and sends it down the pipeline with
    // the step-n message. Adaptive: dt_n = (2 pi / time.nt_per_betatron) / omega_beta with
    // omega_beta^2 = n_max / (2 gamma_eff), gamma_eff = min over beams of max(gamma, gmin) m/|q|,
    // n_max = max plasma density at the box head (the density the whole box sees) during the step.
    // gamma_eff is known globally only with a lag: every rank sends its local minimum after the
    // push of step m to rank 0 (small messages against the pipeline), which uses step m = n - lag
    // (lag >= P, default P + 1, so that rank 0 does not wait), extrapolated to t_{n+1} if it
    // decreases. The result does not depend on timing, so it is reproducible.
    bool adaptive_ = false;
    double nbeta_ = 20, dt_max_ = 0, dt_min_ = 0, gthr_ = 2, adens_ = 0, t_end_ = -1;
    int lag_ = 1;
    double cur_dt_ = 0;                // dt of the current step n (pipeline message)
    bool cur_last_ = false;            // step n is the last one (pipeline message)
    double dt_prev_ = 0;               // dt of step n - 1 (variable-step leapfrog kick)
    bool have_prev_ = false;
    double gmin_init_ = 1e300;         // global gamma_eff at t_0 ("record -1")
    std::map<int, double> grec_;       // rank 0: global gamma_eff after the push of step m (at t_{m+1})
    std::map<int, double> lrec_;       // this rank's records
    std::map<int, double> thist_;      // t_k at the beginning of step k
    int next_rec_ = 0;                 // rank 0: next record to collect
    double last_gamma_ = 0, last_dens_ = 0;   // values behind the last adaptive dt (log)
    std::ofstream dtlog_;
    double choose_dt(int n, bool& last);      // rank 0
    void collect_records(int m);              // rank 0: records up to step m from all ranks
    double local_gamma_eff() const;
    double plasma_density_max(double z0, double z1) const;
    void write_adapt_state(const std::string& dir) const;
    void read_adapt_state(const std::string& dir);
    // ---- two-stage runs through the rear boundary (SimulationIO.cpp)
    // Stage 1 (boundary.write = file): the last rank pushes the plasma through its last slice as well
    // and appends, every step, the plasma state leaving the box (the downstream pipeline message:
    // particles with their Adams-Bashforth history, B+ warm start) to the file. Stage 2
    // (boundary.read = file): a box starting at the next slice takes this state, interpolated linearly
    // in t between the stored steps, instead of fresh plasma at its head; the plasma density profile
    // still refers to the head of the stage-1 box (z = t - xi_head).
    std::string bwrite_path_, bread_path_;
    Real xi_head_ = 0;                  // xi of the plasma column's front (stage 2: stage-1 xi.min)
    std::FILE* bwf_ = nullptr;
    std::FILE* brf_ = nullptr;
    struct BRec { double t; long step; long long off; long long len; double rhob; };
    std::vector<BRec> bindex_;
    std::vector<double> bbuf_;          // interpolated state for this step
    bool bwarned_ = false;
    void open_boundary_write();
    void write_boundary(int n);
    void open_boundary_read();
    void boundary_state(double t);      // fills bbuf_
    std::vector<char> boundary_mask(const std::vector<double>& d) const;
};

} // namespace quarz
