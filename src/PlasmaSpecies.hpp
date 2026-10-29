// Plasma macro-particles advanced in xi = t - z.
//
// Particles are Cartesian in the transverse plane (x, y, p_x, p_y) with
// Delta = gamma - p_z; with only the m = 0 mode each particle represents a ring
// (fields are axisymmetric), with m = 1 the rings are split into n_theta
// particles at different azimuths.  Momenta in units of m_s c.
//   q^ = q/m,  W+ = W_x + i W_y = (E_x - B_y) + i (E_y + B_x) = -(d_x + i d_y) psi
//   dx/dxi   = p_x / Delta,               dy/dxi = p_y / Delta
//   dp_x/dxi = q^ [ gamma W_x/Delta + B_y + p_y B_z/Delta ]
//   dp_y/dxi = q^ [ gamma W_y/Delta - B_x - p_x B_z/Delta ]
//   dDelta/dxi = q^ [ (p_perp . W_perp)/Delta - E_z ]
// integrated with Adams-Bashforth of order 1..5.
#pragma once

#include "Ionization.hpp"
#include "Parser.hpp"
#include "Profiles.hpp"
#include "RadialGrid.hpp"
#include "SliceData.hpp"

#include <string>
#include <vector>

namespace quarz {

class Config;

struct ABCoeffs {
    Real c[5] = {0, 0, 0, 0, 0};
    int order = 1;
};
ABCoeffs make_ab(int order);
// Adams-Bashforth coefficients for non-uniform steps: history points t[0] (current) > t[1] > ...
// (order of them), new step h; c[m] = (1/h) * integral over [t0, t0 + h] of the Lagrange basis l_m
ABCoeffs make_ab_variable(int order, const double* t, double h);

struct IonizeResult {
    double born_w = 0, born_wp2 = 0;   // sum of weights of the new electrons, sum w p_perp^2
    long born_n = 0;                   // number of new macro-electrons
};

class PlasmaSpecies {
public:
    PlasmaSpecies(const Config& cfg, const std::string& name, const RadialGrid& grid, const DensityProfile& prof,
                  bool mode1);

    void load(Real z, unsigned long long seed);

    // all sources except S
    void deposit(const SliceSources& s, const SliceFields& f) const;   // f: laser <a^2> (if any)
    // rho - J_z and rho only (background / neutralisation), all modes
    void deposit_rho(const MScalar& rhot, const MScalar& rho) const;
    // S+ (needs W+, E_z, B_z)
    void deposit_S(const SliceFields& f, const MVector& S) const;
    // advance slice k -> k+1
    // one Adams-Bashforth step of length h (the steps may vary: adaptive sub-slicing)
    void push(const SliceFields& f, Real h);
    // largest number of radial cells a particle would cross in a step dxi (straight path, cell
    // size at its closest approach to the axis): criterion for adaptive sub-slicing
    Real max_cells(Real dxi) const;
    // length of the last push (0 before the first): multistep methods are only stable if the
    // step grows by at most ~2 from one push to the next, so sub-slicing relaxes gradually
    double last_step() const { return npush_ > 0 ? xi_cur_ - xi_slot_[(npush_ - 1) % ab_.order] : 0.0; }

    // ---- ionization (species with <name>.element or <name>.ionization_energies_eV)
    bool ionizable() const { return ionizable_; }
    int ab_order() const { return ab_.order; }
    const IonSpeciesInfo& ion_info() const { return ion_; }
    void set_laser(Real k0, bool circular) { ion_.P.laser = true; ion_.P.k0 = k0; ion_.P.circular = circular; }
    // slice kl (global k) of time step `step`: ionization with the fields f of the slice, beam impact
    // sources imp(kl, node, 3) (empty: none); new electrons are appended to `prod`
    IonizeResult ionize(const SliceFields& f, const View3D& imp, int kl, Real dxi, int k, int step, PlasmaSpecies& prod,
                        const View3D& eimp = View3D(), const EITargets& eit = EITargets());
    // electron-impact ionization (Ionization.hpp): projectile electrons deposit their rates per target and
    // energy bin; ionizable species deposit the density of every target level; projectiles lose energy
    void deposit_eimpact(const View3D& eimp, const EITargets& eit) const;
    void deposit_targets(const View2D& ntg) const;
    // discrete = true: each collision (ionization: Ip + W with W sampled; excitation: W_j) is drawn per
    // electron (straggling); false: the expected loss (continuous slowing down)
    void eloss(const View2D& ntg, const EITargets& eit, Real h, bool discrete, int k, int step, int sub);
    IonSpeciesInfo& ion_info_mut() { return ion_; }
    bool tail_dump() const { return tail_dump_; }
    void write_state(const std::string& fn) const;   // x y px py Delta w of the live particles (text)
    // <name>.ionization.born_dump: electrons born in this sweep, 8 numbers each (see ionize)
    std::vector<double>& born_records() { return born_rec_; }
    // charge density sum z w (gamma/Delta) / V of the ions (m = 0) -> out(kl, node, comp)
    void deposit_charge_state(const View3D& out, int kl, int comp) const;
    // sum z w and sum w over live particles (mean charge state)
    void charge_sums(double& zw, double& w) const;
    void ensure_capacity(int n);   // grow the particle arrays (keeps the contents)
    int num_loaded() const { return Nload_; }

    const std::string& name() const { return name_; }
    Real charge() const { return q_; }
    // maximum density (units of n0) for lab z in [z0, z1], r in [0, R] (sampled; adaptive time step)
    double max_density(double z0, double z1, double R) const;
    Real mass() const { return m_; }
    bool mobile() const { return mobile_; }
    bool frozen() const { return frozen_; }
    int  num_particles() const { return Np_; }
    int  lost_count() const;
    void reset_lost();

    // ---- trapping (<name>.trap = 1): particles beyond the quasi-static limit (gamma/Delta >
    // pusher.max_qsa_factor or Delta < pusher.delta_min) become particles of the beam trap_beam()
    bool trap() const { return trap_; }
    const std::string& trap_beam() const { return trap_beam_; }
    int trap_nphi() const { return trap_nphi_; }
    // particles trapped in this sweep, in index order: x y p_x p_y p_z xi_rel w per particle
    // (state at the slice where the limit was reached, xi_rel = xi - xi of the box head, w the
    // plasma weight = particles per unit length); clears the trap records. Returns the count.
    int collect_trapped(std::vector<double>& out);

    // complete particle state incl. the Adams-Bashforth history (hand-off to the downstream rank)
    void pack_state(std::vector<double>& buf) const;
    size_t unpack_state(const double* p);   // returns the number of doubles consumed

    View1D x_, y_, px_, py_, dl_, w_;
    IView1D fresh_;        // 1: (re)start the Adams-Bashforth history at the next push (born / charge changed)
    View1D lev_, wq_;      // ionizable species: charge state, split quantum (0: no splitting)

private:
    std::string name_;
    const RadialGrid& grid_;
    DensityProfile prof_;
    bool m1_ = false;
    Real q_ = -1, m_ = 1;
    bool mobile_ = true;
    int ppc_ = 4;
    int ntheta_ = 1;
    Parser uth_;          // thermal momentum spread uth(x,y,z) (m_s c), isotropic
    double udrift_[3] = {0, 0, 0};   // <name>.u_drift: momentum added at loading (tests)
    bool tail_dump_ = false;         // <name>.tail_dump: write the particles at the end of the box
    Real delta_min_ = 1e-3;
    Real max_qsa_ = 35;   // particles with gamma/Delta = 1/(1 - v_z) above this are removed (trapped)
    Real density_factor_ = 1;
    ABCoeffs ab_;
    // AB history bookkeeping (host; the same for all particles of the species, travels with them
    // between MPI ranks): number of pushes since the load, xi of the history point in each slot
    long npush_ = 0;
    double xi_cur_ = 0;
    double xi_slot_[5] = {0, 0, 0, 0, 0};
    int Np_ = 0;      // active particles (loaded + born in this sweep)
    int Nload_ = 0;   // loaded at the head of the box
    int cap_ = 0;     // allocated
    bool ionizable_ = false;
    bool frozen_ = false;     // ionizable species with <name>.mobile = 0: particles, but not pushed
    IonSpeciesInfo ion_;
    std::vector<double> born_rec_;
    IView1D cnt_e_, cnt_c_;   // ionization scratch (per particle)
    unsigned long long seed_ = 1;
    View3D hist_;   // (order, 5, Np): dx, dy, dpx, dpy, dDelta
    IView1D lost_;
    bool trap_ = false;
    std::string trap_beam_;
    int trap_nphi_ = 1;
    View1D tw_, txi_, tpz_;   // trap records: weight (0 = none), xi_rel and p_z at the trapping slice
    IView1D ntrap_;
};

} // namespace quarz
