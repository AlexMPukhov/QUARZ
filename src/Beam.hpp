// Relativistic particle beams (drivers / witnesses): 3D macro-particles
// (x, y, p_x, p_y, p_z, xi, w) pushed in lab time t with a selectable momentum
// pusher (see Pushers.hpp), using the mode-decomposed fields stored on the
// (xi, r) grid.  They deposit rho - J_z, J_z, J+ and rho in azimuthal modes.
#pragma once

#include "FieldStore.hpp"
#include "Parser.hpp"
#include "RadialGrid.hpp"

#include <array>
#include <string>
#include <vector>

namespace quarz {

class Config;

struct BeamDiag {
    double t = 0, npart = 0, charge = 0, gamma_mean = 0, gamma_rms = 0, xi_mean = 0, xi_rms = 0;
    double r_rms = 0, emit_nx = 0, emit_ny = 0, x_mean = 0, y_mean = 0, sigma_x = 0, sigma_y = 0;
    long alive = 0;
};

struct BeamGrid {           // longitudinal grid of the box
    Real xi_min = 0, dxi = 0;
    int nxi = 0;               // global number of slices
    int k0 = 0, nloc = 0;      // slices owned by this rank: [k0, k0 + nloc)  (field arrays are local)
    bool ngp = false;          // nearest-slice (NGP) beam deposit/gather in xi instead of linear
};

enum class BeamPusher { Vay = 0, Boris = 1, HC = 2, IMP = 3, IMP_RR = 4 };

class Beam {
public:
    // qm != nullptr: an initially empty beam with charge qm[0] and mass qm[1] (trapped plasma
    // particles); the pusher, spin and scattering keys of <name> still apply
    Beam(const Config& cfg, const std::string& name, const RadialGrid& grid, const BeamGrid& bg, bool mode1,
         const double* qm = nullptr);

    // add this beam's sources to src(xi, r, comp)  (layout BeamComp)
    void deposit(const View3D& src) const;
    // impact-ionization sources imp(xi, r, 3) (m = 0):  sum z^2 n beta,  sum z^2 n L / beta,  sum z^2 n / beta,
    // L = ln(beta^2 gamma^2) - beta^2  (beam number density n in units of n0)
    void deposit_impact(const View3D& imp) const;
    // advance by dt with the stored fields fld(xi, r, comp)  (layout FieldComp)
    // pond: laser ponderomotive arrays (local slice, node, {<a^2>, d/dr, d/dxi}); empty = no laser
    // Leapfrog with a variable step: the momenta live at t_{n-1/2}; the kick spans
    // dt_kick = (dt_{n-1} + dt_n)/2 and the drift dt_drift = dt_n (equal for a constant dt).
    // diag (local slice, node, D_*): plasma densities for Coulomb scattering; step: random key
    void push(const View3D& fld, Real dt_kick, Real dt_drift, const View3D& pond = View3D(),
              const View3D& diag = View3D(), int step = 0);
    // one pusher step (move=false: explicit kick only, for the leapfrog start). Public because
    // CUDA extended lambdas may not live in private member functions.
    void advance(const View3D& fld, Real dt_kick, Real dt_drift, bool move, const View3D& pond,
                 const View3D& diag = View3D(), int step = 0, int i0 = 0);
    // new particles (7 doubles each: x y p_x p_y p_z xi w, momenta at the current time t_n) are
    // appended; if the leapfrog is running their momenta (and spins) are moved back to
    // t_n - dt_prev/2 with the fields fld of this step
    void add_new(const std::vector<double>& p7, const View3D& fld, Real dt_prev, const View3D& pond);
    // multiple Coulomb scattering on the plasma ions and electrons (<beam>.scattering)
    struct Scatter {
        bool on = false;
        Real kappa = 0;          // k_p r_e (q/m)^2 * scattering.factor: d<p_x^2>/dt = kappa n Z^2 L per target
        Real Z = 1, zeta = 1;    // nuclear charge, ion charge state (bound electrons Z - zeta)
        Real Li = -1, Le = -1;   // fixed Coulomb logarithms (< 0: computed per particle)
        Real bmax_i = 1;         // ion screening length (k_p^-1 if fully ionized, else Thomas-Fermi), c/omega_p
        Real a0 = 0;             // Bohr radius (bound electrons), c/omega_p
        Real RN = 0, lamC = 0;   // nuclear radius, reduced Compton wavelength / (m/m_e), c/omega_p
        Real lnbmax_i = 0, lnRN = 0, lnlamC = 0, lna0 = 0;   // their logarithms
        uint64_t seed = 1;
    };
    bool scattering() const { return sc_.on && !rigid_; }
    std::string scattering_description() const;
    // adaptive time step: min over live particles of max(gamma, gthr) * m / |q|
    // (omega_beta^2 = (|q|/m) n / (2 gamma)); +inf for rigid beams or with <beam>.adaptive_dt = 0
    double min_gamma_eff(double gthr) const;

    BeamDiag diagnostics(Real t) const;
    // weighted sums for (global) diagnostics: S[0] = sum w, S[1..15] moments, S[16] = live count
    std::array<double, 17> sums() const;
    BeamDiag from_sums(const std::array<double, 17>& S, Real t) const;   // S from sums() (any ranks summed)
    std::vector<double> slice_sums(double lo, double hi, int nbins) const;
    static void write_slices_file(const std::string& filename, Real t, Real q, double lo, double hi, int nbins,
                                  const std::vector<double>& sums);
    void dump(const std::string& filename, int stride = 1) const;   // every stride-th live particle
    Real charge() const { return q_; }
    Real mass() const { return m_; }
    bool adaptive_dt() const { return adaptive_dt_ && !rigid_; }
    bool passive() const { return passive_; }

    // decomposition along xi: ownership by nearest slice, migration to the downstream rank
    std::vector<double> packed(const Kokkos::View<int*, HostSpace>* sel = nullptr) const;
    // checkpoints: all particles in memory order, including removed ones (w = 0), so that a
    // restarted run is bit-identical; the leapfrog start flag
    std::vector<double> packed_all() const;
    bool started() const { return started_; }
    void set_started(bool s) { started_ = s; }
    void set_particles(const std::vector<double>& p);
    void restrict_to_local();
    std::vector<double> extract_outgoing();
    void append(const double* p, size_t n);
    int slice_of(double xi) const;

    const std::string& name() const { return name_; }
    bool rigid() const { return rigid_; }
    bool analytic() const { return analytic_; }
    int  num_particles() const { return Np_; }
    int  ncomp() const { return nc_; }              // packed doubles per particle (7, 10 with spin)
    bool spin() const { return spin_; }
    Real anomalous() const { return anom_; }
    std::array<double, 4> spin_sums() const;        // sum w, sum w s_x, s_y, s_z (live particles)
    long num_global() const { return np_global_; }   // particles of the whole beam at the start

private:
    void init_gaussian(const Config& cfg);
    void init_file(const std::string& filename);
    void init_parsed(const Config& cfg);
public:
    void deposit_analytic(const View3D& src) const;
    void deposit_analytic_parsed(const View3D& src) const;
private:

    std::string name_;
    const RadialGrid& grid_;
    BeamGrid bg_;
    bool m1_ = false;
    Real q_ = -1, m_ = 1;
    double gref_ = 0, xiref_ = 0;   // reference values for the moment sums
    void set_reference();
    // the generators draw the whole beam (same random sequence on every rank) but store only
    // the particles of the local slices; the reference values are those of the whole beam,
    // accumulated in generation order exactly as sums() would
    bool local_xi(double xi) const;
    struct RefAcc { double w = 0, wg = 0, wxi = 0; long n = 0; };
    void accumulate(RefAcc& a, const double* p) const;   // p = x y px py pz xi w
    void adopt_local(const std::vector<double>& all);    // whole beam (packed 7) -> local particles + refs
    void finish_init(std::vector<double>& local, const RefAcc& a);
    long np_global_ = 0;
    bool rigid_ = false;
    bool adaptive_dt_ = true;
    bool passive_ = false;      // <beam>.passive: test particles, no charge or current deposited   // <beam>.adaptive_dt: counts for the adaptive time step
    bool started_ = false;
    // analytic rigid beam: rho = q n0 exp(-|x_perp - c(xi)|^2 / 2 sigma^2) f(xi), c = centroid
    bool analytic_ = false;
    Real an_n0_ = 0, an_sr_ = 1, an_xi0_ = 0, an_sxi_ = 1, an_len_ = 0, an_gamma_ = 1, an_cut_head_ = -1e300;
    Real an_x0_ = 0, an_y0_ = 0, an_xs_ = 0, an_ys_ = 0;
    bool an_flat_ = false;
    // parsed profile: density(x, y, xi) in units of n0
    bool parsed_ = false;
    Parser dens_;
    BeamPusher pusher_ = BeamPusher::Vay;
    Scatter sc_;   // <beam>.scattering
    Real rr_ = 0;   // (2/3) r_e k_p q^2/m, 0 = no radiation reaction
    int Np_ = 0;
    View1D x_, y_, px_, py_, pz_, xi_, w_;
    View1D sx_, sy_, sz_;           // spin (unit vector), with <beam>.spin
    bool spin_ = false;
    Real anom_ = 0;                 // anomalous magnetic moment a = (g - 2)/2
    double s0_[3] = {0, 0, 1};      // initial spin direction
    int nc_ = 7;                    // doubles per packed particle: 7, or 10 with spin
    std::vector<View1D*> comps();
    std::vector<const View1D*> comps() const;
};

} // namespace quarz
