// Laser envelope solver (axisymmetric, m = 0) on the non-uniform radial grid.
//
// The laser vector potential is  a(r, xi, t) = Re[ a^(r, xi, t) exp(-i k0 xi) ],  xi = t - z,
// k0 = omega0 / omega_p.  Without the second time derivative (slowly varying in the frame
// moving with c) the envelope obeys
//
//     [ grad_perp^2 + 2 d/dt ( i k0 - d/dxi ) ] a^ = chi a^ ,   chi = sum_s q_s^2 n_s / (m_s gamma_s),
//
// where chi is the plasma susceptibility (the same chi = sum q^2 w / (m Delta) per volume that
// enters the B_perp solve).  Discretisation, time step n -> n+1 (dt of the beam push), for the
// increment X = a^(n+1) - a^n:
//   * Crank-Nicolson in t:   dX/dxi = G = i k0 X + (dt/4) (grad_perp^2 - chi)(X + 2 a^n);
//   * this ODE in xi is integrated with the trapezoidal rule from slice k-1 to slice k, which
//     the sweep from the head of the box has just produced (the laser must vanish at the head
//     of the box).  Both directions are centred: second order; in vacuum every Fourier mode
//     keeps its amplitude (|g| = 1) and the march in xi damps (|z| < 1) instead of amplifying;
//   * grad_perp^2 = the m = 0 operator L0 of the field solver (lumped FEM on the hat functions).
// Each slice is one complex tridiagonal solve (Thomas, one thread; dominated by the plasma
// work for typical grids).
//
// Coupling to the plasma (ponderomotive force, time-averaged): with <a^2> = |a^|^2 / 2
// (linear polarisation; |a^|^2 for circular) the plasma particles use
//     gamma = (1 + p_perp^2 + qhat^2 <a^2> + Delta^2) / (2 Delta),
//     dp_perp/dxi  += - qhat^2 grad_perp <a^2> / (2 Delta)          (qhat = q/m)
// and Delta = gamma - p_z is unchanged (d/dt + d/dz of a function of xi vanishes).  Beam
// particles get the kick  du/dt = - qhat^2 grad <a^2> / (2 gamma_bar),  gamma_bar^2 = 1 + u^2 + qhat^2 <a^2>,
// with d/dz = -d/dxi.  The ponderomotive potential is evaluated at time level n.
//
// Vortex: laser.l = l >= 1 gives an LG_0l pulse a^(r) e^{i l theta}; |a^|^2 stays axisymmetric, grad_perp^2 -> L_l.
// Limitations: one azimuthal index per run (the laser drives only m = 0 of the plasma, also with modes = 1);
// the d/dxi discretisation is accurate while the local wavenumber stays close to k0
// (|k - k0| dxi << 1); no second time derivative (no backward-propagating light).
#pragma once

#include "Beam.hpp"
#include "Config.hpp"
#include "FieldSolver.hpp"
#include "RadialGrid.hpp"
#include "SliceData.hpp"

#include <array>
#include <string>
#include <vector>

namespace quarz {

class Laser {
public:
    Laser(const Config& cfg, const RadialGrid& grid, const FieldSolver& solver, const BeamGrid& box, Real t0, Real dt);

    Real k0() const { return k0_; }
    bool circular() const { return circular_; }
    std::string description() const;

    // start of a sweep: the envelope advanced in the previous sweep becomes the current one
    void begin_step();
    // from the upstream rank: a^n of slices k0-2, k0-1 and the carry (X, G) of slice k0-1: 8 M doubles
    void set_guard(const double* p);
    void get_guard(std::vector<double>& buf) const;   // the same for the last local slices
    static size_t guard_size(int M) { return 8 * static_cast<size_t>(M); }

    // slice kl: <a^2>, d<a^2>/dr (into f.aa, f.daa) and the beam ponderomotive arrays
    void prepare_slice(int kl, SliceFields& f);
    // slice kl: advance the envelope to the next time level with the plasma susceptibility chi0
    void advance_slice(int kl, const View1D& chi0);
    void end_sweep() { advanced_ = true; }

    // (local slice, node, 3): <a^2>, d<a^2>/dr, d<a^2>/dxi at the current time (beam push)
    const View3D& ponderomotive() const { return pond_; }
    // (local slice, node, 2): Re, Im of the current envelope
    const View3D& envelope() const { return a_; }
    // checkpoints: the envelope at the start of the next step (a^n+1 once the sweep is done)
    const View3D& current() const { return advanced_ ? an_ : a_; }
    void restore(const View3D& a) { Kokkos::deep_copy(a_, a); advanced_ = false; }

    // per-rank sums for diagnostics: {sum |a|^2 dV, sum |a|^2 xi dV, sum |a|^2 r^2 dV, max |a|, sum |a|^2 xi^2 dV}
    std::array<double, 5> sums() const;

private:
    const RadialGrid& grid_;
    BeamGrid box_;
    Real k0_ = 1, dt_ = 0;
    bool circular_ = false;
    bool advanced_ = false;
    int l_ = 0;          // laser.l: azimuthal index of an LG_0l vortex, a^ e^{i l theta} (0 = Gaussian)
    int xi_scheme_ = 1;  // laser.xi_scheme: 0 trapezoid, 1 bdf2 (default), 2 euler (backward differences)
    View3D a_, an_;      // (KL, M, 2): envelope at time n and n+1
    View3D pond_;        // (KL, M, 3)
    View3D guard_;       // (4, M, 2): [a^n at k0-2, a^n at k0-1, X(k0-1), G(k0-1)] from the upstream rank
    View3D carry_;       // (2, M, 2): X, G of the previous slice (bdf2/euler: X_{k-1}, X_{k-2})
    View1D la_, lb_, lc_;                   // L0 coefficients
    View1D aa_tmp_;                         // scratch
    View1D cpr_, cpi_, dpr_, dpi_;          // Thomas scratch (complex)
    const FieldSolver& solver_;
};

} // namespace quarz
