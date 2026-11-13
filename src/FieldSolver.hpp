// Per-slice quasi-static field solver on the non-uniform radial grid, for
// azimuthal modes.  All equations reduce to scalar radial problems
//
//     (L_n - chi0) f = rhs,     L_n f = (1/r)(r f')' - n^2 f / r^2 ,   n = 0, 1, 2
//
// with the discretisations
//   n = 0 : lumped-mass linear FEM on the hat functions (conservative, exact Gauss law);
//           axis: regular;  wall: f(R) = 0
//   n >= 1: conservative finite volume of  L_n f = r^{n-1} d/dr[ r^{1-2n} d/dr( r^n f ) ];
//           axis: f(0) = 0;  wall: f(R) = 0  or (n = 1) the flux condition
//           (1/r) d(r f)/dr |_R = g   (used for B_theta of mode 0, g = J_z(R))
// Both are second order on smoothly stretched grids (tests/test_solvers.cpp).
#pragma once

#include "RadialGrid.hpp"
#include "Tridiag.hpp"

namespace quarz {

enum class OpKind { L0, L1D, L1F, L2D };

class FieldSolver {
public:
    FieldSolver(const RadialGrid& grid, TridiagMethod method);

    // (L - chi) f = rhs.  chi may be an empty View (then chi = 0); wall (only for L1F) supplies g = wall(N).
    //  the right-hand side used is scale * rhs.
    void solve(OpKind op, const View1D& rhs, const View1D& f, const View1D& chi = View1D(),
               const View1D& wall = View1D(), Real scale = Real(1)) const;

    // node derivative df/dr (2nd order on smooth non-uniform grids); df/dr(0) = 0 (even functions)
    void gradient(const View1D& f, const View1D& df) const;
    // derivative of a function vanishing on the axis (odd): df/dr(0) = f_1 / r_1
    void gradient_odd(const View1D& f, const View1D& df) const;
    // f / r, with the axis value f_1 / r_1 (for functions vanishing linearly on the axis)
    void over_r(const View1D& f, const View1D& out) const;

    // Radial smoothing filter  f <- (1 - div a^2 grad)^{-1} f  for angular number n = 0, 1, 2
    // (the scalar filter applied to each Cartesian component / azimuthal mode). a2mid: a^2 at the
    // cell mid points (N values). n = 0: lumped FEM with a zero-flux wall (conserves sum V_j f_j and
    // constants); n >= 1: finite volume, the axis and wall values are passed through unchanged
    // (the wall value of S enters the flux condition of B_theta). Green's function in 2D
    // for constant a: K_0(r/a) / (2 pi a^2).
    void set_filter(const std::vector<Real>& a2mid);
    bool filter_on() const { return filter_on_; }
    void filter(int n, const View1D& f) const;

    const RadialGrid& grid() const { return grid_; }
    // tridiagonal coefficients of the m = 0 operator L0 (rows 0..N-1; row N is the Dirichlet row 0 1 0)
    void l0_coefficients(View1D& a, View1D& b, View1D& c) const { a = L0_.a; b = L0_.b; c = L0_.c; }
    // the same for L_n, any n >= 0 (n >= 1: finite volume, rows 0 and N Dirichlet; laser envelope with e^{i n theta})
    void ln_coefficients(int n, View1D& a, View1D& b, View1D& c) const;

private:
    struct Op { View1D a, b, c; bool flux = false; bool axis_regular = false; };
    const Op& op(OpKind k) const;

    const RadialGrid& grid_;
    TridiagSolver tri_;
    Op L0_, L1D_, L1F_, L2D_;
    Op F_[3];                     // smoothing filter operators, n = 0, 1, 2
    bool filter_on_ = false;
    View1D fw_;                   // filter work array
    Real flux_coef_ = 0;          // coefficient of g in the last row of L1F
    View1D bw_, dw_;              // work arrays
    View1D gc_m_, gc_0_, gc_p_;   // gradient stencils
    View1D inv_r_;                // 1/r_j (axis: 1/r_1)
};

} // namespace quarz
