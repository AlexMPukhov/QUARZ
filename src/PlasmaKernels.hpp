// PlasmaKernels.hpp: particle geometry and field gather of the plasma push (PlasmaSpecies*.cpp)
#pragma once
#include "PlasmaSpecies.hpp"

namespace quarz {
namespace detail {

// geometry of one particle: radius, azimuth, hat-function weights
struct PGeo {
    Real r, c, s;          // radius, cos/sin of azimuth
    int j;                 // cell
    Real W[2], dW[2], Wor[2];   // W, dW/dr, W/r (W/r := 0 on the axis node)
};
KOKKOS_INLINE_FUNCTION PGeo geometry(const GridD& g, Real x, Real y) {
    PGeo p;
    p.r = Kokkos::sqrt(x * x + y * y);
    if (p.r > Real(0)) { p.c = x / p.r; p.s = y / p.r; } else { p.c = Real(1); p.s = Real(0); }
    Real t;
    g.weights(p.r, p.j, t);
    const Real ih = g.inv_h(p.j);
    p.W[0] = Real(1) - t; p.W[1] = t;
    p.dW[0] = -ih; p.dW[1] = ih;
    const Real ir = p.r > Real(0) ? Real(1) / p.r : Real(0);
    p.Wor[0] = (p.j == 0) ? Real(0) : p.W[0] * ir;
    p.Wor[1] = (p.j == 0) ? ih : p.W[1] * ir;      // W_1/r = 1/h_0 exactly in the first cell
    return p;
}
KOKKOS_INLINE_FUNCTION Real lin(const View1D& f, const PGeo& p) { return p.W[0] * f(p.j) + p.W[1] * f(p.j + 1); }

// fields at a particle
struct PFields { Real Wx, Wy, Bx, By, Bz, Ez, A, dA; };   // A = <a^2>, dA = d<a^2>/dr (laser)
KOKKOS_INLINE_FUNCTION PFields gather(const SliceFields& f, const PGeo& p, bool m1, bool needB) {
    PFields o;
    eval_vector(lin(f.wp.r0, p), lin(f.wp.i0, p), lin(f.wp.r1, p), lin(f.wp.i1, p), lin(f.wp.r2, p),
                lin(f.wp.i2, p), p.c, p.s, m1, o.Wx, o.Wy);
    o.Ez = eval_scalar(lin(f.ez.a0, p), lin(f.ez.ar, p), lin(f.ez.ai, p), p.c, p.s, m1);
    o.Bz = eval_scalar(lin(f.bz.a0, p), lin(f.bz.ar, p), lin(f.bz.ai, p), p.c, p.s, m1);
    if (needB)
        eval_vector(lin(f.bp.r0, p), lin(f.bp.i0, p), lin(f.bp.r1, p), lin(f.bp.i1, p), lin(f.bp.r2, p),
                    lin(f.bp.i2, p), p.c, p.s, m1, o.Bx, o.By);
    else
        o.Bx = o.By = Real(0);
    if (f.laser) { o.A = lin(f.aa, p); o.dA = lin(f.daa, p); }
    else o.A = o.dA = Real(0);
    return o;
}

} // namespace detail
} // namespace quarz
