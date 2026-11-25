// QUARZ - quasi-static axisymmetric (r, xi) PIC code with non-uniform radial grid
// Common types and constants.
#pragma once

#include <Kokkos_Core.hpp>

namespace quarz {

using Real = double;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using HostSpace = Kokkos::HostSpace;

using View1D  = Kokkos::View<Real*, MemSpace>;
using View2D  = Kokkos::View<Real**, Kokkos::LayoutRight, MemSpace>;
using View3D  = Kokkos::View<Real***, Kokkos::LayoutRight, MemSpace>;
using IView1D = Kokkos::View<int*, MemSpace>;

using Range = Kokkos::RangePolicy<ExecSpace>;

constexpr Real PI = 3.14159265358979323846;

// Longitudinal coordinate.  The solver works internally with xi = t - z (head of the box at the
// smallest xi, slice 0; the sweep runs towards larger xi).  Everything the user sees -- deck
// keys, parsed functions of xi, beam files, native output files and logs -- uses the
// left-to-right convention xi_user = z - t = -xi (beam moving towards +xi, head at the largest
// xi_user).  Convert only at the input/output boundary with these two functions.
// (0 - x rather than -x: exact negation, but +0 instead of -0 for the head at xi = 0)
KOKKOS_INLINE_FUNCTION constexpr Real xi_from_user(Real xu) { return Real(0) - xu; }
KOKKOS_INLINE_FUNCTION constexpr Real xi_to_user(Real xi) { return Real(0) - xi; }

// true if the default execution space runs on the host (Serial/OpenMP/Threads)
inline constexpr bool exec_is_host() {
    return Kokkos::SpaceAccessibility<HostSpace, MemSpace>::accessible;
}

} // namespace quarz
