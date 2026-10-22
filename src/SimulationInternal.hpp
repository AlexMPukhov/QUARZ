// SimulationInternal.hpp: helpers shared by the Simulation sources (Simulation*.cpp): the note and the line
// format of the ionization log.
#pragma once
#include <array>
#include <iomanip>
#include <ostream>
#include <vector>

namespace quarz {
namespace detail {
inline const char* const ION_LOG_NOTE =
    "\n# born_weight: sum of the weights of the electrons created in the sweep (n0 (c/omega_p)^2 per c/omega_p)"
    "\n# born_per_m: the same as electrons per metre of plasma; born_p2: mean p_perp^2 at birth (m_e c)^2"
    "\n# z_tail: mean charge state at the end of the box\n";
inline void ion_line(std::ostream& o, int n, double t, const std::vector<std::array<double, 4>>& A, double perm) {
    o << n << " " << std::setprecision(10) << t;
    for (const auto& a : A)
        o << " " << a[0] << " " << a[0] * perm << " " << (a[0] > 0 ? a[1] / a[0] : 0.0) << " " << (a[3] > 0 ? a[2] / a[3] : 0.0);
    o << "\n";
}
} // namespace detail
} // namespace quarz
