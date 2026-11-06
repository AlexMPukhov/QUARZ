// Stochastic radiation reaction of beam particles (pusher.rr_model = stochastic | gaussian).
//
// Classical synchrotron emission (chi << 1) of a particle with charge Z e, mass M m_e, Lorentz factor gamma
// over a step dt (units: t in 1/omega_p, energies in M c^2), in the local constant-crossed-field picture:
//   chi    = K (|Z| / M^2) gamma sqrt(w),        w = (E + v x B)^2 - (v.E)^2 (fields in m_e c omega_p / e),
//            K = hbar omega_p / (m_e c^2)
//   Lambda = (5 / (2 sqrt 3)) Z^2 alpha (M / K) chi dt / gamma            (mean number of photons)
//   m1     = (4 sqrt 3 / 15) chi gamma,  m2 = (11/12) chi^2 gamma^2         (1st, 2nd moment of one photon)
// so that Lambda m1 = (2/3) Z^2 alpha (M/K) chi^2 dt is the Landau-Lifshitz energy loss (identical to the mean
// loss of pusher.beam = imp_rr) and Lambda m2 = (55 / (24 sqrt 3)) Z^2 alpha (M/K) chi^3 gamma dt the quantum
// excitation (Sokolov-Ternov).
//
// The energy loss S of the step (N ~ Poisson(Lambda) photons) is sampled from a zero-inflated Gamma law
// (N. Elkina, A. Pukhov, "Coarse-grained stochastic radiation reaction with one-sided quantum recoil"):
//   S = 0 with probability exp(-Lambda), otherwise S ~ Gamma(k, theta) with the conditional mean r1 and
//   variance s2 of the compound-Poisson sum:  r1 = Lambda m1 / q,  s2 = (q Lambda m2 - Lambda^2 m1^2 e^-Lambda) / q^2,
//   q = 1 - exp(-Lambda),  k = r1^2 / s2,  theta = s2 / r1.
// Mean and variance of S are exact, S >= 0 (no radiative energy gain). The classical synchrotron photon spectrum
// is itself close to a Gamma law (k = 0.30 for one photon), which is why the shape is good for every Lambda.
// rr_model = gaussian draws S = Lambda m1 + sqrt(Lambda m2) N(0, 1) instead (the Fokker-Planck model, which can
// gain energy); for comparison only.
#pragma once

#include "Ionization.hpp"   // ion_mix, ion_rand (counter-based random numbers)
#include "Types.hpp"

namespace quarz {
namespace rad {

constexpr Real ALPHA = 7.2973525693e-3;

enum Model { None = 0, Stochastic = 1, Gaussian = 2 };

struct Step { Real lam, m1, m2; };   // photon number, one-photon moments (units M c^2)

// chiK = K |Z| / M^2, nK = Z^2 alpha M / K  (precomputed per beam)
KOKKOS_INLINE_FUNCTION Real chi(Real gamma, Real w, Real chiK) { return chiK * gamma * Kokkos::sqrt(w); }

KOKKOS_INLINE_FUNCTION Step step(Real gamma, Real chi_, Real nK, Real dt) {
    Step s;
    s.lam = Real(1.4433756729740645) * nK * chi_ * dt / gamma;     // 5 / (2 sqrt 3)
    s.m1 = Real(0.46188021535170065) * chi_ * gamma;                // 4 sqrt 3 / 15
    s.m2 = Real(11.0 / 12.0) * chi_ * chi_ * gamma * gamma;
    return s;
}

// standard normal from two uniforms (Box-Muller, one of the pair)
KOKKOS_INLINE_FUNCTION Real normal(uint64_t key, uint64_t id, uint64_t d) {
    const Real u1 = Kokkos::fmax(ion_rand(key, id, d), Real(1e-300)), u2 = ion_rand(key, id, d + 1);
    return Kokkos::sqrt(Real(-2) * Kokkos::log(u1)) * Kokkos::cos(Real(6.283185307179586) * u2);
}

// Gamma(k, 1) variate, Marsaglia & Tsang (2000); k < 1 via Gamma(k + 1) U^(1/k). Draw numbers d0, d0 + 1, ...
// The acceptance probability is > 0.95, so the cap of 64 trials is never reached in practice (then: the mean).
KOKKOS_INLINE_FUNCTION Real gamma_variate(Real k, uint64_t key, uint64_t id, uint64_t d0) {
    Real boost = Real(1);
    if (k < Real(1)) {
        const Real u = Kokkos::fmax(ion_rand(key, id, d0), Real(1e-300));
        boost = Kokkos::exp(Kokkos::log(u) / k);
        k += Real(1);
    }
    const Real d = k - Real(1) / Real(3), c = Real(1) / Kokkos::sqrt(Real(9) * d);
    uint64_t n = d0 + 1;
    for (int it = 0; it < 64; ++it, n += 3) {
        const Real x = normal(key, id, n);
        Real v = Real(1) + c * x;
        if (v <= Real(0)) continue;
        v = v * v * v;
        const Real u = Kokkos::fmax(ion_rand(key, id, n + 2), Real(1e-300));
        if (Kokkos::log(u) < Real(0.5) * x * x + d - d * v + d * Kokkos::log(v)) return boost * d * v;
    }
    return boost * k;
}

// energy loss of one step (units M c^2)
KOKKOS_INLINE_FUNCTION Real sample_loss(const Step& s, int model, uint64_t key, uint64_t id) {
    if (!(s.lam > Real(0))) return Real(0);
    if (model == Gaussian) return s.lam * s.m1 + Kokkos::sqrt(s.lam * s.m2) * normal(key, id, 0);
    const Real e = Kokkos::exp(-s.lam);
    if (ion_rand(key, id, 0) < e) return Real(0);                 // no photon
    const Real q = -Kokkos::expm1(-s.lam);
    const Real r1 = s.lam * s.m1 / q;
    const Real s2 = (q * s.lam * s.m2 - s.lam * s.lam * s.m1 * s.m1 * e) / (q * q);
    if (!(s2 > Real(0))) return r1;
    return (s2 / r1) * gamma_variate(r1 * r1 / s2, key, id, 1);
}

// remove the energy S from the momentum u (along u); gamma >= 1 is kept (a loss larger than the kinetic energy,
// probability ~ Q(k, gamma / theta), stops the particle)
KOKKOS_INLINE_FUNCTION void apply_loss(Real& ux, Real& uy, Real& uz, Real S) {
    const Real p2 = ux * ux + uy * uy + uz * uz;
    if (!(p2 > Real(0)) || S == Real(0)) return;
    const Real g = Kokkos::sqrt(Real(1) + p2);
    Real gn = g - S;
    if (gn < Real(1)) gn = Real(1);
    const Real f = Kokkos::sqrt((gn - Real(1)) * (gn + Real(1)) / p2);
    ux *= f; uy *= f; uz *= f;
}

} // namespace rad
} // namespace quarz
