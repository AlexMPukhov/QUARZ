// BeamKernels.hpp: device helpers of the beam deposition (Beam.cpp, BeamPush.cpp)
#pragma once
#include "Beam.hpp"

namespace quarz {
namespace detail {
// exponentially scaled modified Bessel functions e^{-z} I0(z), e^{-z} I1(z), z >= 0
// (Abramowitz & Stegun 9.8.1-9.8.4, |rel. error| < 2e-7)
KOKKOS_INLINE_FUNCTION void bessel_i01_scaled(Real z, Real& i0, Real& i1) {
    if (z < Real(3.75)) {
        const Real t = (z / Real(3.75)) * (z / Real(3.75));
        const Real I0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        const Real I1 = z * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        const Real e = Kokkos::exp(-z);
        i0 = I0 * e; i1 = I1 * e;
    } else {
        const Real u = Real(3.75) / z;
        const Real s = Real(1) / Kokkos::sqrt(z);
        i0 = s * (0.39894228 + u * (0.01328592 + u * (0.00225319 + u * (-0.00157565 + u * (0.00916281 + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377))))))));
        i1 = s * (0.39894228 + u * (-0.03988024 + u * (-0.00362018 + u * (0.00163801 + u * (-0.01031555 + u * (0.02282967 + u * (-0.02895312 + u * (0.01787654 - u * 0.00420059))))))));
    }
}
} // namespace detail
} // namespace quarz
