// Stochastic radiation reaction (src/Radiation.hpp): the zero-inflated Gamma sampler of the step energy loss.
//   1. the photon statistics reproduce the Landau-Lifshitz loss of imp_rr and the Sokolov-Ternov excitation
//   2. sampler: P(S = 0) = exp(-Lambda), E[S] = Lambda m1, Var[S] = Lambda m2 exactly, S >= 0; the Gaussian model
//      has the same two moments but gains energy
//   3. shape: Wasserstein distance to the exact compound-Poisson sum of classical synchrotron photons (independent
//      reference: photon energies drawn from the spectrum int_y^inf K_{5/3}, std::cyl_bessel_k)
#include "Radiation.hpp"
#include "Pushers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace quarz;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static void photon_statistics() {
    std::printf("1. Photon statistics vs Landau-Lifshitz (imp_rr) and the quantum excitation\n");
    const double n0 = 1e19, kp = 5.64146e4 * std::sqrt(n0) / 2.99792458e10, re = 2.8179403262e-13;
    const double K = kp * 3.8615926796e-11, rr = 2.0 / 3.0 * re * kp;   // electron: q = -1, m = 1
    const double chiK = K, nK = rad::ALPHA / K;
    const push::V3 u{0.3, -0.2, 2e5}, E{0.05, 0.02, -0.3}, B{-0.01, 0.04, 0.0};
    const double g = std::sqrt(1 + push::dot(u, u)), dt = 7.0;
    const push::V3 v = (1 / g) * u, F = E + push::cross(v, B);
    const double w = push::dot(F, F) - push::dot(v, E) * push::dot(v, E);
    const double chi = rad::chi(g, w, chiK);
    const rad::Step s = rad::step(g, chi, nK, dt);
    const double ll = push::rr_rate(u, E, B, -1.0, rr) * push::dot(u, v) * dt;   // -dgamma = nu u.v dt
    const double st = 55.0 / (24.0 * std::sqrt(3.0)) * rad::ALPHA * chi * chi * chi * g / K * dt;
    std::printf("    chi = %.3e, Lambda = %.4f: Lambda m1 / LL loss - 1 = %.1e, Lambda m2 / (55/(24 sqrt3)) alpha chi^3 gamma dt - 1 = %.1e\n",
                chi, s.lam, s.lam * s.m1 / ll - 1, s.lam * s.m2 / st - 1);
    check(std::abs(s.lam * s.m1 / ll - 1) < 1e-8, "mean loss = Landau-Lifshitz loss of imp_rr (ultrarelativistic)");
    check(std::abs(s.lam * s.m2 / st - 1) < 1e-12, "variance = Sokolov-Ternov quantum excitation");
}

static void sampler_moments() {
    std::printf("2. Sampler: zero probability, mean, variance (2e6 draws each, statistical tolerance 5 sigma)\n");
    const int N = 2000000;
    for (double lam : {0.02, 0.3, 3.0, 30.0}) {
        const rad::Step s{lam, 0.7, 0.7 * 0.7 * 4.296875};   // m2 / m1^2 = (11/12) / (4 sqrt3/15)^2 (synchrotron photon)
        for (int model : {rad::Stochastic, rad::Gaussian}) {
            double zero = 0, s1 = 0, s2 = 0, s4 = 0, neg = 0, mn = 1e300;
            for (int i = 0; i < N; ++i) {
                const double S = rad::sample_loss(s, model, 0x1234 + model, static_cast<uint64_t>(i));
                zero += S == 0;
                neg += S < 0;
                mn = std::min(mn, S);
                s1 += S; s2 += S * S; s4 += S * S * S * S;
            }
            const double mean = s1 / N, var = s2 / N - mean * mean, M1 = lam * s.m1, M2 = lam * s.m2;
            const double em = std::sqrt(M2 / N), ev = std::sqrt((s4 / N) / N);
            std::printf("    %-10s Lambda = %5.2f: P(0) = %.5f (exp(-L) %.5f), mean/(L m1) = %.5f, var/(L m2) = %.5f, "
                        "P(S<0) = %.4f\n", model == rad::Stochastic ? "stochastic" : "gaussian", lam, zero / N,
                        std::exp(-lam), mean / M1, var / M2, neg / N);
            char m[160];
            std::snprintf(m, sizeof m, "%s, Lambda = %g: mean and variance exact", model == rad::Stochastic ? "stochastic" : "gaussian", lam);
            check(std::abs(mean - M1) < 5 * em && std::abs(var - M2) < 5 * ev, m);
            if (model == rad::Stochastic) {
                const double p0 = std::exp(-lam);
                check(std::abs(zero / N - p0) < 5 * std::sqrt(p0 * (1 - p0) / N) + 1e-12 && mn >= 0,
                      "  zero-loss probability exp(-Lambda), no energy gain");
            } else {
                check(neg > 0, "  gaussian model gains energy (expected)");
            }
        }
    }
}

// exact compound Poisson with classical synchrotron photons, y = omega / (chi gamma)
struct Synchrotron {
    std::vector<double> y, cdf;
    Synchrotron() {
        // number spectrum in x = omega/omega_c: n(x) = int_x^inf K_{5/3}(t) dt = 2 K_{2/3}(x) - int_x^inf K_{1/3}
        const int n = 6000;
        const double lo = std::log(1e-9), hi = std::log(60.0);
        std::vector<double> x(n), f(n);
        for (int i = 0; i < n; ++i) x[i] = std::exp(lo + (hi - lo) * i / (n - 1));
        // int_x^inf K_{1/3}: trapezoid on the log grid from the top
        std::vector<double> I(n, 0.0);
        for (int i = n - 2; i >= 0; --i)
            I[i] = I[i + 1] + 0.5 * (std::cyl_bessel_k(1.0 / 3, x[i]) * x[i] + std::cyl_bessel_k(1.0 / 3, x[i + 1]) * x[i + 1]) *
                                  (hi - lo) / (n - 1);
        for (int i = 0; i < n; ++i) f[i] = 2 * std::cyl_bessel_k(2.0 / 3, x[i]) - I[i];
        // cumulative number, n(x) ~ c x^(-2/3) below x[0]
        cdf.assign(n, 0.0);
        cdf[0] = 3 * f[0] * x[0];
        for (int i = 1; i < n; ++i) cdf[i] = cdf[i - 1] + 0.5 * (f[i] * x[i] + f[i - 1] * x[i - 1]) * (hi - lo) / (n - 1);
        for (double& c : cdf) c /= cdf.back();
        y.resize(n);
        for (int i = 0; i < n; ++i) y[i] = 1.5 * x[i];   // omega_c = (3/2) chi gamma
    }
    double draw(std::mt19937_64& g) const {
        const double u = std::uniform_real_distribution<double>(0, 1)(g);
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        if (it == cdf.begin()) return y[0] * std::pow(u / cdf[0], 3.0);
        if (it == cdf.end()) return y.back();
        const size_t i = it - cdf.begin();
        const double t = (u - cdf[i - 1]) / (cdf[i] - cdf[i - 1]);
        return y[i - 1] + t * (y[i] - y[i - 1]);
    }
};

static void sampler_shape() {
    std::printf("3. Shape vs the exact compound-Poisson sum of synchrotron photons (chi gamma = 1, 4e5 draws):\n"
                "   Wasserstein-1 distance / sqrt(Lambda m2)\n");
    const Synchrotron sy;
    std::mt19937_64 g(7);
    const int N = 400000;
    double m1 = 0, m2 = 0;
    {
        std::vector<double> X(N);
        for (double& x : X) x = sy.draw(g);
        for (double x : X) { m1 += x; m2 += x * x; }
        m1 /= N; m2 /= N;
        std::printf("    reference photons: m1 = %.4f (4 sqrt3/15 = %.4f), m2 = %.4f (11/12 = %.4f)\n", m1,
                    4 * std::sqrt(3.0) / 15, m2, 11.0 / 12);
        check(std::abs(m1 / (4 * std::sqrt(3.0) / 15) - 1) < 0.01 && std::abs(m2 / (11.0 / 12) - 1) < 0.02,
              "reference spectrum has the moments used by the sampler");
    }
    for (double lam : {0.3, 1.0, 3.0, 10.0, 100.0}) {
        const rad::Step s = rad::step(1.0, 1.0, 1.0, lam / 1.4433756729740645);   // gamma = chi = 1: Lambda = lam
        std::vector<double> ex(N), zg(N), ga(N);
        std::poisson_distribution<int> P(lam);
        for (int i = 0; i < N; ++i) {
            double S = 0;
            for (int k = P(g); k > 0; --k) S += sy.draw(g);
            ex[i] = S;
            zg[i] = rad::sample_loss(s, rad::Stochastic, 99, static_cast<uint64_t>(i));
            ga[i] = rad::sample_loss(s, rad::Gaussian, 98, static_cast<uint64_t>(i));
        }
        std::sort(ex.begin(), ex.end()); std::sort(zg.begin(), zg.end()); std::sort(ga.begin(), ga.end());
        double wz = 0, wg = 0;
        for (int i = 0; i < N; ++i) { wz += std::abs(ex[i] - zg[i]); wg += std::abs(ex[i] - ga[i]); }
        const double sd = std::sqrt(lam * s.m2);
        wz /= N * sd; wg /= N * sd;
        std::printf("    Lambda = %5.1f: zero-inflated Gamma %.4f, Gaussian %.4f\n", lam, wz, wg);
        char m[128];
        std::snprintf(m, sizeof m, "Lambda = %g: Gamma within 0.04 sigma of the exact distribution, better than Gaussian", lam);
        check(wz < 0.04 && wz < wg, m);
    }
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    photon_statistics();
    sampler_moments();
    sampler_shape();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
