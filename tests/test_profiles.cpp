// Position-dependent momentum spreads: parsed beams (ux_std(x,y,xi), ...) and plasma
// temperature (<species>.uth(x,y,z)).  Checks the sampled particle statistics.
#include "Beam.hpp"
#include "Config.hpp"
#include "PlasmaSpecies.hpp"
#include "Profiles.hpp"
#include "RadialGrid.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace quarz;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// weighted standard deviation of v in bins of key
static void binned_std(const std::vector<double>& key, const std::vector<double>& v, const std::vector<double>& w,
                       double lo, double hi, int nb, std::vector<double>& centre, std::vector<double>& sd,
                       std::vector<double>& mean) {
    std::vector<double> s0(nb, 0), s1(nb, 0), s2(nb, 0);
    for (size_t i = 0; i < key.size(); ++i) {
        const int b = static_cast<int>((key[i] - lo) / (hi - lo) * nb);
        if (b < 0 || b >= nb) continue;
        s0[b] += w[i]; s1[b] += w[i] * v[i]; s2[b] += w[i] * v[i] * v[i];
    }
    centre.clear(); sd.clear(); mean.clear();
    for (int b = 0; b < nb; ++b) {
        if (s0[b] <= 0) continue;
        const double m = s1[b] / s0[b];
        centre.push_back(lo + (b + 0.5) * (hi - lo) / nb);
        mean.push_back(m);
        sd.push_back(std::sqrt(std::max(0.0, s2[b] / s0[b] - m * m)));
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    RadialGrid g(RadialGrid::make_uniform(2.0, 0.01));
    // internal box xi = t - z in [-10, 0], i.e. user xi = z - t in [0, 10] (deck keys and dumps use the user xi)
    BeamGrid bg; bg.xi_min = -10; bg.dxi = 0.01; bg.nxi = 1001;

    std::printf("Parsed beam with position-dependent momentum spread\n");
    {
        Config cfg;
        cfg.set("b.profile", "parsed");
        cfg.set("b.density(x,y,xi)", "exp(-(x^2+y^2)/(2*0.2^2))");
        cfg.set("b.xi_range", "4 6");
        cfg.set("b.r_max", "0.8");
        cfg.set("b.ppc", "4 2");
        cfg.set("b.ntheta", "16");
        cfg.set("my_constants.a", "0.5");
        cfg.set("b.ux_std(x,y,xi)", "0.01 + a*abs(x)");          // grows with |x|
        cfg.set("b.uy_std", "0.02");                               // plain number
        cfg.set("b.uz_std(x,y,xi)", "10*(xi - 4)");                // grows along the bunch
        cfg.set("b.uz_mean(x,y,xi)", "1000 + 50*(xi - 5)");        // chirp
        Beam b(cfg, "b", g, bg, true);
        const std::string fn = "test_profiles_beam.bin";
        b.dump(fn);
        FILE* f = std::fopen(fn.c_str(), "rb");
        int32_t n = 0;
        if (std::fread(&n, 4, 1, f) != 1) n = 0;
        std::vector<double> X(n), Y(n), PX(n), PY(n), PZ(n), XI(n), W(n);
        for (int i = 0; i < n; ++i) {
            double rec[7];
            if (std::fread(rec, sizeof(rec), 1, f) != 1) break;
            X[i] = rec[0]; Y[i] = rec[1]; PX[i] = rec[2]; PY[i] = rec[3]; PZ[i] = rec[4]; XI[i] = rec[5]; W[i] = 1.0;
        }
        std::fclose(f);
        std::remove(fn.c_str());
        std::printf("    %d macro-particles\n", n);
        std::vector<double> ax(n);
        for (int i = 0; i < n; ++i) ax[i] = std::fabs(X[i]);
        std::vector<double> c, sd, m;
        binned_std(ax, PX, W, 0.0, 0.6, 6, c, sd, m);
        double err = 0;
        for (size_t k = 0; k < c.size(); ++k) {
            // expected rms in a bin of |x|: sqrt(< (0.01 + 0.5|x|)^2 >) ~ value at the bin centre
            const double ex = 0.01 + 0.5 * c[k];
            std::printf("    |x| = %.2f   std(p_x) = %.4f   expected %.4f\n", c[k], sd[k], ex);
            err = std::max(err, std::fabs(sd[k] / ex - 1));
        }
        check(err < 0.05, "std(p_x) follows 0.01 + 0.5|x| (within 5 %)");
        binned_std(XI, PY, W, 4.0, 6.0, 4, c, sd, m);
        double e2 = 0;
        for (size_t k = 0; k < c.size(); ++k) e2 = std::max(e2, std::fabs(sd[k] / 0.02 - 1));
        check(e2 < 0.03, "plain number uy_std = 0.02");
        binned_std(XI, PZ, W, 4.0, 6.0, 8, c, sd, m);
        double e3 = 0, e4 = 0;
        for (size_t k = 0; k < c.size(); ++k) {
            const double ex = std::sqrt(100.0 * ((c[k] - 4) * (c[k] - 4)) + 100.0 * 0.25 * 0.25 / 12 + 2500.0 * 0.25 * 0.25 / 12);
            std::printf("    xi = %.3f  std(p_z) = %.3f  expected %.3f   mean(p_z) = %.2f  expected %.2f\n", c[k], sd[k], ex, m[k],
                        1000 + 50 * (c[k] - 5));
            e3 = std::max(e3, std::fabs(sd[k] / ex - 1));
            e4 = std::max(e4, std::fabs(m[k] - (1000 + 50 * (c[k] - 5))));
        }
        check(e3 < 0.06, "std(p_z) follows 10 (xi - 4) (within 6 %, incl. the chirp across the bin)");
        check(e4 < 0.5, "mean p_z follows the chirp 1000 + 50 (xi - 5)");
    }

    std::printf("Plasma temperature uth(x,y,z)\n");
    for (int m1 = 0; m1 <= 1; ++m1) {
        Config cfg;
        cfg.set("e.ppc", "64");
        cfg.set("e.uth(x,y,z)", "0.001 + 0.01*sqrt(x^2+y^2)*(1 + z/100)");
        DensityProfile prof;
        PlasmaSpecies e(cfg, "e", g, prof, m1 == 1);
        e.load(50.0, 7);
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), e.x_);
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), e.y_);
        auto hpx = Kokkos::create_mirror_view_and_copy(HostSpace(), e.px_);
        const int n = e.num_particles();
        // normalise each momentum by the requested local spread -> unit Gaussian in every radial bin
        std::vector<double> R(n), PX(n), W(n, 1.0);
        for (int i = 0; i < n; ++i) {
            R[i] = std::sqrt(hx(i) * hx(i) + hy(i) * hy(i));
            PX[i] = hpx(i) / (0.001 + 0.01 * R[i] * 1.5);
        }
        std::vector<double> c, sd, m;
        binned_std(R, PX, W, 0.0, 2.0, 5, c, sd, m);
        double err = 0;
        for (size_t k = 0; k < c.size(); ++k) {
            std::printf("    modes=%d  r = %.2f  std(p_x / uth(r, z)) = %.4f  (expected 1)\n", m1, c[k], sd[k]);
            err = std::max(err, std::fabs(sd[k] - 1));
        }
        check(err < 0.05, std::string("plasma temperature follows uth(r, z) (modes = ") + (m1 ? "1)" : "0)"));
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
