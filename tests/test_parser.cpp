// Unit tests of the expression parser: host and device evaluation, precedence,
// functions, constants, error messages.
#include "Config.hpp"
#include "Parser.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace quarz;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    const std::map<std::string, double> C = {{"kp", 2.0}, {"L", 3.0}, {"w0", 0.5}};
    struct Case { std::string expr; std::function<double(double, double, double)> f; };
    const double pi = PI;
    std::vector<Case> cases = {
        {"1 + 2*3 - 4/2", [](double, double, double) { return 5.0; }},
        {"-x^2", [](double x, double, double) { return -x * x; }},
        {"2^-1", [](double, double, double) { return 0.5; }},
        {"2^3^2", [](double, double, double) { return 512.0; }},
        {"x**3 - y", [](double x, double y, double) { return x * x * x - y; }},
        {"(x+y)*(x-y)/(1+z*z)", [](double x, double y, double z) { return (x + y) * (x - y) / (1 + z * z); }},
        {"exp(-(x^2+y^2)/(2*w0^2)) * cos(kp*z)",
         [](double x, double y, double z) { return std::exp(-(x * x + y * y) / (2 * 0.25)) * std::cos(2 * z); }},
        {"if(z < L, z/L, 1)", [](double, double, double z) { return z < 3 ? z / 3 : 1.0; }},
        {"(z>=0)*(z<=L) + 2*(x>0 && y>0) - (x<0 || y<0)",
         [](double x, double y, double z) { return (z >= 0 && z <= 3) + 2.0 * (x > 0 && y > 0) - (x < 0 || y < 0); }},
        {"sqrt(x*x+y*y) <= 1 ? 0 : 0", nullptr},   // ternary syntax is not supported -> must throw
        {"atan2(y,x) + min(x,y) - max(x,z) + mod(z, 0.7) + abs(y)",
         [](double x, double y, double z) {
             return std::atan2(y, x) + std::min(x, y) - std::max(x, z) + std::fmod(z, 0.7) + std::fabs(y);
         }},
        {"tanh(x) + sinh(y)/cosh(y) + erf(z) + floor(x) + ceil(y) + heaviside(z-0.5)",
         [](double x, double y, double z) {
             return std::tanh(x) + std::sinh(y) / std::cosh(y) + std::erf(z) + std::floor(x) + std::ceil(y) +
                    (z > 0.5 ? 1.0 : (z < 0.5 ? 0.0 : 0.5));
         }},
        {"pi*e + log(2) + log10(100) + pow(2, 0.5)",
         [pi](double, double, double) { return pi * std::exp(1.0) + std::log(2.0) + 2.0 + std::sqrt(2.0); }},
        {"!(x > 0) + (x != y) + (x == x)", [](double x, double y, double) { return (x <= 0) + (x != y) + 1.0; }},
        {"0.5*(1 - cos(pi*z/L))^2 * (abs(x) < 1e-3 + 2)", [pi](double x, double, double z) {
             const double c = 1 - std::cos(pi * z / 3); return 0.5 * c * c * (std::fabs(x) < 2.001); }},
    };
    std::printf("Expression parser\n");
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> U(-2, 2);
    std::vector<std::array<double, 3>> pts(200);
    for (auto& p : pts) p = {U(rng), U(rng), std::fabs(U(rng)) * 2};
    for (const auto& cs : cases) {
        if (!cs.f) {
            bool threw = false;
            try { Parser p(cs.expr, {"x", "y", "z"}, C); } catch (const std::exception& e) { threw = true; std::printf("      (%s)\n", e.what()); }
            check(threw, "rejects \"" + cs.expr + "\"");
            continue;
        }
        Parser p(cs.expr, {"x", "y", "z"}, C);
        double err = 0;
        for (const auto& q : pts) err = std::max(err, std::fabs(p(q[0], q[1], q[2]) - cs.f(q[0], q[1], q[2])));
        char m[64]; std::snprintf(m, sizeof m, "  (max err %.1e)", err);
        check(err < 1e-12, "\"" + cs.expr + "\"" + m);
    }

    // device evaluation must give the same numbers: bit-identical on host backends (the same math
    // library); on GPUs the device exp/sin/pow (1-2 ulp) and fused multiply-adds differ from the host
    // in the last bits, so there a few ulp relative to the function's scale are allowed
    {
        Parser p("exp(-(x^2+y^2)/(2*w0^2)) * if(z < L, sin(pi*z/(2*L))^2, 1) + 0.1*x*y", {"x", "y", "z"}, C);
        const ParserExec e = p.exec();
        const int n = 1000;
        View1D in("in", 3 * n), out("out", n);
        auto hin = Kokkos::create_mirror_view(in);   // the points are computed on the host: identical inputs
        for (int i = 0; i < n; ++i) {
            hin(3 * i) = -2 + 4.0 * i / n; hin(3 * i + 1) = 1 - 2.0 * i / n; hin(3 * i + 2) = 6.0 * i / n;
        }
        Kokkos::deep_copy(in, hin);
        Kokkos::parallel_for("parser_dev", Range(0, n), KOKKOS_LAMBDA(int i) {
            out(i) = e(in(3 * i), in(3 * i + 1), in(3 * i + 2));
        });
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), out);
        double err = 0, scale = 0;
        for (int i = 0; i < n; ++i) {
            const double f = p(hin(3 * i), hin(3 * i + 1), hin(3 * i + 2));
            err = std::max(err, std::fabs(h(i) - f));
            scale = std::max(scale, std::fabs(f));
        }
        std::printf("    device vs host: max |difference| = %.3e (max |f| = %.3e, backend %s)\n", err, scale,
                    ExecSpace::name());
        if (exec_is_host()) check(err == 0.0, "device (Kokkos kernel) evaluation identical to host (host backend)");
        else check(err <= 1e-14 * scale, "device (Kokkos kernel) evaluation equal to host within 1e-14 relative (GPU backend)");
    }

    // constant folding: an expression without variables is a single constant
    {
        Parser p("2*pi*kp/L + sqrt(16)", {"x"}, C);
        check(p.is_constant() && p.exec().n == 1, "constant folding of variable-free sub-expressions");
    }

    // my_constants (in any order, referring to each other) and read_function
    {
        Config cfg;
        cfg.set("my_constants.Lup", "2*L0");
        cfg.set("my_constants.L0", "kp0^2");
        cfg.set("my_constants.kp0", "3");
        cfg.set("plasma.density(x, y, zz)", "\"if(zz < Lup, zz/Lup, 1)\"");
        auto c = read_constants(cfg);
        check(c.at("Lup") == 18.0, "my_constants resolved in dependency order (Lup = 18)");
        Parser p;
        const bool found = read_function(cfg, "plasma.density", 3, p, {"x", "y", "z"});
        check(found && std::fabs(p(0, 0, 9) - 0.5) < 1e-15 && p(0, 0, 40) == 1.0,
              "read_function: key variables (x, y, zz), quoted expression");
        Config bad;
        bad.set("my_constants.a", "b + 1");
        bad.set("my_constants.b", "a * 2");
        bool threw = false;
        try { read_constants(bad); } catch (const std::exception& e) { threw = true; std::printf("      (%s)\n", e.what()); }
        check(threw, "circular my_constants detected");
    }
    for (const char* badexpr : {"x +", "sin(x", "foo(x)", "x y", "max(x)", "q*2"}) {
        bool threw = false;
        try { Parser p(badexpr, {"x", "y", "z"}, C); } catch (const std::exception& e) { threw = true; std::printf("      (%s)\n", e.what()); }
        check(threw, std::string("error message for \"") + badexpr + "\"");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
