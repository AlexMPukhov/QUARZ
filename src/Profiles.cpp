#include "Profiles.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace quarz {

Table1D::Table1D(std::vector<std::pair<double, double>> pts) : pts_(std::move(pts)) {
    std::sort(pts_.begin(), pts_.end());
}

double Table1D::operator()(double x) const {
    if (pts_.empty()) return 1.0;
    if (x <= pts_.front().first) return pts_.front().second;
    if (x >= pts_.back().first) return pts_.back().second;
    auto it = std::upper_bound(pts_.begin(), pts_.end(), std::make_pair(x, -1e300),
                               [](const auto& a, const auto& b) { return a.first < b.first; });
    const auto& p1 = *it;
    const auto& p0 = *(it - 1);
    const double t = (x - p0.first) / (p1.first - p0.first);
    return (1 - t) * p0.second + t * p1.second;
}

namespace {
// length units of density files -> c/omega_p. len_m: c/omega_p in metres (0: unknown)
double length_to_code(double x, const std::string& unit, double len_m) {
    if (unit == "kp^-1" || unit == "1/kp" || unit == "c/omega_p" || unit == "c/wp") return x;
    double m = 0;
    if (unit == "m") m = 1;
    else if (unit == "cm") m = 1e-2;
    else if (unit == "mm") m = 1e-3;
    else if (unit == "um") m = 1e-6;
    else if (unit == "nm") m = 1e-9;
    else if (unit.empty()) throw std::invalid_argument("a length needs a unit (nm, um, mm, cm, m, kp^-1)");
    else throw std::invalid_argument("unknown length unit '" + unit + "' (nm, um, mm, cm, m, kp^-1)");
    if (!(len_m > 0))
        throw std::invalid_argument("lengths in '" + unit + "' need units.n0_cm3 (the density that defines c/omega_p)");
    return x * m / len_m;
}

// "exp(-(r / (50 um))^2)": numbers followed by a length unit are replaced by their value in c/omega_p
std::string convert_units(const std::string& s, double len_m) {
    static const char* units[] = {"kp^-1", "um", "mm", "cm", "nm", "m"};
    std::string out;
    size_t i = 0;
    auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    while (i < s.size()) {
        const bool start = (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.') && (i == 0 || !ident(s[i - 1]));
        if (!start) { out += s[i++]; continue; }
        char* end = nullptr;
        const double val = std::strtod(s.c_str() + i, &end);
        size_t j = size_t(end - s.c_str());
        if (j == i) { out += s[i++]; continue; }
        size_t k = j;
        while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k;
        bool done = false;
        for (const char* u : units) {
            const std::string name(u);
            if (s.compare(k, name.size(), name) != 0) continue;
            const size_t e = k + name.size();
            if (e < s.size() && (ident(s[e]) || s[e] == '(')) continue;
            char buf[64];
            std::snprintf(buf, sizeof buf, "(%.17g)", length_to_code(val, name, len_m));
            out += buf;
            i = e;
            done = true;
            break;
        }
        if (!done) { out += s.substr(i, j - i); i = j; }
    }
    return out;
}
} // namespace

DensityProfile::DensityProfile(const Config& cfg, const std::string& p) : prefix_(p) {
    parsed_ = read_function(cfg, p + ".density", 3, parser_, {"x", "y", "z"});
    if (cfg.has(p + ".density_file")) {
        for (const char* k : {".z_table", ".radial", ".radial_table", ".channel_depth", ".channel_radius"})
            if (cfg.has(p + k))
                throw std::runtime_error(p + k + ": the profile is given by " + p + ".density_file (remove the key; " + p +
                                         ".rmax and " + p + ".density(x,y,z) still apply)");
        const double n0 = cfg.get_double("units.n0_cm3", 0.0);
        if (n0 > 0) len_m_ = 5314.09 / std::sqrt(n0);   // c/omega_p [m]
        const double len_m = len_m_;
        densfile::DensityFileUnits u;
        u.length = [len_m](double x, const std::string& unit) { return length_to_code(x, unit, len_m); };
        u.density = [n0](double x, const std::string& unit) {
            if (unit == "n0") return x;
            double cm3 = 0;
            if (unit == "cm^-3" || unit == "cm-3" || unit == "1/cm^3") cm3 = 1;
            else if (unit == "m^-3" || unit == "m-3" || unit == "1/m^3") cm3 = 1e-6;
            else throw std::invalid_argument("unknown density unit '" + unit + "' (cm^-3, m^-3, n0; none: a factor)");
            if (!(n0 > 0)) throw std::invalid_argument("a density in '" + unit + "' needs units.n0_cm3 (the unit density n0)");
            return x * cm3 / n0;
        };
        const std::string path = cfg.get_string(p + ".density_file");
        auto D = std::make_shared<densfile::DensityFile>(
            densfile::readDensityFile(path, cfg.get_string(p + ".file_mesh", ""), cfg.get_int(p + ".file_iteration", -1), u));
        if (D->geometry == "xyz")
            throw std::runtime_error(path + ": geometry xyz: a 3D profile n(x, y, z) is for the 3D codes; QUARZ reads "
                                            "geometry z and rz (the file is not averaged over the azimuth: give n(r, z))");
        if (D->geometry == "xz")
            throw std::runtime_error(path + ": geometry xz: a Cartesian 2D profile n(x, z); QUARZ reads geometry z and rz");
        if (!D->transverse.empty()) {
            try {
                trans_ = Parser(convert_units(D->transverse, len_m_), {"r", "x", "y"});
            } catch (const std::exception& e) {
                throw std::runtime_error(path + ": transverse function '" + D->transverse + "': " + e.what());
            }
            has_trans_ = true;
        }
        file_ = D;
        file_z_ = cfg.get_double(p + ".file_z", 0.0);
        const std::string out = cfg.get_string(p + ".file_outside", "zero");
        if (out != "zero" && out != "hold") throw std::runtime_error(p + ".file_outside must be zero or hold");
        file_hold_ = out == "hold";
        rmax_ = cfg.get_double(p + ".rmax", 1e300);
        return;
    }
    for (const char* k : {".file_z", ".file_outside", ".file_mesh", ".file_iteration"})
        if (cfg.has(p + k)) throw std::runtime_error(p + k + " is only used with " + p + ".density_file");
    if (parsed_) return;   // built-in keys (radial, rmax, channel_*, tables) are then not read
    type_ = cfg.get_string(p + ".radial", "uniform");
    rmax_ = cfg.get_double(p + ".rmax", 1e300);
    if (type_ == "uniform") {
    } else if (type_ == "channel") {
        depth_ = cfg.get_double(p + ".channel_depth");
        rc_ = cfg.get_double(p + ".channel_radius");
    } else if (type_ == "table") {
        rtab_ = Table1D(cfg.get_pairs(p + ".radial_table"));
    } else {
        throw std::runtime_error("profile: unknown " + p + ".radial '" + type_ + "' (uniform|channel|table)");
    }
    if (cfg.has(p + ".z_table")) ztab_ = Table1D(cfg.get_pairs(p + ".z_table"));
}

double DensityProfile::eval(double x, double y, double z) const {
    double n = 1.0;
    if (parsed_) {
        n = parser_(x, y, z);
        if (!(n > 0)) return 0.0;
    }
    if (file_) {
        const double r = std::sqrt(x * x + y * y);
        if (r > rmax_) return 0.0;
        double f = file_->at(z - file_z_, r, 0.0, file_hold_);
        if (has_trans_) {
            const double t = trans_(r, x, y);
            f *= t > 0 ? t : 0.0;
        }
        n = parsed_ ? n * f : f;
    }
    return n;
}

double DensityProfile::eval_avg(double r, double z) const {
    if (file_ && !parsed_ && !has_trans_) return eval(r, 0.0, z);   // axisymmetric: no average needed
    constexpr int NQ = 16;
    double s = 0;
    for (int q = 0; q < NQ; ++q) s += eval(r * std::cos(2 * PI * q / NQ), r * std::sin(2 * PI * q / NQ), z);
    return s / NQ;
}

std::vector<std::string> DensityProfile::file_report(double z0, double z1, double dt) const {
    std::vector<std::string> out;
    if (!file_) return out;
    const auto& F = *file_;
    char buf[640];
    const bool two = !F.a.empty();
    const size_t nz = F.z.size(), nr = F.na();
    const std::string kind = F.absolute ? "absolute density [" + F.nUnit + "]" : std::string("density in units of n0");
    if (two)
        std::snprintf(buf, sizeof buf, "%s.density_file %s (%s; %s): n(r, z) on %zu x %zu points, %s", prefix_.c_str(),
                      F.path.c_str(), F.format.c_str(), F.detail.c_str(), nr, nz, kind.c_str());
    else
        std::snprintf(buf, sizeof buf, "%s.density_file %s (%s; %s): n(z) on %zu points, %s", prefix_.c_str(),
                      F.path.c_str(), F.format.c_str(), F.detail.c_str(), nz, kind.c_str());
    out.push_back(buf);
    const double za = F.z.front() + file_z_, zb = F.z.back() + file_z_;
    const bool open = !(z1 < 1e200);   // run end not known in advance (adaptive steps without t_end)
    char run[96];
    if (open) std::snprintf(run, sizeof run, "run: from %.6g", z0);
    else std::snprintf(run, sizeof run, "run: %.6g ... %.6g", z0, z1);
    if (two)
        std::snprintf(buf, sizeof buf, "  z = %.6g ... %.6g (%s), r = %.6g ... %.6g  [c/omega_p]", za, zb, run, F.a.front(), F.a.back());
    else
        std::snprintf(buf, sizeof buf, "  z = %.6g ... %.6g (%s)  [c/omega_p]", za, zb, run);
    out.push_back(buf);
    const std::string rp = two ? ", r = " + std::to_string(F.aPeak) : std::string();
    if (len_m_ > 0)
        std::snprintf(buf, sizeof buf, "  peak %.4g n0 at z = %.6g (%.6g mm)%s", F.peak, F.zPeak + file_z_,
                      (F.zPeak + file_z_) * len_m_ * 1e3, rp.c_str());
    else
        std::snprintf(buf, sizeof buf, "  peak %.4g n0 at z = %.6g%s", F.peak, F.zPeak + file_z_, rp.c_str());
    out.push_back(buf);
    if (has_trans_) out.push_back("  times the transverse function of the file, T(r, x, y) = " + F.transverse);
    if (file_hold_) out.push_back("  outside the file: the edge values are continued (file_outside = hold)");
    else out.push_back("  outside the file the density is zero");
    if (!file_hold_ && (zb <= z0 || (!open && za >= z1))) {
        std::snprintf(buf, sizeof buf, "%s.density_file: the file covers z = %.6g ... %.6g, outside the run (%s): no "
                      "density (shift it with %s.file_z, the z of the simulation at the file's z = 0)", prefix_.c_str(), za,
                      zb, run, prefix_.c_str());
        throw std::runtime_error(buf);
    }
    const double tol = open ? 1e-9 : 1e-9 * std::max(1.0, std::abs(z1));
    auto edge = [&](size_t i) {
        double m = 0;
        for (size_t j = 0; j < nr; ++j) m = std::max(m, F.v[i * nr + j]);
        return m / F.peak;
    };
    auto cut = [&](const char* what, double pos, double frac) {
        if (frac <= 0.01) return;
        std::snprintf(buf, sizeof buf, "Warning: %s.density_file %s at z = %.6g with %.3g %% of its peak; beyond it the "
                      "density is zero (a jump): extend the file, or set %s.file_outside = hold", prefix_.c_str(), what, pos,
                      100.0 * frac, prefix_.c_str());
        out.push_back(buf);
    };
    if (!file_hold_) {
        if (za > z0 + tol) cut("starts", za, edge(0));
        if (!open && zb < z1 - tol) cut("ends", zb, edge(nz - 1));
    }
    if (dt > 0) {   // the density of the box is that at its head, taken once per time step
        double q = 0, qpos = 0;
        for (size_t i = 0; i + 1 < nz; ++i) {
            const double a = F.z[i] + file_z_, b = F.z[i + 1] + file_z_;
            if (b <= z0 || (!open && a >= z1)) continue;
            for (size_t j = 0; j < nr; ++j) {
                const double s = std::abs(F.v[(i + 1) * nr + j] - F.v[i * nr + j]) / (b - a) * dt / F.peak;
                if (s > q) { q = s; qpos = 0.5 * (a + b); }
            }
        }
        if (q > 0.25) {
            std::snprintf(buf, sizeof buf, "Warning: %s.density_file changes by %.3g %% of its peak per time step (dt = %.4g) "
                          "at z = %.6g: scale length n_peak / |dn/dz| = %.4g c/omega_p, less than 4 steps; the steps do not "
                          "resolve this gradient", prefix_.c_str(), 100.0 * q, dt, qpos, dt / q);
            out.push_back(buf);
        }
    }
    return out;
}

double DensityProfile::radial(double r) const {
    if (r > rmax_) return 0.0;
    if (type_ == "channel") return 1.0 + depth_ * (r / rc_) * (r / rc_);
    if (type_ == "table") return std::max(0.0, rtab_(r));
    return 1.0;
}

double DensityProfile::longitudinal(double z) const {
    return ztab_.empty() ? 1.0 : std::max(0.0, ztab_(z));
}

} // namespace quarz
