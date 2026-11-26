#include "Config.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace quarz {

static std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// the statements of a line: split at ';' outside quotes
static std::vector<std::string> statements(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    char quote = 0;
    for (char c : line) {
        if (quote) { if (c == quote) quote = 0; cur += c; }
        else if (c == '"' || c == '\'') { quote = c; cur += c; }
        else if (c == ';') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

// '}' minus '{' outside quotes
static int closing_excess(const std::string& s) {
    int n = 0;
    char quote = 0;
    for (char c : s) {
        if (quote) { if (c == quote) quote = 0; }
        else if (c == '"' || c == '\'') quote = c;
        else if (c == '}') ++n;
        else if (c == '{') --n;
    }
    return n;
}

Config::Config(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Config: cannot open input file '" + filename + "'");
    std::string line;
    int lineno = 0;
    std::vector<std::string> prefix;                       // open blocks: "xi.", "driver.", ...
    std::vector<int> opened;                               // their lines
    std::map<std::string, int> where;                      // key -> line
    std::vector<std::pair<std::string, std::string>> named;   // (list key, name) of the named blocks, in order
    const auto err = [&](const std::string& m) {
        throw std::runtime_error("Config: " + filename + ":" + std::to_string(lineno) + ": " + m);
    };
    const auto close = [&]() {
        if (prefix.empty()) err("'}' without an open block");
        prefix.pop_back();
        opened.pop_back();
    };
    while (std::getline(in, line)) {
        ++lineno;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        for (std::string s : statements(line)) {
            s = trim(s);
            while (!s.empty()) {
                if (s[0] == '}') { close(); s = trim(s.substr(1)); continue; }
                const auto eq = s.find('=');
                const auto br = s.find('{');
                if (br != std::string::npos && (eq == std::string::npos || br < eq)) {   // block header
                    std::istringstream hs(s.substr(0, br));
                    std::vector<std::string> w;
                    for (std::string t; hs >> t;) w.push_back(t);
                    const std::string outer = prefix.empty() ? std::string() : prefix.back();
                    if (w.size() == 1) prefix.push_back(outer + w[0] + ".");
                    else if (w.size() == 2 && outer.empty() && (w[0] == "bunch" || w[0] == "beam")) {
                        named.emplace_back("beams", w[1]);
                        prefix.push_back(w[1] + ".");
                    } else if (w.size() == 2 && outer.empty() && w[0] == "species") {
                        named.emplace_back("plasma.species", w[1]);
                        prefix.push_back(w[1] + ".");
                    } else if (w.size() == 2 && outer.empty() && w[0] == "diag") {
                        named.emplace_back("diag.names", w[1]);
                        prefix.push_back("diag." + w[1] + ".");
                    } else
                        err("block header '" + trim(s.substr(0, br)) + " {': expected '<name> {' or, at the top level, "
                            "'bunch <name> {', 'species <name> {', 'diag <name> {'");
                    opened.push_back(lineno);
                    s = trim(s.substr(br + 1));
                    continue;
                }
                if (eq == std::string::npos) err("missing '=' in '" + s + "'");
                std::string key = trim(s.substr(0, eq));
                std::string val = trim(s.substr(eq + 1));
                if (key.empty()) err("empty key");
                int nclose = 0;                            // "name { key = value }" on one line
                while (!prefix.empty() && !val.empty() && val.back() == '}' && closing_excess(val) > 0) {
                    val = trim(val.substr(0, val.size() - 1));
                    ++nclose;
                }
                if (!prefix.empty()) key = prefix.back() + key;
                if (where.count(key))
                    err("key '" + key + "' is given twice (first in line " + std::to_string(where[key]) + ")");
                where[key] = lineno;
                kv_[key] = val;
                for (int i = 0; i < nclose; ++i) close();
                s.clear();
            }
        }
    }
    if (!prefix.empty())
        throw std::runtime_error("Config: " + filename + ": the block opened in line " + std::to_string(opened.back()) +
                                 " is not closed");
    for (const auto& nl : named) {                         // a named block joins its list (beams, plasma.species, diag.names)
        std::istringstream ls(kv_.count(nl.first) ? kv_[nl.first] : std::string());
        bool have = false;
        for (std::string t; ls >> t;) have = have || t == nl.second;
        if (!have) kv_[nl.first] += (kv_[nl.first].empty() ? "" : " ") + nl.second;
    }
}

bool Config::has(const std::string& key) const {
    if (kv_.count(key)) return true;
    return key == "units.n0_cm3" && kv_.count("units.n0");   // the reference density with its unit
}

double Config::n0_cm3() const {
    const bool plain = kv_.count("units.n0_cm3") > 0;
    if (!plain && !kv_.count("units.n0")) return 0.0;
    const std::string& v = raw(plain ? "units.n0_cm3" : "units.n0");
    size_t pos = 0;
    double x = 0;
    try { x = std::stod(v, &pos); } catch (...) { pos = 0; }
    const std::string u = pos ? trim(v.substr(pos)) : std::string("?");
    if (plain) {
        if (!u.empty()) throw std::runtime_error("Config: key 'units.n0_cm3' is a plain number of cm^-3: " + v);
        return x;
    }
    if (u == "cm^-3" || u == "cm-3" || u == "1/cm3" || u == "1/cm^3") return x;
    if (u == "m^-3" || u == "m-3" || u == "1/m3" || u == "1/m^3") return x * 1e-6;
    throw std::runtime_error("Config: key 'units.n0' needs a density with its unit (cm^-3 or m^-3): " + v);
}

double Config::number(const std::string& text, const std::string& what) const {
    const std::string v = trim(text);
    size_t pos = 0;
    double x = 0;
    try { x = std::stod(v, &pos); } catch (...) { pos = 0; }
    if (pos == 0) throw std::runtime_error("Config: " + what + " is not a number: " + text);
    const std::string u = trim(v.substr(pos));
    if (u.empty()) return x;                               // normalized units
    struct U { const char* name; char kind; double si; };  // L length [m], T time [s], N density [cm^-3], 1 pure number
    static const U table[] = {
        {"nm", 'L', 1e-9}, {"um", 'L', 1e-6}, {"\xC2\xB5m", 'L', 1e-6}, {"micron", 'L', 1e-6}, {"mm", 'L', 1e-3},
        {"cm", 'L', 1e-2}, {"m", 'L', 1.0}, {"mm*mrad", 'L', 1e-6}, {"mm.mrad", 'L', 1e-6},
        {"as", 'T', 1e-18}, {"fs", 'T', 1e-15}, {"ps", 'T', 1e-12}, {"ns", 'T', 1e-9}, {"s", 'T', 1.0},
        {"cm^-3", 'N', 1.0}, {"cm-3", 'N', 1.0}, {"1/cm3", 'N', 1.0}, {"1/cm^3", 'N', 1.0},
        {"m^-3", 'N', 1e-6}, {"m-3", 'N', 1e-6}, {"1/m3", 'N', 1e-6}, {"1/m^3", 'N', 1e-6},
        {"rad", '1', 1.0}, {"mrad", '1', 1e-3}, {"urad", '1', 1e-6}, {"deg", '1', 0.017453292519943295},
        {"%", '1', 0.01},
        {"1/kp", '1', 1.0}, {"kp^-1", '1', 1.0}, {"c/wp", '1', 1.0}, {"1/wp", '1', 1.0}, {"wp^-1", '1', 1.0}, {"n0", '1', 1.0},
    };
    for (const U& t : table) {
        if (u != t.name) continue;
        if (t.kind == '1') return x * t.si;
        const double n0 = n0_cm3();
        if (!(n0 > 0))
            throw std::runtime_error("Config: " + what + " = " + text + ": a value with a unit needs the reference "
                                     "density units.n0 (e.g. units.n0 = 1e17 cm^-3)");
        if (t.kind == 'N') return x * t.si / n0;
        const double wp = 5.64146e4 * std::sqrt(n0);       // rad/s, as in Ionization.cpp
        if (t.kind == 'T') return x * t.si * wp;
        return x * t.si * wp / 2.99792458e8;               // length: k_p x
    }
    throw std::runtime_error("Config: " + what + " = " + text + ": unknown unit '" + u + "' (lengths nm um mm cm m, "
                             "times as fs ps ns s, densities cm^-3 m^-3, angles rad mrad urad deg, %; no unit: 1/k_p, "
                             "1/omega_p, n0)");
}

const std::string& Config::raw(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("Config: required key '" + key + "' is missing");
    used_.insert(key);
    return it->second;
}

double Config::get_double(const std::string& key) const {
    if (key == "units.n0_cm3" && !kv_.count(key) && kv_.count("units.n0")) return n0_cm3();
    return number(raw(key), "key '" + key + "'");
}
double Config::get_double(const std::string& key, double def) const {
    return has(key) ? get_double(key) : def;
}
int Config::get_int(const std::string& key) const {
    const std::string& v = raw(key);
    try { return static_cast<int>(std::stol(v)); }
    catch (...) { throw std::runtime_error("Config: key '" + key + "' is not an integer: " + v); }
}
int Config::get_int(const std::string& key, int def) const {
    return has(key) ? get_int(key) : def;
}
bool Config::get_bool(const std::string& key, bool def) const {
    if (!has(key)) return def;
    std::string v = raw(key);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("Config: key '" + key + "' is not a boolean: " + v);
}
std::string Config::get_string(const std::string& key) const { return raw(key); }
std::string Config::get_string(const std::string& key, const std::string& def) const {
    return has(key) ? raw(key) : def;
}
std::vector<std::string> Config::get_list(const std::string& key) const {
    std::vector<std::string> out;
    if (!has(key)) return out;
    std::istringstream ss(raw(key));
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}
std::vector<std::pair<double, double>> Config::get_pairs(const std::string& key) const {
    std::vector<std::pair<double, double>> out;
    for (const auto& tok : get_list(key)) {
        auto c = tok.find(':');
        if (c == std::string::npos)
            throw std::runtime_error("Config: key '" + key + "': expected a:b pairs, got '" + tok + "'");
        out.emplace_back(number(tok.substr(0, c), "key '" + key + "'"), number(tok.substr(c + 1), "key '" + key + "'"));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------
// Common key names of the VLPL family (README section 5, "Common names"). Each is rewritten into the key of
// QUARZ that the code reads; giving both spellings of one quantity is an error.
namespace {
// "<number> <unit>" -> value in the unit named by the table (energy: MeV, charge: C); false if the unit is none of them
bool with_unit(const std::string& text, const char* const* names, const double* factors, int n, double& value) {
    const std::string v = trim(text);
    size_t pos = 0;
    double x = 0;
    try { x = std::stod(v, &pos); } catch (...) { return false; }
    const std::string u = trim(v.substr(pos));
    for (int i = 0; i < n; ++i)
        if (u == names[i]) { value = x * factors[i]; return true; }
    return false;
}
const char* const charge_units[] = {"fC", "pC", "nC", "uC", "C"};
const double charge_factors[] = {1e-15, 1e-12, 1e-9, 1e-6, 1.0};
const char* const energy_units[] = {"eV", "keV", "MeV", "GeV", "TeV"};
const double energy_factors[] = {1e-6, 1e-3, 1.0, 1e3, 1e6};
}  // namespace

void Config::apply_standard_names() {
    // new -> old, the same value
    const auto rename = [&](const std::string& neu, const std::string& old) {
        if (!kv_.count(neu)) return;
        if (kv_.count(old))
            throw std::runtime_error("Config: '" + neu + "' and '" + old + "' are two names of one quantity: give one");
        kv_[old] = kv_[neu];
        kv_.erase(neu);
    };
    // the common value of new1, new2 (e.g. sigma_x, sigma_y of a round beam) -> old
    const auto merge = [&](const std::string& n1, const std::string& n2, const std::string& old) {
        if (!kv_.count(n1) && !kv_.count(n2)) return;
        if (kv_.count(old))
            throw std::runtime_error("Config: '" + (kv_.count(n1) ? n1 : n2) + "' and '" + old +
                                     "' are two names of one quantity: give one");
        const std::string& first = kv_.count(n1) ? n1 : n2;
        const double a = number(kv_[first], "key '" + first + "'");
        if (kv_.count(n1) && kv_.count(n2)) {
            const double b = number(kv_[n2], "key '" + n2 + "'");
            if (std::abs(a - b) > 1e-12 * std::max(std::abs(a), std::abs(b)))
                throw std::runtime_error("Config: '" + n1 + "' and '" + n2 + "' differ: the beams of QUARZ are round (" +
                                         old + ")");
        }
        std::ostringstream os;
        os.precision(17);
        os << a;
        kv_[old] = os.str();
        kv_.erase(n1);
        kv_.erase(n2);
    };
    const auto put = [&](const std::string& key, double x) {
        std::ostringstream os;
        os.precision(17);
        os << x;
        kv_[key] = os.str();
    };

    // ---- grid
    rename("grid.m_max", "modes");
    rename("grid.r_max", "grid.rmax");

    // ---- laser
    rename("laser.waist", "laser.w0");
    rename("laser.z_focus", "laser.focus");
    if (kv_.count("laser.tau_fwhm")) {                     // FWHM of the intensity; a ~ exp(-xi^2 / L0^2): FWHM = sqrt(2 ln 2) L0
        if (kv_.count("laser.L0"))
            throw std::runtime_error("Config: 'laser.tau_fwhm' and 'laser.L0' are two names of one quantity: give one");
        put("laser.L0", number(kv_["laser.tau_fwhm"], "key 'laser.tau_fwhm'") / std::sqrt(2.0 * std::log(2.0)));
        kv_.erase("laser.tau_fwhm");
    }
    if (kv_.count("laser.wavelength")) {                   // k0 = omega0 / omega_p = 2 pi / (k_p lambda)
        if (kv_.count("laser.k0") || kv_.count("laser.lambda0_um"))
            throw std::runtime_error("Config: 'laser.wavelength' and 'laser.k0' / 'laser.lambda0_um' are names of one "
                                     "quantity: give one");
        const double lam = number(kv_["laser.wavelength"], "key 'laser.wavelength'");
        if (!(lam > 0)) throw std::runtime_error("Config: laser.wavelength must be positive");
        put("laser.k0", 2.0 * 3.14159265358979323846 / lam);
        kv_.erase("laser.wavelength");
    }

    // ---- bunches
    std::vector<std::string> beams;
    {
        std::istringstream ls(kv_.count("beams") ? kv_["beams"] : std::string());
        for (std::string t; ls >> t;) beams.push_back(t);
    }
    for (const std::string& b : beams) {
        rename(b + ".sigma_z", b + ".sigma_xi");
        merge(b + ".sigma_x", b + ".sigma_y", b + ".sigma_r");
        merge(b + ".emittance_x", b + ".emittance_y", b + ".emit_n");
        rename(b + ".energy_spread", b + ".espread");
        rename(b + ".macroparticles", b + ".nparticles");
        rename(b + ".file_species", b + ".openpmd_species");
        rename(b + ".file_iteration", b + ".openpmd_iteration");

        // total charge: "<beam>.charge" with a unit of charge (a plain number is the charge number of the particles)
        double qtot = 0;
        bool total = false;
        if (kv_.count(b + ".charge") && with_unit(kv_[b + ".charge"], charge_units, charge_factors, 5, qtot)) {
            total = true;
            kv_.erase(b + ".charge");
        }
        // particle species
        if (kv_.count(b + ".species")) {
            const std::string s = trim(kv_[b + ".species"]);
            if (kv_.count(b + ".charge") || kv_.count(b + ".mass"))
                throw std::runtime_error("Config: '" + b + ".species' and '" + b + ".charge' / '.mass' (numbers): give one");
            if (s == "electron") { put(b + ".charge", -1); put(b + ".mass", 1); }
            else if (s == "positron") { put(b + ".charge", 1); put(b + ".mass", 1); }
            else if (s == "proton") { put(b + ".charge", 1); put(b + ".mass", 1836.15267343); }
            else throw std::runtime_error("Config: " + b + ".species = " + s + ": electron, positron or proton");
            kv_.erase(b + ".species");
        }
        const double qnum = kv_.count(b + ".charge") ? number(kv_[b + ".charge"], "key '" + b + ".charge'") : -1.0;
        const double mass = kv_.count(b + ".mass") ? number(kv_[b + ".mass"], "key '" + b + ".mass'") : 1.0;
        // kinetic energy -> gamma
        if (kv_.count(b + ".energy")) {
            if (kv_.count(b + ".gamma"))
                throw std::runtime_error("Config: '" + b + ".energy' and '" + b + ".gamma': give one");
            double mev = 0;
            if (!with_unit(kv_[b + ".energy"], energy_units, energy_factors, 5, mev))
                throw std::runtime_error("Config: " + b + ".energy = " + kv_[b + ".energy"] +
                                         ": the kinetic energy with its unit (eV keV MeV GeV TeV)");
            put(b + ".gamma", 1.0 + mev / (0.51099895 * mass));
            kv_.erase(b + ".energy");
        }
        // position counted from the rear end of the window (as in GEM-PIC-RZ): xi0 = xi.min + zeta_center
        if (kv_.count(b + ".zeta_center")) {
            if (kv_.count(b + ".xi0"))
                throw std::runtime_error("Config: '" + b + ".zeta_center' and '" + b + ".xi0': give one");
            if (!kv_.count("xi.min")) throw std::runtime_error("Config: " + b + ".zeta_center needs xi.min");
            put(b + ".xi0", number(kv_["xi.min"], "key 'xi.min'") + number(kv_[b + ".zeta_center"], "key '" + b + ".zeta_center'"));
            kv_.erase(b + ".zeta_center");
        }
        // total charge -> peak density of the Gaussian or flat-top bunch (the weight sum of Beam::init_gaussian)
        if (total) {
            if (kv_.count(b + ".n0"))
                throw std::runtime_error("Config: '" + b + ".charge' with a unit (total charge) and '" + b + ".n0': give one");
            const std::string prof = kv_.count(b + ".profile") ? trim(kv_[b + ".profile"]) : std::string("gaussian");
            if ((prof != "gaussian" && prof != "flattop") || kv_.count(b + ".xi_cut_head"))
                throw std::runtime_error("Config: " + b + ".charge as the total charge: only for profile = gaussian | "
                                         "flattop without xi_cut_head (files carry their charge)");
            const double n0 = n0_cm3();
            if (!(n0 > 0)) throw std::runtime_error("Config: " + b + ".charge as the total charge needs units.n0");
            if (!kv_.count(b + ".sigma_r")) throw std::runtime_error("Config: " + b + ".charge needs " + b + ".sigma_r");
            const double sr = number(kv_[b + ".sigma_r"], "key '" + b + ".sigma_r'");
            const double cut = kv_.count(b + ".cut") ? number(kv_[b + ".cut"], "key '" + b + ".cut'") : 3.0;
            const double pi = 3.14159265358979323846;
            const double Pr = 1.0 - std::exp(-0.5 * cut * cut);
            double vol;                                     // weight sum per unit peak density, in k_p^-3
            if (prof == "gaussian") {
                if (!kv_.count(b + ".sigma_xi")) throw std::runtime_error("Config: " + b + ".charge needs " + b + ".sigma_z");
                const double sxi = number(kv_[b + ".sigma_xi"], "key '" + b + ".sigma_xi'");
                vol = std::pow(2 * pi, 1.5) * sr * sr * sxi * Pr * std::erf(cut / std::sqrt(2.0));
            } else {
                if (!kv_.count(b + ".length")) throw std::runtime_error("Config: " + b + ".charge needs " + b + ".length");
                vol = 2 * pi * sr * sr * Pr * number(kv_[b + ".length"], "key '" + b + ".length'");
            }
            const double kp_cm = 5.64146e4 * std::sqrt(n0) / 2.99792458e10;          // 1/cm
            const double npart = qtot / (1.602176634e-19 * std::abs(qnum));           // real particles
            put(b + ".n0", npart * kp_cm * kp_cm * kp_cm / n0 / vol);
        }
    }
}

std::vector<std::string> Config::keys() const {
    std::vector<std::string> out;
    for (const auto& kv : kv_) out.push_back(kv.first);
    return out;
}

std::vector<std::string> Config::unused_keys() const {
    std::vector<std::string> out;
    for (const auto& kv : kv_)
        if (!used_.count(kv.first)) out.push_back(kv.first);
    return out;
}

void Config::print(std::ostream& os) const {
    for (const auto& kv : kv_) os << "  " << kv.first << " = " << kv.second << "\n";
}

} // namespace quarz
