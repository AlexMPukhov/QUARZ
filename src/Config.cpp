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
