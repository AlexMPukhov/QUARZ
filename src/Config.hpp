// Input-file reader: "key = value" statements, flat or in blocks (README section "Input file").
//   - '#' starts a comment; statements end at the end of the line or at ';' (outside quotes)
//   - flat form:   xi.step = 0.01          block form:   xi { step = 0.01; min = -12 }
//     a block only puts "<name>." in front of its keys; blocks may be nested
//   - named blocks: "bunch <name> {" (or "beam <name> {"), "species <name> {", "diag <name> {" put the keys under
//     "<name>." ("diag.<name>." for diag) and add the name to the list beams / plasma.species / diag.names
//   - a key given twice in the file is an error; keys that were never read are an error before the first step
//     (main.cpp), so a misspelt key cannot change a run silently
//   - a number may carry a unit (um, mm, fs, cm^-3, mrad, %, ...): it is converted to the normalized units
//     (lengths 1/k_p, times 1/omega_p, densities n0) with the reference density units.n0 (or units.n0_cm3)
//   - values are whitespace separated lists; get_list() returns all tokens
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace quarz {

class Config {
public:
    Config() = default;
    explicit Config(const std::string& filename);

    void set(const std::string& key, const std::string& value) { kv_[key] = value; }
    bool has(const std::string& key) const;
    // a number with an optional unit ("3", "3 um", "3um", "1e17cm^-3") in normalized units; what: for messages
    double number(const std::string& text, const std::string& what) const;
    // reference density in cm^-3 from units.n0_cm3 or units.n0 (0 if neither is given)
    double n0_cm3() const;

    double      get_double(const std::string& key) const;                 // required
    double      get_double(const std::string& key, double def) const;     // optional
    int         get_int(const std::string& key) const;
    int         get_int(const std::string& key, int def) const;
    bool        get_bool(const std::string& key, bool def) const;
    std::string get_string(const std::string& key) const;
    std::string get_string(const std::string& key, const std::string& def) const;
    std::vector<std::string> get_list(const std::string& key) const;       // empty if missing
    // list of "a:b" pairs -> vector of (a,b)
    std::vector<std::pair<double, double>> get_pairs(const std::string& key) const;

    std::vector<std::string> unused_keys() const;
    std::vector<std::string> keys() const;
    void print(std::ostream& os) const;

private:
    const std::string& raw(const std::string& key) const;
    std::map<std::string, std::string> kv_;
    mutable std::set<std::string> used_;
};

} // namespace quarz
