#include "Ionization.hpp"
#include "Config.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>

namespace quarz {

namespace {
struct Element {
    double amu;
    std::vector<double> ip;   // ionization energies, eV (NIST / CRC Handbook)
    double M2, C;             // Bethe parameters for fast-particle impact ionization (0 = not tabulated)
    double J;                 // width of the delta-electron energy spectrum, eV (0 = not tabulated)
    std::vector<std::pair<double, int>> sh = {};   // inner subshells (binding energy eV, electrons) whose
                                                   // free-electron tail the fit of J does not contain
};
// Bethe parameters: Rieke & Prepejchal, PRA 6, 1507 (1972), Table II, quality-A entries (He: the
// H2-additive run, the pair behind their minimum cross-section 1.86e-19 cm^2; Ar, Xe: pure gas).  They
// reproduce the measured minimum primary ionization (Blum, Riegler & Rolandi, Particle Detection with
// Drift Chambers, Table 1.2) to < 0.3 %: ~25 (Ar), ~43 (Xe) primary ion pairs/cm at NTP.  No data for
// H atoms and the alkali vapours (Li, Rb, Cs) and the atomic C, N, O: give <species>.impact.M2 and
// <species>.impact.C, or <species>.impact.sigma_cm2.
// J: Opal, Peterson & Beaty, J. Chem. Phys. 55, 4100 (1971), fits to the measured secondary-electron
// spectra (the authors call the fits for Ar, Kr, Xe poor); for others give <species>.impact.J_eV.
// Inner subshells: X-ray Data Booklet (LBNL), Table 1-1; Kr n = 1, 2 and Xe n = 1-3.  The 1/W^2 tail of
// the fits corresponds to ~21 free electrons in Ar (Z = 18), ~28 in Kr (Z = 36), ~34 in Xe (Z = 54).
const std::map<std::string, Element>& elements() {
    static const std::map<std::string, Element> t = {
        {"H", {1.008, {13.59844}, 0, 0, 0}},
        {"He", {4.0026, {24.58739, 54.41776}, 0.745, 8.005, 15.8}},
        {"Li", {6.94, {5.39172, 75.6402, 122.4543}, 0, 0, 0}},
        {"C", {12.011, {11.2603, 24.3833, 47.8878, 64.4939, 392.087, 489.993}, 0, 0, 0}},
        {"N", {14.007, {14.5341, 29.6013, 47.44924, 77.4735, 97.8902, 552.0718, 667.046}, 0, 0, 0}},
        {"O", {15.999, {13.6181, 35.1211, 54.9355, 77.41353, 113.899, 138.1189, 739.29, 871.4101}, 0, 0, 0}},
        {"Ne", {20.180, {21.5646, 40.96328, 63.45, 97.12, 126.21, 157.93, 207.2759, 239.0989, 1195.8286, 1362.1995}, 2.02, 18.17, 24.2}},
        {"Ar", {39.948, {15.7596, 27.62966, 40.74, 59.81, 75.02, 91.009, 124.323, 143.460, 422.45, 478.69, 538.96,
                         618.26, 686.10, 755.74, 854.77, 918.03, 4120.8857, 4426.2296}, 4.22, 37.93, 10.0}},
        {"Kr", {83.798, {13.99961, 24.35985, 36.950, 52.5, 64.7, 78.5, 111.0, 125.802}, 6.09, 52.38, 9.6,
                {{14326, 2}, {1921, 2}, {1730.9, 2}, {1678.4, 4}}}},
        {"Rb", {85.468, {4.17713, 27.2895, 40.0}, 0, 0, 0}},
        {"Xe", {131.293, {12.1298, 20.975, 32.1230}, 8.04, 72.35, 8.7,
                {{34561, 2}, {5453, 2}, {5107, 2}, {4786, 4}, {1148.7, 2}, {1002.1, 2}, {940.6, 4}, {689.0, 4}, {676.4, 6}}}},
        {"Cs", {132.905, {3.8939, 23.15744}, 0, 0, 0}},
    };
    return t;
}
constexpr double HARTREE_EV = 27.211386245988;
constexpr double E_AU = 5.14220674763e11;          // V/m
constexpr double T_AU = 2.4188843265857e-17;       // s
constexpr double SIGMA_BETHE = 1.87386e-20;        // 4 pi a0^2 R / (m c^2 / 2), cm^2
constexpr double AMU_ME = 1822.888486;
constexpr double ME_C2_EV = 510998.95;
constexpr double R_E_CM = 2.8179403262e-13;
} // namespace

bool is_ionizable(const Config& cfg, const std::string& name) {
    return cfg.has(name + ".element") || cfg.has(name + ".ionization_energies_eV");
}

bool receives_delta_electrons(const Config& cfg, const std::string& name) {
    if (!cfg.has("plasma.species")) return false;
    for (const auto& s : cfg.get_list("plasma.species"))
        if (is_ionizable(cfg, s) && cfg.get_bool(s + ".impact.secondary_energy", false) &&
            cfg.get_string(s + ".ionization.product", "electrons") == name)
            return true;
    return false;
}

IonSpeciesInfo make_ion_species(const Config& cfg, const std::string& name, double n0_cm3) {
    IonSpeciesInfo I;
    if (!(n0_cm3 > 0)) throw std::runtime_error(name + ": ionization needs the plasma density units.n0_cm3");
    std::vector<double> ip;
    double M2 = 0, C = 0, J = 0;
    std::vector<std::pair<double, int>> sh;
    if (cfg.has(name + ".ionization_energies_eV")) {
        for (const auto& s : cfg.get_list(name + ".ionization_energies_eV")) ip.push_back(std::stod(s));
        if (cfg.has(name + ".element")) {
            I.element = cfg.get_string(name + ".element");
            auto it = elements().find(I.element);
            if (it != elements().end()) { I.mass_me = it->second.amu * AMU_ME; M2 = it->second.M2; C = it->second.C; J = it->second.J; sh = it->second.sh; }
        }
    } else {
        I.element = cfg.get_string(name + ".element");
        auto it = elements().find(I.element);
        if (it == elements().end()) {
            std::string known;
            for (const auto& e : elements()) known += " " + e.first;
            throw std::runtime_error(name + ".element = " + I.element + " is not tabulated (known:" + known +
                                     "); give " + name + ".ionization_energies_eV");
        }
        ip = it->second.ip;
        I.mass_me = it->second.amu * AMU_ME;
        M2 = it->second.M2;
        C = it->second.C;
        J = it->second.J;
        sh = it->second.sh;
    }
    if (ip.empty()) throw std::runtime_error(name + ": no ionization energies");
    IonParams& P = I.P;
    P.z0 = cfg.get_int(name + ".z0", 0);
    P.zmax = cfg.get_int(name + ".z_max", static_cast<int>(ip.size()));
    if (P.z0 < 0 || P.zmax > static_cast<int>(ip.size()) || P.zmax < P.z0 || P.zmax > ION_MAX_LEVELS)
        throw std::runtime_error(name + ": need 0 <= z0 <= z_max <= number of tabulated ionization energies (" +
                                 std::to_string(ip.size()) + ")");
    // units: omega_p, E0 = m c omega_p / e
    const double wp = 5.64146e4 * std::sqrt(n0_cm3);            // rad/s
    const double E0 = 9.1093837015e-31 * 2.99792458e8 * wp / 1.602176634e-19;
    const double kp_inv_cm = 2.99792458e10 / wp;
    P.econv = E0 / E_AU;
    for (int l = P.z0; l < P.zmax; ++l) {
        const double Ip = ip[l] / HARTREE_EV;                     // Hartree
        const double Z = l + 1;
        const double ns = Z / std::sqrt(2 * Ip);
        const double C2 = std::pow(2.0, 2 * ns) / (ns * std::tgamma(2 * ns));
        const double e32 = std::pow(2 * Ip, 1.5);
        P.pw[l] = 2 * ns - 1;
        P.kap[l] = 2.0 * e32 / 3.0;
        P.lnA[l] = std::log(Ip * C2) + P.pw[l] * std::log(2 * e32) - std::log(T_AU * wp);
        I.ip_eV.push_back(ip[l]);
    }
    P.field = cfg.get_bool(name + ".ionization.field", true);
    // impact ionization by the beams (first ionization only)
    M2 = cfg.get_double(name + ".impact.M2", M2);
    C = cfg.get_double(name + ".impact.C", C);
    const double sig = cfg.get_double(name + ".impact.sigma_cm2", 0.0);
    const bool have = sig > 0 || M2 > 0 || C > 0;
    P.impact = cfg.get_bool(name + ".ionization.impact", have && P.z0 == 0);
    if (P.impact && !have)
        throw std::runtime_error(name + ".ionization.impact: no cross-section; give " + name + ".impact.M2 and " +
                                 name + ".impact.C (Bethe), or " + name + ".impact.sigma_cm2");
    if (P.impact && P.z0 != 0) throw std::runtime_error(name + ": impact ionization is modelled for neutral atoms (z0 = 0)");
    if (P.impact) {
        if (sig > 0) { P.s0 = sig * n0_cm3 * kp_inv_cm; I.sigma_cm2 = sig; }
        else {
            P.s1 = SIGMA_BETHE * M2 * n0_cm3 * kp_inv_cm;
            P.s2 = SIGMA_BETHE * C * n0_cm3 * kp_inv_cm;
            I.M2 = M2; I.C = C;
        }
    }
    // delta electrons
    P.delta = cfg.get_bool(name + ".impact.secondary_energy", false);
    if (P.delta) {
        if (!P.impact) throw std::runtime_error(name + ".impact.secondary_energy needs beam impact ionization");
        I.J_eV = cfg.get_double(name + ".impact.J_eV", J);
        if (!(I.J_eV > 0))
            throw std::runtime_error(name + ".impact.secondary_energy: no width of the energy spectrum tabulated; give " + name +
                                     ".impact.J_eV");
        P.dJ = I.J_eV / ME_C2_EV;
        const std::string a = cfg.get_string(name + ".impact.secondary_angle", "kinematic");
        if (a != "kinematic" && a != "isotropic")
            throw std::runtime_error(name + ".impact.secondary_angle must be kinematic or isotropic");
        P.delta_iso = a == "isotropic";
        // inner subshells: built in (Kr, Xe) or <name>.impact.inner_shells_eV / inner_shells_n
        if (cfg.has(name + ".impact.inner_shells_eV")) {
            const auto b = cfg.get_list(name + ".impact.inner_shells_eV");
            const auto n = cfg.get_list(name + ".impact.inner_shells_n");
            if (b.size() != n.size())
                throw std::runtime_error(name + ".impact.inner_shells_eV and inner_shells_n must have the same length");
            sh.clear();
            for (size_t i = 0; i < b.size(); ++i) sh.push_back({std::stod(b[i]), std::stoi(n[i])});
        }
        if (!cfg.get_bool(name + ".impact.inner_shells", true)) sh.clear();
        if (sh.size() > static_cast<size_t>(ION_MAX_SHELLS))
            throw std::runtime_error(name + ": at most " + std::to_string(ION_MAX_SHELLS) + " inner subshells");
        I.shells = sh;
        P.nsh = static_cast<int>(sh.size());
        double tot = 0;
        for (const auto& s : sh) {
            if (!(s.first > 0) || s.second < 1) throw std::runtime_error(name + ": inner subshells need B > 0 and n >= 1");
            tot += s.second / s.first;
        }
        double c = 0;
        for (int i = 0; i < P.nsh; ++i) {
            c += sh[i].second / sh[i].first;
            P.shB[i] = sh[i].first / ME_C2_EV;
            P.shP[i] = c / tot;
        }
        P.s_in = 2 * PI * R_E_CM * R_E_CM * ME_C2_EV * tot * n0_cm3 * kp_inv_cm;
    }
    I.born_dump = cfg.get_bool(name + ".ionization.born_dump", false);
    I.nsplit = cfg.get_int(name + ".ionization.nsplit", 1);
    if (I.nsplit < 1) throw std::runtime_error(name + ".ionization.nsplit must be >= 1");
    I.product = cfg.get_string(name + ".ionization.product", "electrons");
    return I;
}

double impact_sigma_cm2(const IonSpeciesInfo& I, double z, double gamma) {
    if (I.sigma_cm2 > 0) return I.sigma_cm2 * z * z;
    const double b2 = 1 - 1 / (gamma * gamma);
    return SIGMA_BETHE * z * z / b2 * (I.M2 * (std::log(b2 * gamma * gamma) - b2) + I.C);
}

std::string IonSpeciesInfo::description() const {
    std::ostringstream o;
    o << "  ionization: " << (element.empty() ? std::string("custom") : element) << ", charge states " << P.z0 << " .. "
      << P.zmax << " (Ip =";
    for (double e : ip_eV) o << " " << e;
    o << " eV); field (ADK): " << (P.field ? "on" : "off") << "; beam impact: ";
    if (!P.impact) o << "off";
    else if (sigma_cm2 > 0) o << "sigma = " << sigma_cm2 << " cm^2";
    else {
        char b[128];
        std::snprintf(b, sizeof(b), "Bethe M^2 = %g, C = %g (sigma = %.3g cm^2 at gamma = 427)", M2, C,
                      impact_sigma_cm2(*this, 1, 427));
        o << b;
    }
    if (P.delta) {
        char b[160];
        std::snprintf(b, sizeof(b), "; delta electrons: dsigma/dW ~ 1/(J^2 + W^2), J = %g eV, %s angle", J_eV,
                      P.delta_iso ? "isotropic" : "binary-collision");
        o << b;
        if (!shells.empty()) {
            int ne = 0;
            for (const auto& s : shells) ne += s.second;
            o << ", free-electron tail of " << shells.size() << " inner subshells (" << ne << " electrons, B >= "
              << shells.back().first << " eV)";
        }
    }
    o << "; electrons -> '" << product << "'";
    if (nsplit > 1) o << "; first ionization split in " << nsplit << " quanta";
    return o.str();
}

} // namespace quarz
