#include "Ionization.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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
// Drift Chambers, Table 1.2) to < 0.3 %: ~25 (Ar), ~43 (Xe) primary ion pairs/cm at NTP.
// Rb (no gas-cell data): high-energy limit of the counting cross-section of Kim, Migdalek, Siegel & Bieron,
// PRA 57, 246 (1998) (5s BED + core BEB + 4s, 4p excitation-autoionization in the Born approximation; agrees with
// the absolute MOT data of Schappe et al., PRL 76, 4328 (1996) up to 500 eV), n = 1, 2 shells added by BEB:
// M^2 = 5.89, C = 65.8 (sigma = 2.46e-18 cm^2 at gamma = 427, +-15 %). The dipole integral of the Henke
// photoabsorption tables gives M^2 = 6.01 independently. These are prompt atomic values: for He-Xe the gas-cell
// M^2 of Rieke & Prepejchal lies 9-50 % above the dipole (Henke) M^2.  No data for H atoms, Li, Cs and the atomic
// C, N, O: give <species>.impact.M2 and <species>.impact.C, or <species>.impact.sigma_cm2.
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
        {"Rb", {85.468, {4.17713, 27.2895, 40.0}, 5.89, 65.8, 0}},
        {"Xe", {131.293, {12.1298, 20.975, 32.1230}, 8.04, 72.35, 8.7,
                {{34561, 2}, {5453, 2}, {5107, 2}, {4786, 4}, {1148.7, 2}, {1002.1, 2}, {940.6, 4}, {689.0, 4}, {676.4, 6}}}},
        {"Cs", {132.905, {3.8939, 23.15744}, 0, 0, 0}},
    };
    return t;
}
// electron-impact ionization: Lotz, Z. Phys. 206, 205 (1967), Table 1 (fits to the measured cross-sections;
// formula (7): common a, b, c with U = E/P_1); P_1 is replaced by the ionization energy of the table above
struct LotzFit {
    std::vector<double> P;
    std::vector<int> q;
    double a, b, c;
};
const std::map<std::pair<std::string, int>, LotzFit>& lotz_fits() {
    static const std::map<std::pair<std::string, int>, LotzFit> t = {
        {{"H", 0}, {{13.6}, {1}, 4.0, 0.60, 0.56}},
        {{"He", 0}, {{24.6}, {2}, 4.0, 0.75, 0.46}},
        {{"He", 1}, {{54.4}, {1}, 4.4, 0.38, 0.60}},
        {{"Li", 0}, {{5.39, 58}, {1, 2}, 4.0, 0.70, 2.4}},
        {{"Li", 1}, {{75.6}, {2}, 4.0, 0.48, 0.60}},
        {{"N", 0}, {{14.5, 20.3}, {3, 2}, 3.4, 0.83, 0.22}},
        {{"N", 1}, {{29.6, 36.7}, {2, 2}, 4.1, 0.46, 0.62}},
        {{"O", 0}, {{13.6, 28.5}, {4, 2}, 2.8, 0.74, 0.24}},
        {{"Ne", 0}, {{21.6, 48.5}, {6, 2}, 2.8, 0.92, 0.19}},
        {{"Ne", 1}, {{41.1, 66.4}, {5, 2}, 3.2, 0.83, 0.48}},
        {{"Ar", 0}, {{15.8, 29.2}, {6, 2}, 4.0, 0.62, 0.40}},
        {{"Kr", 0}, {{14.0, 27.5, 93}, {6, 2, 10}, 4.0, 0.71, 0.76}},
        {{"Rb", 0}, {{4.18, 15.3}, {1, 6}, 4.0, 0, 0}},
        {{"Xe", 0}, {{12.1, 23.4, 67}, {6, 2, 10}, 4.0, 0.54, 0.64}},
        {{"Cs", 0}, {{3.89, 12.3}, {1, 6}, 4.0, 0, 0}},
    };
    return t;
}
// excitation of the neutral atoms: (W eV, F, c, allowed); a = b = 1.  Xe: Puech & Mizzi (1991) as tabulated
// by Mukundan & Bhardwaj, arXiv:1604.08449, Table 3 (c = 0: allowed, F = optical oscillator strength)
struct ExcLevel { double W, F, c; bool allowed; };
const std::map<std::string, std::vector<ExcLevel>>& excitation_levels() {
    static const std::map<std::string, std::vector<ExcLevel>> t = {
        {"Xe", {{8.315, 53.7, 2, false}, {8.437, 0.26, 0, true}, {9.447, 27.0, 2, false}, {9.570, 0.19, 0, true},
                {9.580, 2.57, 1, false}, {9.706, 4.89, 1, false}, {9.809, 2.54, 1, false}, {9.917, 0.01, 0, true},
                {10.11, 20.0, 1, false}, {10.40, 0.395, 0, true}, {10.59, 0.097, 0, true}, {11.0, 5.0, 1, false},
                {11.7, 0.689, 0, true}}},
    };
    return t;
}
const std::map<std::string, int>& atomic_numbers() {
    static const std::map<std::string, int> t = {{"H", 1}, {"He", 2}, {"Li", 3}, {"C", 6}, {"N", 7}, {"O", 8},
                                                {"Ne", 10}, {"Ar", 18}, {"Kr", 36}, {"Rb", 37}, {"Xe", 54}, {"Cs", 55}};
    return t;
}
// electrons in the outermost subshell of the ion with charge z: the neutral configuration (Madelung order),
// then the electrons removed from the subshell with the largest n (then l)
int outer_subshell_electrons(int Z, int z) {
    struct Sub { int n, l, cap, occ; };
    const int ord[][2] = {{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {4, 0}, {3, 2}, {4, 1}, {5, 0}, {4, 2}, {5, 1},
                          {6, 0}, {4, 3}, {5, 2}, {6, 1}, {7, 0}};
    std::vector<Sub> s;
    int left = Z;
    for (const auto& o : ord) {
        if (left == 0) break;
        const int cap = 2 * (2 * o[1] + 1), occ = std::min(cap, left);
        s.push_back({o[0], o[1], cap, occ});
        left -= occ;
    }
    auto outer = [&]() {
        int k = -1;
        for (int i = 0; i < static_cast<int>(s.size()); ++i)
            if (s[i].occ > 0 && (k < 0 || s[i].n > s[k].n || (s[i].n == s[k].n && s[i].l > s[k].l))) k = i;
        return k;
    };
    for (int r = 0; r < z; ++r) { const int k = outer(); if (k >= 0) --s[k].occ; }
    const int k = outer();
    return k >= 0 ? s[k].occ : 0;
}
std::vector<std::pair<double, double>> read_table(const std::string& fn) {
    std::ifstream f(fn);
    if (!f) throw std::runtime_error("cannot open the cross-section table " + fn);
    std::vector<std::pair<double, double>> t;
    std::string line;
    while (std::getline(f, line)) {
        const auto h = line.find('#');
        if (h != std::string::npos) line.erase(h);
        std::istringstream is(line);
        double e, s;
        if (is >> e >> s) t.push_back({e, s});
    }
    if (t.size() < 2) throw std::runtime_error(fn + ": need at least two lines 'E_eV sigma_cm2'");
    for (size_t i = 1; i < t.size(); ++i)
        if (!(t[i].first > t[i - 1].first)) throw std::runtime_error(fn + ": energies must increase");
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
    I.quiet = cfg.get_bool(name + ".ionization.quiet", false);
    // electron-impact ionization by the plasma electrons: cross-section parameters of every level
    P.eimp = cfg.get_bool(name + ".ionization.electron_impact", false);
    if (P.eimp) {
        int Z = 0;
        if (!I.element.empty() && atomic_numbers().count(I.element)) Z = atomic_numbers().at(I.element);
        const double J0 = cfg.get_double(name + ".impact.J_eV", J > 0 ? J : ip[P.z0]);
        std::vector<std::string> qlist;
        if (cfg.has(name + ".eimpact.q")) qlist = cfg.get_list(name + ".eimpact.q");
        for (int l = P.z0; l < P.zmax; ++l) {
            std::vector<double> p(EI_NPAR, 0.0);
            p[0] = ip[l];
            p[11] = J0 * ip[l] / ip[P.z0];
            const std::string tk = name + ".eimpact.table_" + std::to_string(l);
            std::vector<double> te, ts;
            if (cfg.has(tk)) {
                for (const auto& v : read_table(cfg.get_string(tk))) { te.push_back(v.first); ts.push_back(v.second); }
            } else {
                auto it = lotz_fits().find({I.element, l});
                if (it != lotz_fits().end() && !cfg.has(name + ".ionization_energies_eV")) {
                    const LotzFit& L = it->second;
                    p[1] = static_cast<double>(L.P.size());
                    for (size_t i = 0; i < L.P.size(); ++i) { p[2 + i] = i == 0 ? ip[l] : L.P[i]; p[5 + i] = L.q[i]; }
                    p[8] = L.a; p[9] = L.b; p[10] = L.c;
                } else {
                    int q = 0;
                    if (!qlist.empty()) {
                        if (static_cast<int>(qlist.size()) != P.zmax - P.z0)
                            throw std::runtime_error(name + ".eimpact.q: one number per level z0 .. z_max-1");
                        q = std::stoi(qlist[l - P.z0]);
                    } else if (Z > 0) q = outer_subshell_electrons(Z, l);
                    else
                        throw std::runtime_error(name + ".ionization.electron_impact: give " + name + ".eimpact.q (electrons in "
                                                 "the outermost subshell, per level) or " + tk);
                    p[1] = 1; p[2] = ip[l]; p[5] = q; p[8] = 4.5;
                }
            }
            I.ei_par.push_back(p);
            I.ei_tabE.push_back(te);
            I.ei_tabS.push_back(ts);
        }
        // excitation of the neutral atoms (item C): built in (Xe) or a table; on by default when data exist
        const bool tab = cfg.has(name + ".excitation.table");
        const auto xl = excitation_levels().find(I.element);
        const bool have = tab || (xl != excitation_levels().end() && !cfg.has(name + ".ionization_energies_eV"));
        if (cfg.get_bool(name + ".excitation", have) && P.z0 == 0) {
            if (!have)
                throw std::runtime_error(name + ".excitation: no built-in data for this gas; give " + name +
                                         ".excitation.table (E_eV sigma_cm2) and " + name + ".excitation.energy_eV");
            if (tab) {
                for (const auto& v : read_table(cfg.get_string(name + ".excitation.table"))) {
                    I.ex_tabE.push_back(v.first); I.ex_tabS.push_back(v.second);
                }
                if (!cfg.has(name + ".excitation.energy_eV"))
                    throw std::runtime_error(name + ".excitation.table needs " + name + ".excitation.energy_eV");
                I.ex_ch.push_back({cfg.get_double(name + ".excitation.energy_eV"), 2, 0, static_cast<double>(I.ex_tabE.size()), 1, 1});
            } else {
                for (const auto& L : xl->second) I.ex_ch.push_back({L.W, L.allowed ? 0.0 : 1.0, L.F, L.c, 1, 1});
            }
        }
    }
    I.nsplit = cfg.get_int(name + ".ionization.nsplit", 1);
    if (I.nsplit < 1) throw std::runtime_error(name + ".ionization.nsplit must be >= 1");
    I.product = cfg.get_string(name + ".ionization.product", "electrons");
    return I;
}

EITargets make_ei_targets(std::vector<IonSpeciesInfo*>& ions, double n0_cm3) {
    EITargets T;
    std::vector<double> par, tE, tS, ex;
    int nx = 0;
    for (IonSpeciesInfo* I : ions) {
        if (!I->P.eimp) continue;
        I->P.et0 = T.T;
        for (size_t l = 0; l < I->ei_par.size(); ++l) {
            std::vector<double> p = I->ei_par[l];
            if (l == 0 && !I->ex_ch.empty()) {
                p[14] = nx;
                p[15] = static_cast<double>(I->ex_ch.size());
                for (auto c : I->ex_ch) {
                    if (c[1] == 2) {   // table channel: offset into the concatenated tables
                        c[2] = static_cast<double>(tE.size());
                        tE.insert(tE.end(), I->ex_tabE.begin(), I->ex_tabE.end());
                        tS.insert(tS.end(), I->ex_tabS.begin(), I->ex_tabS.end());
                    }
                    ex.insert(ex.end(), c.begin(), c.end());
                    ++nx;
                }
            }
            if (!I->ei_tabE[l].empty()) {
                p[12] = static_cast<double>(tE.size());
                p[13] = static_cast<double>(I->ei_tabE[l].size());
                tE.insert(tE.end(), I->ei_tabE[l].begin(), I->ei_tabE[l].end());
                tS.insert(tS.end(), I->ei_tabS[l].begin(), I->ei_tabS[l].end());
            }
            par.insert(par.end(), p.begin(), p.end());
            ++T.T;
        }
    }
    if (T.T == 0) return T;
    T.par = View2D("ei.par", T.T, EI_NPAR);
    auto hp = Kokkos::create_mirror_view(T.par);
    for (int t = 0; t < T.T; ++t)
        for (int c = 0; c < EI_NPAR; ++c) hp(t, c) = par[static_cast<size_t>(t) * EI_NPAR + c];
    Kokkos::deep_copy(T.par, hp);
    T.exc = View2D("ei.exc", std::max(nx, 1), EX_NPAR);
    auto hx = Kokkos::create_mirror_view(T.exc);
    for (int j = 0; j < nx; ++j)
        for (int c = 0; c < EX_NPAR; ++c) hx(j, c) = ex[static_cast<size_t>(j) * EX_NPAR + c];
    Kokkos::deep_copy(T.exc, hx);
    const size_t nt = std::max<size_t>(tE.size(), 1);
    T.tabE = View1D("ei.tabE", nt);
    T.tabS = View1D("ei.tabS", nt);
    auto he = Kokkos::create_mirror_view(T.tabE);
    auto hs = Kokkos::create_mirror_view(T.tabS);
    for (size_t i = 0; i < tE.size(); ++i) { he(i) = tE[i]; hs(i) = tS[i]; }
    Kokkos::deep_copy(T.tabE, he);
    Kokkos::deep_copy(T.tabS, hs);
    const double wp = 5.64146e4 * std::sqrt(n0_cm3);
    T.norm = n0_cm3 * 2.99792458e10 / wp;
    return T;
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
    if (P.eimp) {
        o << "; electron impact (Lotz";
        for (size_t l = 0; l < ei_par.size(); ++l)
            if (!ei_tabE[l].empty()) o << ", table for z = " << P.z0 + l;
        o << ")";
        if (!ex_ch.empty()) o << " with excitation (" << ex_ch.size() << " channel" << (ex_ch.size() > 1 ? "s" : "") << ")";
    }
    o << "; electrons -> '" << product << "'";
    if (nsplit > 1) o << "; first ionization split in " << nsplit << " quanta";
    if (quiet) o << "; quiet (accumulator emission, stratified draws)";
    return o.str();
}

} // namespace quarz
