// Ionization of plasma ion species (atoms / ions with a per-particle charge state).
//
// Three channels, all evaluated slice by slice during the quasi-static sweep, after the fields
// of the slice are known and before the particles are pushed to the next slice:
//
//  1. Field (tunnel) ionization by the plasma and beam fields: ADK rate for the static field
//     |E| = (E_x^2 + E_y^2 + E_z^2)^{1/2} (Ammosov-Delone-Krainov with l = 0, m = 0 and
//     l* = n* - 1, as in WarpX/HiPACE++/Smilei, Chen et al. JCP 236, 220 (2013)):
//        W = Ip C^2 [2 (2 Ip)^{3/2} / E]^{2 n* - 1} exp[-2 (2 Ip)^{3/2} / (3 E)]   (atomic units)
//        C^2 = 2^{2 n*} / (n* Gamma(2 n*)),  n* = Z / (2 Ip)^{1/2},  Z = charge after ionization.
//  2. Field ionization by the laser (envelope model): the ADK rate of the total field
//     E(phi) = E_slow + E_L cos(phi) e_x (linear) or E_slow + E_L (cos phi, sin phi) (circular),
//     E_L = k0 |a^|, averaged over the laser period phi by quadrature around the field maxima
//     (two per period for linear, one for circular polarisation): Gauss-Hermite scaled with the
//     Laplace width sigma_phi = |d^2 ln W / d phi^2|^{-1/2} for narrow peaks, Gauss-Legendre over
//     the lobe otherwise.  (The usual Laplace formula W(E_L) [3 E_L / (pi (2 Ip)^{3/2})]^{1/2} is
//     10-40 % too high where the ionization actually happens.)  The static rate is recovered for
//     E_L -> 0.  Electrons born in the laser get the residual (drift) momentum p_perp = -a(phi_birth),
//     with the birth phase sampled exactly from W(E(phi)) by rejection (cf. Massimo et al.,
//     PRE 102, 033204 (2020), who use the Gaussian of the Laplace expansion);
//     Delta = gamma - p_z of the new electron equals that of its parent (born at rest in the
//     wave), so its cycle-averaged p_z follows from the envelope gamma.
//  3. Impact ionization by beam particles (charge z_b, velocity beta c) for the first ionization
//     (neutral -> 1+), Bethe cross-section in the form of Rieke & Prepejchal, PRA 6, 1507 (1972):
//        sigma = 4 pi a0^2 (R / (m c^2 / 2)) z_b^2 beta^-2 [ M^2 (ln(beta^2 gamma^2) - beta^2) + C ]
//              = 1.874e-20 cm^2 z_b^2 beta^-2 [ ... ],
//     or a constant sigma.  An atom at rest sees  dP = sigma n_b beta dxi  per slice (n_b: beam
//     density in the box frame); the beams deposit  N0 = sum z^2 beta n,  N1 = sum z^2 L n / beta,
//     N2 = sum z^2 n / beta  (L = ln(beta^2 gamma^2) - beta^2), so  rate/dxi = s0 N0 + s1 N1 + s2 N2.
//     Delta electrons (<name>.impact.secondary_energy = 1): the electron of a beam impact ionization
//     gets a kinetic energy W from  dsigma/dW ~ 1/(J^2 + W^2),  0 <= W <= W_max  (Opal, Peterson & Beaty,
//     J. Chem. Phys. 55, 4100 (1971); normalised to the total cross-section above, so the ionization
//     rate is unchanged), sampled exactly by W = J tan(u atan(W_max/J)), and the polar angle of a binary
//     collision with a free electron at rest,  cos(theta) = kappa sqrt(W/(W + 2 m c^2)),
//     kappa = (E_b + m c^2)/(p_b c)  (exact two-body kinematics; 1/beta for heavy projectiles), or an
//     isotropic direction.  W_max = 2 m c^2 beta^2 gamma^2/(1 + 2 gamma m/M + (m/M)^2), (gamma - 1)/2 m c^2
//     for electrons (the slower of the two is the secondary), (gamma - 1) m c^2 for positrons.  Several
//     beams: the beams deposit two more sums  N3 = sum z^2 n ln(W_max)/beta,  N4 = sum z^2 n kappa/beta,
//     and an atom uses W_max = exp(N3/N2), kappa = N4/N2 (exact for one kind of projectile).
//     Inner shells (heavy gases: the 1/W^2 tail of the fit above covers only the outer electrons): a
//     subshell with n_i electrons bound by B_i adds the free-electron (Rutherford) cross-section
//     dsigma/dE = 2 pi r_e^2 m c^2 z^2 n_i / (beta^2 E^2) for the energy transfer B_i <= E <= W_max; the
//     ejected electron gets W = E - B_i (the vacancy energy, Auger electrons, is not followed).  Their
//     rate s_in N2, s_in = 2 pi r_e^2 (m c^2) sum n_i/B_i n0 k_p^-1, is taken out of the total (Bethe)
//     rate, so the ionization rate is unchanged; an impact ionization is an inner-shell one with
//     probability s_in N2 / R_imp.
//
// Monte Carlo: per slice the probability of the transition z -> z+1 is P = 1 - exp(-W dt - R dxi),
// dt = dxi gamma/Delta (time an ion spends in the slice; = dxi for atoms at rest).  Several
// levels may be crossed in one slice.  The macro-particle changes its charge state and a
// macro-electron of the same weight is created at its position (in the product species), with the
// parent's velocity plus the laser drift momentum.  Optional splitting (nsplit > 1) for the first
// ionization of each loaded atom: the expected ionized weight dN = w P of a slice is emitted as a
// new ion + electron pair (weight dN if dN >= w0/nsplit, otherwise weight w0/nsplit with
// probability dN nsplit / w0 -- unbiased), and the parent keeps the rest.  This resolves small
// ionization fractions (impact ionization, onset of field ionization) without the all-or-nothing
// noise of whole macro-particles.  Random numbers come from a counter-based hash of
// (seed, step, slice, species, particle, draw): results do not depend on threads or MPI ranks.
#pragma once

#include "Types.hpp"

#include <cstdint>
#include <utility>
#include <string>
#include <vector>

namespace quarz {

class Config;

constexpr int ION_MAX_LEVELS = 36;
constexpr int ION_MAX_SHELLS = 12;

// device-side parameters of one ionizable species
struct IonParams {
    int z0 = 0, zmax = 0;                 // initial and highest charge state
    Real lnA[ION_MAX_LEVELS] = {};        // ln(ADK prefactor), rate in units of omega_p, transition z -> z+1
    Real pw[ION_MAX_LEVELS] = {};         // 2 n* - 1
    Real kap[ION_MAX_LEVELS] = {};        // (2/3) (2 Ip)^{3/2}   [atomic field units]
    Real econv = 0;                       // field (m c omega_p / e) -> atomic units
    bool field = true;                    // ADK by the plasma/beam (and laser) fields
    bool impact = false;                  // beam impact ionization 0 -> 1
    Real s0 = 0, s1 = 0, s2 = 0;          // impact: dP/dxi = s0 N0 + s1 N1 + s2 N2
    bool delta = false;                   // impact-born electrons get the delta-electron energy and angle
    bool delta_iso = false;               //   isotropic emission instead of the binary-collision angle
    Real dJ = 0;                          //   width J of the energy spectrum dsigma/dW ~ 1/(J^2 + W^2), m c^2
    int nsh = 0;                          //   inner subshells with a free-electron (Rutherford) tail
    Real shB[ION_MAX_SHELLS] = {};        //     binding energies, m c^2
    Real shP[ION_MAX_SHELLS] = {};        //     cumulative probabilities ~ n_i / B_i
    Real s_in = 0;                        //     inner-shell rate = s_in N2
    bool eimp = false;                    // electron-impact ionization by the plasma electrons (all levels)
    int et0 = 0;                          //   index of level z0 in the electron-impact target table
    bool laser = false, circular = false; // laser envelope present, its polarisation
    Real k0 = 0;                          // omega0 / omega_p
};

// ---------------------------------------------------------------------------- device helpers

constexpr Real EI_ME_EV = 510998.95;
// ---- electron-impact ionization by plasma electrons (<name>.ionization.electron_impact)
// Targets t = (ionizable species, level z -> z+1).  Cross-section: Lotz, Z. Phys. 206, 205 (1967),
//   sigma = a {1 - b exp[-c (E/P_1 - 1)]} sum_i q_i ln(E/P_i)/(E P_i),   E >= P_i,
// with the experimentally fitted a, b, c and subshells of his Table 1 for the neutral atoms (and He+,
// Ne+, Li+, N+), else Lotz's high-charge form a = 4.5e-14 cm^2 eV^2, b = c = 0 with the outermost
// subshell of the ion (P_1 = Ip(z), q_1 its electrons), or a table sigma(E) (<name>.eimpact.table_<z>).
// Rate per slice of an atom at rest: R = sum_e n_e sigma(E_e) v_e = sum_e w W/V |u|/Delta sigma(E_e): the
// electrons deposit it per target, binned in E/Ip (bins [2^(b/4), 2^((b+1)/4)) Ip, b = 0 .. EI_NB-1, the
// last one open) together with sum r E, so that the ionization can pick the projectile energy (bin ~ rate, rate-
// weighted mean energy of the bin) for the ejected electron: W from dsigma/dW ~ 1/(J^2 + W^2) on
// [0, (E - Ip)/2], isotropic.  The projectiles lose their expected energy n_t sigma_t (Ip + <W>) per unit
// path (continuous slowing down; energy conserved on average).
// Excitation (<name>.excitation, item C): channels j of a target (neutral atoms) with threshold W_j add
// n sigma_j(E) W_j to the energy loss per unit path; no other effect (no metastables, no radiation).
// Built in for Xe: the 13 levels of Puech & Mizzi (J. Phys. B 24, 4689 (1991)) as tabulated by Mukundan &
// Bhardwaj (arXiv:1604.08449, Table 3), Born-Bethe with a low-energy modifier, x = m c^2 beta^2 (eV):
//   allowed:   sigma = 8 pi a0^2 R^2 F/(x W) [ln(x/(1 - beta^2)/(2 W)) - beta^2] B,
//   forbidden: sigma = 8 pi a0^2 R F/x B,     B = [1 - (2W/x)^a]^b / x^c   (a = b = 1);
// or a table sigma(E) with one excitation energy (<name>.excitation.table, .energy_eV).
constexpr int EI_BPO = 4;                 // energy bins per octave of E/Ip
constexpr int EI_NB = 12 * EI_BPO;        // up to 4096 Ip (the last bin is open)
// par columns: 0 Ip (eV), 1 nsub, 2-4 P_i (eV), 5-7 q_i, 8 a (1e-14 cm^2 eV^2), 9 b, 10 c, 11 J (eV),
//              12 table offset, 13 table length (0: Lotz), 14 first excitation channel, 15 channels
constexpr int EI_NPAR = 16;
// excitation channel columns: 0 W (eV), 1 kind (0 allowed, 1 forbidden, 2 table), 2 F (table: offset),
//                             3 c (table: length), 4 a, 5 b
constexpr int EX_NPAR = 6;
struct EITargets {
    int T = 0;
    View2D par;          // T x EI_NPAR
    View2D exc;          // excitation channels x EX_NPAR
    View1D tabE, tabS;   // concatenated tables: E (eV), sigma (cm^2)
    Real norm = 0;       // n0 [cm^-3] k_p^-1 [cm]: sigma (cm^2) -> rate per unit normalised density and length
};

KOKKOS_INLINE_FUNCTION Real ei_table(const EITargets& T, int o, int nt, Real E, Real Ip) {   // linear; ~ln(E/Ip)/E beyond
    if (E <= T.tabE(o)) return Real(0);
    if (E >= T.tabE(o + nt - 1)) {
        const Real El = T.tabE(o + nt - 1);
        return T.tabS(o + nt - 1) * (Kokkos::log(E / Ip) / E) / (Kokkos::log(El / Ip) / El);
    }
    int lo = o, hi = o + nt - 1;
    while (hi - lo > 1) { const int m = (lo + hi) / 2; if (T.tabE(m) <= E) lo = m; else hi = m; }
    return T.tabS(lo) + (E - T.tabE(lo)) / (T.tabE(hi) - T.tabE(lo)) * (T.tabS(hi) - T.tabS(lo));
}

KOKKOS_INLINE_FUNCTION Real ei_sigma(const EITargets& T, int t, Real E) {   // E in eV, sigma in cm^2
    const Real Ip = T.par(t, 0);
    if (!(E > Ip)) return Real(0);
    const int nt = static_cast<int>(T.par(t, 13));
    if (nt > 0) return ei_table(T, static_cast<int>(T.par(t, 12)), nt, E, Ip);
    const int ns = static_cast<int>(T.par(t, 1));
    Real s = 0;
    for (int i = 0; i < ns; ++i) {
        const Real P = T.par(t, 2 + i);
        if (E > P) s += T.par(t, 5 + i) * Kokkos::log(E / P) / (E * P);
    }
    return Real(1e-14) * T.par(t, 8) * (Real(1) - T.par(t, 9) * Kokkos::exp(-T.par(t, 10) * (E / Ip - Real(1)))) * s;
}

// cross-section (cm^2) of excitation channel j at E (eV)
KOKKOS_INLINE_FUNCTION Real ei_exc_sigma(const EITargets& T, int j, Real E) {
    const Real W = T.exc(j, 0);
    if (!(E > W)) return Real(0);
    const int kind = static_cast<int>(T.exc(j, 1));
    if (kind == 2) return ei_table(T, static_cast<int>(T.exc(j, 2)), static_cast<int>(T.exc(j, 3)), E, W);
    const Real g = Real(1) + E / EI_ME_EV, b2 = Real(1) - Real(1) / (g * g), x = EI_ME_EV * b2;
    constexpr Real A0 = 7.0378e-16, RY = 13.605693;   // 8 pi a0^2 (cm^2), Rydberg (eV)
    const Real B = Kokkos::pow(Kokkos::fmax(Real(1) - Kokkos::pow(Real(2) * W / x, T.exc(j, 4)), Real(0)), T.exc(j, 5)) /
                   Kokkos::pow(x, T.exc(j, 3));
    const Real sg = kind == 0 ? A0 * RY * RY * T.exc(j, 2) / (x * W) * (Kokkos::log(x / (Real(1) - b2) / (Real(2) * W)) - b2) * B
                              : A0 * RY * T.exc(j, 2) / x * B;
    return Kokkos::fmax(sg, Real(0));
}

// excitation energy-loss cross-section sum_j sigma_j(E) W_j (cm^2 eV) of target t, E in eV
KOKKOS_INLINE_FUNCTION Real ei_exc_loss(const EITargets& T, int t, Real E) {
    const int n = static_cast<int>(T.par(t, 15)), o = static_cast<int>(T.par(t, 14));
    Real s = 0;
    for (int j = o; j < o + n; ++j) s += ei_exc_sigma(T, j, E) * T.exc(j, 0);
    return s;
}

// mean energy of the ejected electron (eV) for a projectile of energy E (eV): W_max = (E - Ip)/2
KOKKOS_INLINE_FUNCTION Real ei_mean_w(const EITargets& T, int t, Real E) {
    const Real Ip = T.par(t, 0), J = T.par(t, 11);
    const Real wm = Real(0.5) * (E - Ip);
    if (!(wm > Real(0))) return Real(0);
    const Real x = wm / J;
    return J * Kokkos::log1p(x * x) / (Real(2) * Kokkos::atan(x));
}
KOKKOS_INLINE_FUNCTION uint64_t ion_mix(uint64_t z) {   // splitmix64 finaliser
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// uniform in [0, 1) from (key, particle, draw)
KOKKOS_INLINE_FUNCTION Real ion_rand(uint64_t key, uint64_t i, uint64_t d) {
    const uint64_t h = ion_mix(key ^ ion_mix(i * 0x632be59bd9b4e019ULL + ion_mix(d + 0x1234567ULL)));
    return static_cast<Real>(h >> 11) * Real(1.1102230246251565e-16);   // 2^-53
}

// static ADK rate (units of omega_p) for the transition from level l, field E in atomic units
KOKKOS_INLINE_FUNCTION Real adk_rate(const IonParams& P, int l, Real E) {
    if (!(E > Real(0))) return Real(0);
    const Real ex = P.kap[l] / E;
    if (ex > Real(700)) return Real(0);
    return Kokkos::exp(P.lnA[l] - P.pw[l] * Kokkos::log(E) - ex);
}

struct IonRate {
    Real W = 0;          // cycle-averaged field-ionization rate (units of omega_p)
    Real wl[2] = {0, 0}; // contributions of the lobes (field maxima phi = 0, pi for linear; one for circular)
    Real sg[2] = {Real(1e30), Real(1e30)};   // Laplace width of the birth phase around each lobe
};

// field over the laser period around a lobe:  E(phi)^2 = A + B cos^2 phi + C cos phi  (phi = 0 at the maximum)
KOKKOS_INLINE_FUNCTION Real ion_E(Real A, Real B, Real C, Real ph) {
    const Real c = Kokkos::cos(ph);
    return Kokkos::sqrt(Kokkos::fmax(A + B * c * c + C * c, Real(0)));
}

// period average (1/2pi) int_{-H}^{H} W(E(phi)) dphi of one lobe (H = pi/2 linear, pi circular).
// Gauss-Hermite (12 nodes) around the maximum, scaled with the Laplace width, when the peak is
// narrow; Gauss-Legendre (24 nodes) over the whole lobe otherwise.  Negligible rates: Laplace only.
KOKKOS_INLINE_FUNCTION Real ion_lobe(const IonParams& P, int l, Real A, Real B, Real C, Real H, Real& sg) {
    const Real E = ion_E(A, B, C, Real(0));
    const Real W = adk_rate(P, l, E);
    sg = Real(1e30);
    if (W == Real(0)) return Real(0);
    const Real c2 = E > Real(0) ? (B + Real(0.5) * C) / E : Real(0);   // -d^2E/dphi^2 at phi = 0
    const Real dl = P.kap[l] / (E * E) - P.pw[l] / E;                     // d ln W / dE
    const Real f2 = dl * c2;
    const Real fmax = H / Real(PI);                                       // weight of the whole lobe
    if (f2 > Real(0)) sg = Real(1) / Kokkos::sqrt(f2);
    // negligible (< 1e-10 omega_p) or saturated (> 1e3 omega_p: P = 1 in any slice) rates: Laplace is enough
    const Real Wl = f2 > Real(0) ? W * Kokkos::fmin(fmax, sg / Kokkos::sqrt(Real(2) * Real(PI))) : W * fmax;
    if (Wl < Real(1e-10) || Wl > Real(1e3)) return Wl;
    if (sg * Real(1.41421356237) * Real(3.8897249) < H) {
        constexpr Real t[6] = {0.31424037625435913, 0.9477883912401638, 1.5976826351526048, 2.2795070805010598,
                               3.0206370251208896, 3.889724897869782};
        constexpr Real wt[6] = {0.6293078743694928, 0.6396212320202566, 0.662662773266872, 0.7052203661122188,
                                0.7866439394633225, 0.9896990470922982};
        const Real s2 = sg * Real(1.41421356237);
        Real sum = 0;
        for (int k = 0; k < 6; ++k) sum += wt[k] * Real(2) * adk_rate(P, l, ion_E(A, B, C, s2 * t[k]));
        return sum * s2 / (Real(2) * Real(PI));
    }
    constexpr Real x[12] = {0.06405689286260563, 0.1911188674736163, 0.3150426796961634, 0.4337935076260451,
                            0.5454214713888396, 0.6480936519369755, 0.7401241915785544, 0.820001985973903,
                            0.8864155270044011, 0.9382745520027328, 0.9747285559713095, 0.9951872199970213};
    constexpr Real wx[12] = {0.12793819534675202, 0.12583745634682825, 0.1216704729278033, 0.11550566805372552,
                             0.10744427011596556, 0.09761865210411393, 0.0861901615319532, 0.07334648141108016,
                             0.05929858491543636, 0.04427743881741941, 0.02853138862893356, 0.01234122979998869};
    Real sum = 0;
    for (int k = 0; k < 12; ++k) sum += wx[k] * Real(2) * adk_rate(P, l, ion_E(A, B, C, H * x[k]));
    return sum * H / (Real(2) * Real(PI));
}

// lobe coefficients (A, B, C) and half-range H
KOKKOS_INLINE_FUNCTION void ion_lobe_coeffs(const IonParams& P, int s, Real ex, Real ey, Real ez, Real EL,
                                            Real& A, Real& B, Real& C, Real& H) {
    const Real es2 = ex * ex + ey * ey + ez * ez;
    if (!P.circular) {   // E = E_s + E_L cos(phi) e_x, lobe s = 0: phi = 0, s = 1: phi = pi
        A = es2; B = EL * EL; C = (s == 0 ? Real(2) : -Real(2)) * EL * ex; H = Real(0.5) * Real(PI);
    } else {             // E = E_s + E_L (cos, sin): |E|^2 = es2 + EL^2 + 2 EL |E_s,perp| cos(phi - phi_s)
        A = es2 + EL * EL; B = 0; C = Real(2) * EL * Kokkos::sqrt(ex * ex + ey * ey); H = Real(PI);
    }
}

// rate for slow field (ex, ey, ez) plus laser amplitude EL (all in atomic units)
KOKKOS_INLINE_FUNCTION IonRate ion_field_rate(const IonParams& P, int l, Real ex, Real ey, Real ez, Real EL) {
    IonRate o;
    if (!(EL > Real(0))) {
        o.W = adk_rate(P, l, Kokkos::sqrt(ex * ex + ey * ey + ez * ez));
        o.wl[0] = o.W;
        return o;
    }
    // linear polarisation: the two lobes are mirror images when E_s,x << E_L
    const int nl = (P.circular || Kokkos::fabs(ex) < Real(1e-6) * EL) ? 1 : 2;
    for (int s = 0; s < nl; ++s) {
        Real A, B, C, H;
        ion_lobe_coeffs(P, s, ex, ey, ez, EL, A, B, C, H);
        o.wl[s] = ion_lobe(P, l, A, B, C, H, o.sg[s]);
        o.W += o.wl[s];
    }
    if (nl == 1 && !P.circular) { o.wl[1] = o.wl[0]; o.sg[1] = o.sg[0]; o.W *= Real(2); }
    return o;
}

// ---------------------------------------------------------------------------- host side
struct IonSpeciesInfo {
    IonParams P;
    std::string element;          // "" if the energies were given explicitly
    std::vector<double> ip_eV;    // ionization energies used (levels z0 .. zmax-1)
    double mass_me = 0;           // default mass (electron masses), 0 if unknown
    int nsplit = 1;
    std::string product;          // electron species receiving the new electrons
    double M2 = 0, C = 0, sigma_cm2 = 0;
    double J_eV = 0;              // delta electrons: width of the energy spectrum
    std::vector<std::pair<double, int>> shells;   // inner subshells: binding energy (eV), electrons
    bool born_dump = false;       // <name>.ionization.born_dump: write the new electrons of every sweep
    // electron impact: per level z0 .. zmax-1 the parameters of EITargets::par (host) and the table
    std::vector<std::vector<double>> ei_par;
    std::vector<std::vector<double>> ei_tabE, ei_tabS;
    std::vector<std::vector<double>> ex_ch;            // excitation channels of level z0 (EX_NPAR each)
    std::vector<double> ex_tabE, ex_tabS;              // table channel
    std::string description() const;
};

// reads  <name>.element / <name>.ionization_energies_eV, z0, z_max, ionization.field/impact/...
// n0_cm3: the plasma (normalisation) density
IonSpeciesInfo make_ion_species(const Config& cfg, const std::string& name, double n0_cm3);
bool is_ionizable(const Config& cfg, const std::string& name);
// true if an ionizable species with delta electrons (impact.secondary_energy) sends its electrons to <name>
bool receives_delta_electrons(const Config& cfg, const std::string& name);
// the device target table of all species with electron-impact ionization (sets P.et0 of each)
EITargets make_ei_targets(std::vector<IonSpeciesInfo*>& ions, double n0_cm3);

// impact cross-section (cm^2) of a beam particle with charge z, Lorentz factor gamma (for the log)
double impact_sigma_cm2(const IonSpeciesInfo& I, double z, double gamma);

} // namespace quarz
