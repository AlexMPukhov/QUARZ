// Beam import from openPMD particle data: see OpenPMDReader.hpp.
#include "OpenPMDReader.hpp"

#include <stdexcept>

#ifndef QUARZ_USE_OPENPMD

namespace quarz {
OpenPMDBeamData read_openpmd_beam(const OpenPMDBeamRequest& req, const OpenPMDBlockSink&) {
    throw std::runtime_error(req.species + ": <beam>.profile = openpmd: this QUARZ build has no openPMD support "
                             "(install openPMD-api and reconfigure with -DopenPMD_ROOT=...)");
}
} // namespace quarz

#else

#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>

namespace quarz {

namespace {
constexpr double c_SI = 299792458.0, e_SI = 1.602176634e-19, me_SI = 9.1093837015e-31, eps0_SI = 8.8541878128e-12;

double skin_depth_m(double n0_cm3) {   // c / omega_p
    const double n = n0_cm3 * 1e6;
    return c_SI / std::sqrt(n * e_SI * e_SI / (eps0_SI * me_SI));
}

// the value of a constant record component (without unitSI)
double constant_value(const openPMD::RecordComponent& rc) {
    const auto a = rc.getAttribute("value");
    if (auto d = a.getOptional<double>()) return *d;
    if (auto f = a.getOptional<float>()) return *f;
    if (auto i = a.getOptional<int64_t>()) return static_cast<double>(*i);
    throw std::runtime_error("openPMD import: unsupported type of a constant record component");
}

// particles [i0, i0 + m) of a record component (of n) times its unitSI into v[0..m) (constant components
// included; double or float data): the import reads block by block and never holds a whole species
void load_block(openPMD::Series& S, openPMD::RecordComponent rc, size_t n, size_t i0, size_t m, double* v) {
    const double u = rc.unitSI();
    if (rc.constant()) {
        std::fill(v, v + m, constant_value(rc) * u);
        return;
    }
    if (rc.getExtent().empty() || rc.getExtent()[0] != n)
        throw std::runtime_error("openPMD import: record components of different lengths");
    const openPMD::Offset o{static_cast<uint64_t>(i0)};
    const openPMD::Extent e{static_cast<uint64_t>(m)};
    if (rc.getDatatype() == openPMD::Datatype::FLOAT) {
        auto d = rc.loadChunk<float>(o, e);
        S.flush();
        for (size_t i = 0; i < m; ++i) v[i] = d.get()[i] * u;
    } else {
        auto d = rc.loadChunk<double>(o, e);
        S.flush();
        for (size_t i = 0; i < m; ++i) v[i] = d.get()[i] * u;
    }
}

template <class T> double attr_or(const T& obj, const std::string& name, double def) {
    if (!obj.containsAttribute(name)) return def;
    const auto a = obj.getAttribute(name);
    if (auto d = a.template getOptional<double>()) return *d;
    if (auto i = a.template getOptional<int64_t>()) return static_cast<double>(*i);
    if (auto u = a.template getOptional<uint64_t>()) return static_cast<double>(*u);
    return def;
}

// a file of a file-based series (prefix_NNNNNN.ext) -> pattern prefix_%0NT.ext and its iteration
std::string series_pattern(const std::string& file, long& it_from_name) {
    it_from_name = -1;
    if (file.find('%') != std::string::npos) return file;
    static const std::regex re(R"(^(.*?)(\d+)(\.(h5|bp|bp4|bp5|json))$)");
    std::smatch m;
    if (!std::regex_match(file, m, re)) return file;   // e.g. a group-based single file
    it_from_name = std::stol(m[2].str());
    return m[1].str() + "%0" + std::to_string(m[2].str().size()) + "T" + m[3].str();
}
} // namespace

OpenPMDBeamData read_openpmd_beam(const OpenPMDBeamRequest& req, const OpenPMDBlockSink& sink) {
    using namespace openPMD;
    OpenPMDBeamData out;
    long it_name = -1;
    if (req.file.find('%') == std::string::npos && !std::filesystem::exists(req.file))
        throw std::runtime_error(req.species + ": openPMD file '" + req.file + "' not found");
    const std::string path = series_pattern(req.file, it_name);
    std::unique_ptr<Series> Sp;
    try { Sp = std::make_unique<Series>(path, Access::READ_ONLY); }
    catch (const std::exception& e) {
        throw std::runtime_error(req.species + ": cannot open openPMD series '" + path + "': " + e.what());
    }
    Series& S = *Sp;
    long n_it = req.iteration >= 0 ? req.iteration : it_name;
    if (n_it < 0) {
        for (auto& kv : S.iterations) n_it = std::max<long>(n_it, static_cast<long>(kv.first));
        if (n_it < 0) throw std::runtime_error(req.species + ": no iterations in " + req.file);
    }
    if (!S.iterations.contains(static_cast<uint64_t>(n_it)))
        throw std::runtime_error(req.species + ": iteration " + std::to_string(n_it) + " not in " + req.file);
    Iteration it = S.iterations[static_cast<uint64_t>(n_it)];
    if (!it.particles.contains(req.species)) {
        std::string have;
        for (auto& kv : it.particles) have += " " + kv.first;
        throw std::runtime_error("openPMD import: no species '" + req.species + "' in iteration " + std::to_string(n_it) +
                                 " of " + req.file + " (species:" + (have.empty() ? " none" : have) + ")");
    }
    ParticleSpecies sp = it.particles[req.species];
    out.iteration = n_it;
    const size_t B = std::max<size_t>(1, req.block);   // particles per read
    for (const char* r : {"position", "momentum", "weighting"})
        if (!sp.contains(r)) throw std::runtime_error("openPMD import: species '" + req.species + "' has no " + r + " record");
    Record pos = sp["position"], mom = sp["momentum"], wr = sp["weighting"];
    const size_t n = pos["x"].getExtent().empty() ? 0 : pos["x"].getExtent()[0];
    if (n == 0) throw std::runtime_error("openPMD import: species '" + req.species + "' has no particles");

    // units
    // value x unitSI is SI by the openPMD standard (unitSI = 1 for data already in SI). QUARZ files written
    // without units.n0_cm3 hold code units with unitSI = 1 and say so in the series attribute quarz_units;
    // files of other codes in normalised units must be declared (<beam>.openpmd_units = normalised).
    bool si = true;
    std::string how = "openPMD standard";
    if (req.units == "si" || req.units == "normalised") { si = req.units == "si"; how = req.species + ".openpmd_units"; }
    else if (S.containsAttribute("quarz_units")) {
        si = S.getAttribute("quarz_units").get<std::string>().rfind("SI", 0) == 0;
        how = "QUARZ file";
    }
    double fL = 1, fW = 1, fP = 1.0 / req.mass;   // file values -> code units
    std::ostringstream u;
    if (si) {
        if (!(req.n0_deck_cm3 > 0) && !req.info_only)
            throw std::runtime_error("openPMD import: '" + req.file + "' is in SI units (" + how + "): set units.n0_cm3 to "
                                     "convert it, or " + req.species + ".openpmd_units = normalised if it is not");
        const double L = req.n0_deck_cm3 > 0 ? skin_depth_m(req.n0_deck_cm3) : 1.0;
        fL = 1.0 / L;
        fW = 1.0 / (req.n0_deck_cm3 * 1e6 * L * L * L);
        fP = 1.0 / (req.mass * me_SI * c_SI);
        u << "SI (" << how << "), converted with n0 = " << req.n0_deck_cm3 << " cm^-3";
    } else {
        u << "normalised (" << how << ")";
        if (req.n0_file_cm3 > 0 && req.n0_deck_cm3 > 0 && std::abs(req.n0_file_cm3 / req.n0_deck_cm3 - 1) > 1e-12) {
            const double Lf = skin_depth_m(req.n0_file_cm3), Ld = skin_depth_m(req.n0_deck_cm3);
            fL = Lf / Ld;
            fW = req.n0_file_cm3 * Lf * Lf * Lf / (req.n0_deck_cm3 * Ld * Ld * Ld);
            u << " to n0 = " << req.n0_file_cm3 << " cm^-3, rescaled to the deck's " << req.n0_deck_cm3;
        }
    }

    // charge and mass of the species (uniform over its particles): used for the beam unless the deck sets them
    auto species_value = [&](const char* rec, double scale, double& val) {
        if (!sp.contains(rec)) return false;
        RecordComponent rc = sp[rec][RecordComponent::SCALAR];
        if (rc.constant()) { val = constant_value(rc) * rc.unitSI() * scale; return true; }
        std::vector<double> v(std::min(B, n));
        for (size_t i0 = 0; i0 < n; i0 += B) {   // per-particle record: must be uniform
            const size_t m = std::min(B, n - i0);
            load_block(S, rc, n, i0, m, v.data());
            if (i0 == 0) val = v[0] * scale;
            for (size_t i = 0; i < m; ++i)
                if (std::abs(v[i] * scale - val) > 1e-9 * std::abs(val))
                    throw std::runtime_error("openPMD import: the " + std::string(rec) + " of species '" + req.species +
                                             "' varies between its particles (not supported)");
        }
        return true;
    };
    out.has_charge = species_value("charge", si ? 1.0 / e_SI : 1.0, out.charge);
    out.has_mass = species_value("mass", si ? 1.0 / me_SI : 1.0, out.mass);
    if (req.info_only) return out;
    auto differs = [&](bool has, double file, double deck, const char* what, const char* key, const char* effect) {
        if (!has) { out.warnings.push_back(std::string("no ") + what + " record in the file: " + req.species + "." + key + " = " +
                                           std::to_string(deck) + " is used"); return; }
        if (std::abs(file - deck) <= 1e-6 * std::abs(deck)) return;
        std::ostringstream m;
        m << "the file's " << what << " (" << file << ") differs from " << req.species << "." << key << " = " << deck
          << ": the deck's value is used; " << effect;
        out.warnings.push_back(m.str());
    };
    differs(out.has_charge, out.charge, req.charge, "charge (e)", "charge",
            "positions, momenta and weights are taken unchanged");
    differs(out.has_mass, out.mass, req.mass, "mass (m_e)", "mass",
            "the velocities are kept (p / (m c) = gamma beta from the file's mass), so the physical momenta change");
    // momenta -> gamma beta with the file's mass if it differs from the beam's (the velocity is kept); with equal
    // masses (or no mass record) the beam's, so that a QUARZ file is read back exactly
    if (out.has_mass && std::abs(out.mass - req.mass) > 1e-6 * std::abs(req.mass))
        fP = si ? 1.0 / (out.mass * me_SI * c_SI) : 1.0 / out.mass;

    for (const char* c : {"x", "y", "z"}) {
        if (!pos.contains(c)) throw std::runtime_error(std::string("openPMD import: position has no component ") + c);
        if (!mom.contains(c)) throw std::runtime_error(std::string("openPMD import: momentum has no component ") + c);
    }
    const bool offset = sp.contains("positionOffset");
    const bool macro = attr_or(mom, "macroWeighted", 0) != 0;   // momenta per macro-particle
    const double pw = attr_or(mom, "weightingPower", 1.0);

    // particle stride recorded by QUARZ (particle_stride: every n-th particle, unchanged weights)
    out.stride = static_cast<long>(attr_or(sp, "quarz_particle_stride", 1.0));
    double fS = 1;
    if (out.stride > 1) {
        if (req.scale_stride) fS = static_cast<double>(out.stride);
        else out.warnings.push_back("the file holds every " + std::to_string(out.stride) +
                                    "-th particle with unchanged weights: the charge is 1/" + std::to_string(out.stride) +
                                    " of the original beam (<beam>.openpmd_scale_stride = 1 multiplies the weights)");
    }
    const bool spin = req.want_spin && sp.contains("spin");
    out.ncol = spin ? 10 : 7;

    // block by block: read, convert to code units, hand to the sink (which keeps the particles of its rank)
    const size_t nc = static_cast<size_t>(out.ncol);
    std::vector<double> col[13];   // x y z, offset x y z, px py pz, w, sx sy sz
    for (auto& c : col) c.resize(std::min(B, n));
    std::vector<double> rows(nc * std::min(B, n));
    const char* xyz[3] = {"x", "y", "z"};
    for (size_t i0 = 0; i0 < n; i0 += B) {
        const size_t m = std::min(B, n - i0);
        for (int c = 0; c < 3; ++c) load_block(S, pos[xyz[c]], n, i0, m, col[c].data());
        if (offset) {
            Record off = sp["positionOffset"];
            for (int c = 0; c < 3; ++c) {
                if (!off.contains(xyz[c])) throw std::runtime_error(std::string("openPMD import: positionOffset has no component ") + xyz[c]);
                load_block(S, off[xyz[c]], n, i0, m, col[3 + c].data());
                for (size_t i = 0; i < m; ++i) col[c][i] += col[3 + c][i];
            }
        }
        for (int c = 0; c < 3; ++c) load_block(S, mom[xyz[c]], n, i0, m, col[6 + c].data());
        load_block(S, wr[RecordComponent::SCALAR], n, i0, m, col[9].data());   // physical particles (SI) / code weight
        if (macro)   // macro-particle weighted momenta -> per physical particle
            for (int c = 0; c < 3; ++c)
                for (size_t i = 0; i < m; ++i) col[6 + c][i] /= std::pow(col[9][i], pw);
        if (spin)
            for (int c = 0; c < 3; ++c) {
                Record sr = sp["spin"];
                if (!sr.contains(xyz[c])) throw std::runtime_error(std::string("openPMD import: spin has no component ") + xyz[c]);
                load_block(S, sr[xyz[c]], n, i0, m, col[10 + c].data());
            }
        for (size_t i = 0; i < m; ++i) {
            double* q = &rows[nc * i];
            q[0] = col[0][i] * fL; q[1] = col[1][i] * fL;
            q[2] = col[6][i] * fP; q[3] = col[7][i] * fP; q[4] = col[8][i] * fP;
            q[5] = col[2][i] * fL + req.xi_shift;
            q[6] = col[9][i] * fW * fS;
            if (spin) { q[7] = col[10][i]; q[8] = col[11][i]; q[9] = col[12][i]; }
        }
        sink(rows.data(), m, out.ncol);
    }
    if (req.want_spin && !spin) out.warnings.push_back("no spin record: the spins are set by <beam>.spin");
    std::ostringstream s;
    s << n << " particles of species '" << req.species << "', iteration " << n_it << " of " << req.file << " (" << u.str()
      << (spin ? ", with spins" : "") << (fS > 1 ? ", weights x " + std::to_string(out.stride) : std::string()) << ")";
    out.summary = s.str();
    S.close();
    return out;
}

} // namespace quarz

#endif
