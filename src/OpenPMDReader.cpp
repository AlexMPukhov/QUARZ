// Beam import from openPMD particle data: see OpenPMDReader.hpp.
#include "OpenPMDReader.hpp"

#include <stdexcept>

#ifndef QUARZ_USE_OPENPMD

namespace quarz {
OpenPMDBeamData read_openpmd_beam(const OpenPMDBeamRequest& req) {
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

// values of a record component times its unitSI (constant components included; double or float data)
std::vector<double> load(openPMD::Series& S, openPMD::RecordComponent rc, size_t n) {
    std::vector<double> v(n);
    const double u = rc.unitSI();
    if (rc.constant()) {
        const auto a = rc.getAttribute("value");
        double c = 0;
        if (auto d = a.getOptional<double>()) c = *d;
        else if (auto f = a.getOptional<float>()) c = *f;
        else if (auto i = a.getOptional<int64_t>()) c = static_cast<double>(*i);
        else throw std::runtime_error("openPMD import: unsupported type of a constant record component");
        std::fill(v.begin(), v.end(), c * u);
        return v;
    }
    if (rc.getExtent().empty() || rc.getExtent()[0] != n)
        throw std::runtime_error("openPMD import: record components of different lengths");
    if (rc.getDatatype() == openPMD::Datatype::FLOAT) {
        auto d = rc.loadChunk<float>();
        S.flush();
        for (size_t i = 0; i < n; ++i) v[i] = d.get()[i] * u;
    } else {
        auto d = rc.loadChunk<double>();
        S.flush();
        for (size_t i = 0; i < n; ++i) v[i] = d.get()[i] * u;
    }
    return v;
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

OpenPMDBeamData read_openpmd_beam(const OpenPMDBeamRequest& req) {
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
        const auto v = load(S, sp[rec][RecordComponent::SCALAR], n);
        val = v[0] * scale;
        for (double x : v)
            if (std::abs(x * scale - val) > 1e-9 * std::abs(val))
                throw std::runtime_error("openPMD import: the " + std::string(rec) + " of species '" + req.species +
                                         "' varies between its particles (not supported)");
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
            "the momenta keep their physical value (p_file / (m_deck c)), i.e. gamma and the velocity change");

    auto comp3 = [&](Record r, const char* what) {
        std::vector<std::vector<double>> v;
        for (const char* c : {"x", "y", "z"}) {
            if (!r.contains(c)) throw std::runtime_error(std::string("openPMD import: ") + what + " has no component " + c);
            v.push_back(load(S, r[c], n));
        }
        return v;
    };
    auto X = comp3(pos, "position");
    if (sp.contains("positionOffset")) {
        const auto O = comp3(sp["positionOffset"], "positionOffset");
        for (int c = 0; c < 3; ++c)
            for (size_t i = 0; i < n; ++i) X[c][i] += O[c][i];
    }
    auto P = comp3(mom, "momentum");
    std::vector<double> W = load(S, wr[RecordComponent::SCALAR], n);   // physical particles (SI) / code weight (normalised)

    // macro-particle weighted momenta -> per physical particle
    if (attr_or(mom, "macroWeighted", 0) != 0) {
        const double pw = attr_or(mom, "weightingPower", 1.0);
        for (int c = 0; c < 3; ++c)
            for (size_t i = 0; i < n; ++i) P[c][i] /= std::pow(W[i], pw);
    }

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
    std::vector<std::vector<double>> Sv;
    if (spin) Sv = comp3(sp["spin"], "spin");
    out.ncol = spin ? 10 : 7;
    out.p.resize(static_cast<size_t>(out.ncol) * n);
    for (size_t i = 0; i < n; ++i) {
        double* q = &out.p[static_cast<size_t>(out.ncol) * i];
        q[0] = X[0][i] * fL; q[1] = X[1][i] * fL;
        q[2] = P[0][i] * fP; q[3] = P[1][i] * fP; q[4] = P[2][i] * fP;
        q[5] = X[2][i] * fL + req.xi_shift;
        q[6] = W[i] * fW * fS;
        if (spin) { q[7] = Sv[0][i]; q[8] = Sv[1][i]; q[9] = Sv[2][i]; }
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
