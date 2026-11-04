// Beam import from openPMD particle data (<beam>.profile = openpmd) through openPMD-api.
// Without openPMD-api in the build, read_openpmd_beam() throws.
//
// The file can come from QUARZ or from another code that writes particles in the co-moving
// frame: xi = z + positionOffset_z + xi_shift (code units). Units follow the openPMD metadata:
//   - SI files (value x unitSI is SI: the openPMD standard; QUARZ files say which in quarz_units;
//     other codes' normalised files must be declared, units = "normalised"): converted with the deck's plasma density
//     (units.n0_cm3, required): lengths / (c/omega_p), momentum per physical particle / (m c),
//     weighting (physical particles) / (n0 (c/omega_p)^3);
//   - normalised files (as QUARZ writes them without units.n0_cm3): lengths in c/omega_p,
//     momentum in m_e c per physical particle, weighting in n0 (c/omega_p)^3; if they were normalised
//     to another density (n0_file), lengths and weights are rescaled to the deck's.
// Momenta stored per macro-particle (macroWeighted = 1) are divided by weighting^weightingPower.
// Charge and mass: the beam takes the file's (constant or uniform per-particle records) unless the deck
// sets <beam>.charge / <beam>.mass; a deck value that differs from the file's gives a warning (momenta keep
// their physical value). info_only: only the species' charge and mass (for the Beam constructor).
#pragma once

#include <string>
#include <vector>

namespace quarz {

struct OpenPMDBeamRequest {
    std::string file;            // a file of the series (e.g. out/openpmd/data_000200.h5) or a pattern with %T
    long iteration = -1;         // -1: from the file name, else the last iteration of the series
    std::string species;         // particle species in the file
    double xi_shift = 0;         // added to the co-moving z (code units)
    double n0_deck_cm3 = 0;      // units.n0_cm3 of the deck (0: not given)
    double n0_file_cm3 = 0;      // density a normalised file refers to (0: the deck's)
    std::string units = "auto";  // auto (QUARZ attribute quarz_units, else SI) | si | normalised
    double charge = -1, mass = 1;   // the beam's, in e and m_e (the deck's or, without them, the file's)
    bool info_only = false;      // only read the species' charge and mass
    bool want_spin = false;      // the beam has <beam>.spin: read the spin record if present
    bool scale_stride = false;   // multiply the weights by the recorded particle stride
};

struct OpenPMDBeamData {
    std::vector<double> p;       // ncol doubles per particle: x y px py pz xi w (+ sx sy sz); xi = user xi
    int ncol = 7;
    long iteration = 0;
    long stride = 1;             // quarz_particle_stride of the species (1 if not recorded)
    bool has_charge = false, has_mass = false;   // the species has charge / mass records
    double charge = 0, mass = 0;                  // their values, in e and m_e
    std::string summary;         // one line for the log
    std::vector<std::string> warnings;
};

OpenPMDBeamData read_openpmd_beam(const OpenPMDBeamRequest& req);

} // namespace quarz
