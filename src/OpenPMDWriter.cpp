#include "OpenPMDWriter.hpp"

#include "Beam.hpp"
#include "Config.hpp"
#include "Parallel.hpp"
#include "RadialGrid.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#ifdef QUARZ_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#ifdef QUARZ_USE_MPI
#include <mpi.h>
#endif
#endif

namespace quarz {

#ifndef QUARZ_USE_OPENPMD

struct OpenPMDWriter::Impl {};
OpenPMDWriter::OpenPMDWriter(const Config&, const std::string&, const RadialGrid&, const BeamGrid&, bool, double,
                             const std::string&) {
    throw std::runtime_error("output.format = openpmd: this QUARZ build has no openPMD support "
                             "(install openPMD-api and reconfigure with -DopenPMD_ROOT=...)");
}
OpenPMDWriter::~OpenPMDWriter() = default;
bool OpenPMDWriter::available() { return false; }
void OpenPMDWriter::write(int, double, const FieldTable*, const std::vector<const Beam*>&, int, int) {}
void OpenPMDWriter::set_dt(double dt) { (void)dt;  }

void OpenPMDWriter::close() {}
std::string OpenPMDWriter::description() const { return {}; }

#else

namespace {
// physical constants (CODATA 2018)
constexpr double c_SI = 299792458.0, e_SI = 1.602176634e-19, me_SI = 9.1093837015e-31, eps0_SI = 8.8541878128e-12;

using UD = std::map<openPMD::UnitDimension, double>;

#if defined(QUARZ_USE_MPI) && openPMD_HAVE_MPI
long long allreduce_sum(long long v) {
    long long s = 0;
    MPI_Allreduce(&v, &s, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return s;
}
long long exscan_sum(long long v) {
    long long s = 0;
    int rank = 0;
    MPI_Exscan(&v, &s, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0 ? 0 : s;
}
#else
long long allreduce_sum(long long v) { return v; }
long long exscan_sum(long long) { return 0; }
#endif
} // namespace

struct OpenPMDWriter::Impl {
    std::unique_ptr<openPMD::Series> series;
    std::string filename, backend;
    bool m1 = false, native = false, si = false;
    double n0 = 0;                       // m^-3
    double uL = 1, uT = 1, uE = 1, uB = 1, uPsi = 1, uN = 1, uRho = 1, uW = 1, uQ = 1, uM = 1, uP = 1;
    double dt = 0;
    BeamGrid box;
    std::vector<double> rnodes;          // native nodes
    std::vector<double> rout;            // output radii
    std::vector<int> jl;                 // interpolation: node index left of rout[i]
    std::vector<double> wr;              // weight of node jl+1
    bool closed = false;
    std::string options;
    void open();
};

bool OpenPMDWriter::available() { return true; }

OpenPMDWriter::OpenPMDWriter(const Config& cfg, const std::string& outdir, const RadialGrid& grid, const BeamGrid& box,
                             bool mode1, double dt, const std::string& prefix)
    : impl_(std::make_unique<Impl>()) {
    // group key with fallback to the main output key
    auto key = [&](const std::string& k) { return cfg.has(prefix + k) ? prefix + k : "output." + k; };
    Impl& I = *impl_;
    Comm& comm = Comm::world();
    I.m1 = mode1;
    I.box = box;
    I.dt = dt;
    I.rnodes.assign(grid.r.begin(), grid.r.end());

    // ---- units
    double n0cm3 = cfg.get_double("units.n0_cm3", 0.0);
    if (n0cm3 <= 0) n0cm3 = cfg.get_double("pusher.rr_n0_cm3", 0.0);
    I.si = n0cm3 > 0;
    if (I.si) {
        I.n0 = n0cm3 * 1e6;
        const double wp = std::sqrt(I.n0 * e_SI * e_SI / (eps0_SI * me_SI));
        I.uL = c_SI / wp;
        I.uT = 1.0 / wp;
        I.uE = me_SI * c_SI * wp / e_SI;
        I.uB = me_SI * wp / e_SI;
        I.uPsi = me_SI * c_SI * c_SI / e_SI;
        I.uN = I.n0;
        I.uRho = e_SI * I.n0;
        I.uW = I.n0 * I.uL * I.uL * I.uL;
        I.uQ = e_SI;
        I.uM = me_SI;
        I.uP = me_SI * c_SI;
    }

    // ---- radial output grid
    const std::string g = cfg.get_string(key("openpmd_grid"), "uniform");
    if (g != "uniform" && g != "native") throw std::runtime_error("output.openpmd_grid must be uniform or native");
    I.native = (g == "native");
    const double R = grid.R;
    if (I.native) {
        I.rout = I.rnodes;
        for (size_t i = 0; i < I.rout.size(); ++i) { I.jl.push_back(static_cast<int>(i)); I.wr.push_back(0.0); }
    } else {
        const double rmax = std::min(R, cfg.get_double(key("openpmd_rmax"), cfg.get_double(prefix + "rmax", R)));
        double dr;
        if (cfg.has(key("openpmd_dr"))) {
            dr = cfg.get_double(key("openpmd_dr"));
        } else {
            dr = grid.hmin();
            const int cap = 8192;
            if (rmax / dr + 1 > cap) dr = rmax / (cap - 1);
        }
        if (dr <= 0) throw std::runtime_error("output.openpmd_dr must be > 0");
        const int nr = static_cast<int>(std::floor(rmax / dr * (1 + 1e-12))) + 1;
        const int N = grid.N;
        int j = 0;
        for (int i = 0; i < nr; ++i) {
            const double r = std::min(i * dr, R);
            while (j < N - 1 && grid.r[j + 1] <= r) ++j;
            I.rout.push_back(i * dr);
            I.jl.push_back(j);
            I.wr.push_back(std::clamp((r - grid.r[j]) / (grid.r[j + 1] - grid.r[j]), 0.0, 1.0));
        }
    }

    // ---- series
    I.backend = cfg.get_string(key("openpmd_backend"), "h5");
    if (I.backend != "h5" && I.backend != "bp" && I.backend != "json")
        throw std::runtime_error("output.openpmd_backend must be h5, bp or json");
    const std::string pattern = cfg.get_string(key("openpmd_file"), "openpmd/data_%06T");
    I.filename = outdir + "/" + pattern + "." + I.backend;
    const auto dir = std::filesystem::path(I.filename).parent_path();
    if (comm.root() && !dir.empty()) std::filesystem::create_directories(dir);
    comm.barrier();
    I.options = cfg.get_string(key("openpmd_options"), "{}");
    // the series is created at the first write (collective): a file-based series without
    // iterations cannot be closed
}

void OpenPMDWriter::Impl::open() {
    Comm& comm = Comm::world();
    Impl& I = *this;
    const std::string& options = I.options;
#if defined(QUARZ_USE_MPI) && openPMD_HAVE_MPI
    if (comm.size() > 1)
        I.series = std::make_unique<openPMD::Series>(I.filename, openPMD::Access::CREATE, MPI_COMM_WORLD, options);
    else   // one rank: serial files (e.g. a plain .json instead of the per-rank .json.parallel layout)
        I.series = std::make_unique<openPMD::Series>(I.filename, openPMD::Access::CREATE, options);
#else
    if (comm.size() > 1)
        throw std::runtime_error("openPMD output with several MPI ranks needs openPMD-api built with MPI");
    I.series = std::make_unique<openPMD::Series>(I.filename, openPMD::Access::CREATE, options);
#endif
    openPMD::Series& S = *I.series;
    S.setSoftware("QUARZ", "1.0");
    S.setMeshesPath("fields/");
    S.setParticlesPath("particles/");
    S.setAttribute("quarz_units", I.si ? std::string("SI") : std::string("normalised: c/omega_p, 1/omega_p, n0, m_e c omega_p/e"));
    if (I.si) S.setAttribute("quarz_n0_SI", I.n0);
    S.setAttribute("quarz_coordinates", std::string("z = t - xi (lab frame); xi = t - z is the code's co-moving coordinate"));
}

OpenPMDWriter::~OpenPMDWriter() {
    try { close(); } catch (...) {}
}

void OpenPMDWriter::set_dt(double dt) { impl_->dt = dt;  }

void OpenPMDWriter::close() {
    if (impl_ && impl_->series && !impl_->closed) {
        impl_->series->close();
        impl_->closed = true;
    }
}

std::string OpenPMDWriter::description() const {
    const Impl& I = *impl_;
    const double mb = 8.0 * (I.m1 ? 3 : 1) * static_cast<double>(I.rout.size()) * I.box.nxi * 10 / 1048576.0;
    char b[640];
    std::snprintf(b, sizeof(b),
                  "openPMD output: %s  (%s units; fields %s, %zu radial points%s; about %.0f MB per field output)",
                  I.filename.c_str(), I.si ? "SI" : "normalised",
                  I.native ? "on the native non-uniform nodes" : "on a uniform radial grid", I.rout.size(),
                  I.native ? "" : (", dr = " + std::to_string(I.rout.size() > 1 ? I.rout[1] : 0.0)).c_str(), mb);
    std::string out = b;
    if (!I.native && mb > 4.0 * 8.0 * (I.m1 ? 3 : 1) * I.rnodes.size() * I.box.nxi * 10 / 1048576.0)
        out += "\n  note: the uniform openPMD grid is much finer than the native one; limit it with "
               "output.openpmd_rmax / output.openpmd_dr, or use output.openpmd_grid = native";
    return out;
}

void OpenPMDWriter::write(int step, double t, const FieldTable* T, const std::vector<const Beam*>& beams, int xs,
                          int pstride) {
    using namespace openPMD;
    Impl& I = *impl_;
    if (!I.series) I.open();
    Series& S = *I.series;
    Iteration it = S.iterations[static_cast<uint64_t>(step)];
    it.setTime(t);
    it.setDt(I.dt);
    it.setTimeUnitSI(I.uT);

    // every xs-th slice: global slices 0, xs, 2xs, ...; the local ones are rows[]
    std::vector<int> rows;
    for (int kl = 0; kl < I.box.nloc; ++kl)
        if ((I.box.k0 + kl) % xs == 0) rows.push_back(kl);
    const int K = (I.box.nxi - 1) / xs + 1, KL = static_cast<int>(rows.size()), k0 = (I.box.k0 + xs - 1) / xs;
    const double dxi = I.box.dxi * xs;
    const double zmin = t - (I.box.xi_min + (K - 1) * dxi);   // z = t - xi, increasing z = decreasing xi

    if (T) {
        const int nm = I.m1 ? 3 : 1;
        const size_t NR = I.rout.size();
        const Extent ext{static_cast<uint64_t>(nm), NR, static_cast<uint64_t>(K)};
        const Offset off{0, 0, static_cast<uint64_t>(K - k0 - KL)};
        const Extent cext{static_cast<uint64_t>(nm), NR, static_cast<uint64_t>(KL)};
        double dr = I.rout.size() > 1 ? I.rout[1] - I.rout[0] : 1.0;
        if (I.native)   // informative on a rectilinear axis: the smallest spacing (docs/openpmd_rectilinear.md)
            for (size_t i = 1; i < I.rout.size(); ++i) dr = std::min(dr, I.rout[i] - I.rout[i - 1]);
#ifdef QUARZ_OPENPMD_GRID_COORDINATES
        std::string rpath;
        if (I.native) {
            // radial node positions: one coordinate data set per iteration, <basePath>gridCoordinates/r,
            // referenced by every mesh (collective with parallel HDF5; rank 0 writes the values)
            auto rc = it.gridCoordinates["r"];
            rc.resetDataset(Dataset(Datatype::DOUBLE, {static_cast<uint64_t>(I.rout.size())}));
            if (I.box.k0 == 0) {
                std::shared_ptr<double> buf(new double[I.rout.size()], std::default_delete<double[]>());
                std::copy(I.rout.begin(), I.rout.end(), buf.get());
                rc.storeChunk(buf, {0}, {static_cast<uint64_t>(I.rout.size())});
            }
            rpath = it.gridCoordinatesPath("r");
        }
#endif

        auto setup_mesh = [&](Mesh& m, const UD& ud) {
            if (I.native) {
                // rectilinear mesh (proposed openPMD extension, docs/openpmd_rectilinear.md, shared with
                // GEM-PIC-RZ): radial node positions in gridCoordinates_r, the xi axis stays regular.
                // geometry "other" (intended geometry first in geometryParameters) until the extension
                // is part of the standard, so that readers of regular grids skip these meshes.
                m.setGeometry("other");
                m.setGeometryParameters(std::string("thetaMode;m=") + (I.m1 ? "2" : "1") + ";imag=+");
#ifdef QUARZ_OPENPMD_GRID_COORDINATES
                m.setGridCoordinates({rpath, ""});   // openPMD 2.0 form: coordinate data set
#else
                m.setAttribute("gridType", std::string("rectilinear"));   // interim form: attribute
                m.setAttribute("gridCoordinates_r", I.rout);
#endif
                m.setAttribute("quarz_r_nodes", I.rout);   // deprecated, for older QUARZ readers
            } else {
                m.setGeometry(Mesh::Geometry::thetaMode);
                m.setGeometryParameters(std::string("m=") + (I.m1 ? "2" : "1") + ";imag=+");
            }
            m.setDataOrder(Mesh::DataOrder::C);
            m.setAxisLabels({"r", "z"});
            m.setGridSpacing(std::vector<double>{dr, dxi});
            m.setGridGlobalOffset({0.0, zmin});
            m.setGridUnitSI(I.uL);
            m.setUnitDimension(ud);
            m.setTimeOffset(0.0);
        };
        // one record component: modes of `name` resampled to rout, xi reversed
        auto store = [&](MeshRecordComponent rc, const std::string& name, double unit) {
            rc.setPosition(std::vector<double>{0.0, 0.0});
            rc.setUnitSI(unit);
            rc.resetDataset(Dataset(Datatype::DOUBLE, ext));
            const std::function<double(int, int)>* f[3] = {T->find(name), nullptr, nullptr};
            if (I.m1) { f[1] = T->find(name + "_c"); f[2] = T->find(name + "_s"); }
            for (int m = 0; m < nm; ++m)
                if (!f[m]) throw std::runtime_error("openPMD: missing field component " + name);
            if (KL > 0) {
                std::shared_ptr<double> buf(new double[static_cast<size_t>(nm) * NR * KL], std::default_delete<double[]>());
                double* p = buf.get();
                for (int m = 0; m < nm; ++m)
                    for (size_t i = 0; i < NR; ++i) {
                        const int j = I.jl[i];
                        const double a = I.wr[i];
                        for (int kl = 0; kl < KL; ++kl) {
                            const int k = rows[kl];
                            const double v = a > 0 ? (1 - a) * (*f[m])(k, j) + a * (*f[m])(k, j + 1) : (*f[m])(k, j);
                            p[(static_cast<size_t>(m) * NR + i) * KL + (KL - 1 - kl)] = v;
                        }
                    }
                rc.storeChunk(buf, off, cext);
            }
            S.flush();   // collective with parallel HDF5: every rank calls it the same number of times
        };

        const UD udE{{UnitDimension::L, 1}, {UnitDimension::M, 1}, {UnitDimension::T, -3}, {UnitDimension::I, -1}};
        const UD udB{{UnitDimension::M, 1}, {UnitDimension::T, -2}, {UnitDimension::I, -1}};
        const UD udPsi{{UnitDimension::L, 2}, {UnitDimension::M, 1}, {UnitDimension::T, -3}, {UnitDimension::I, -1}};
        const UD udN{{UnitDimension::L, -3}};
        const UD udRho{{UnitDimension::L, -3}, {UnitDimension::T, 1}, {UnitDimension::I, 1}};
        // vector records with the selected components only
        auto vector = [&](const std::string& rec, const char* const comp[3], const char* const name[3], const UD& ud,
                          double unit) {
            bool any = false;
            for (int c = 0; c < 3; ++c) any = any || T->find(name[c]);
            if (!any) return;
            Mesh m = it.meshes[rec];
            setup_mesh(m, ud);
            for (int c = 0; c < 3; ++c)
                if (T->find(name[c])) store(m[comp[c]], name[c], unit);
        };
        const char* const rtz[3] = {"r", "t", "z"};
        const char* const en[3] = {"er", "eth", "ez"};
        const char* const bn[3] = {"br", "bth", "bz"};
        vector("E", rtz, en, udE, I.uE);
        vector("B", rtz, bn, udB, I.uB);
        auto scalar = [&](const std::string& rec, const std::string& name, const UD& ud, double unit) {
            if (!T->find(name)) return;
            Mesh m = it.meshes[rec];
            setup_mesh(m, ud);
            store(m[MeshRecordComponent::SCALAR], name, unit);
        };
        scalar("psi", "psi", udPsi, I.uPsi);
        scalar("n_e", "ne", udN, I.uN);
        scalar("n_i", "ni", udN, I.uN);
        scalar("rho_beam", "rhob", udRho, I.uRho);
        for (const auto& c : T->comps)   // ion charge densities z n of the ionizable species
            if (c.first.rfind("nz_", 0) == 0 && c.first.size() > 3 && c.first.substr(c.first.size() - 2) != "_c" &&
                c.first.substr(c.first.size() - 2) != "_s")
                scalar("n_z_" + c.first.substr(3), c.first, udN, I.uN);
        if (T->find("a_re")) {   // laser envelope a^ (normalised vector potential, dimensionless)
            Mesh L = it.meshes["laserEnvelope"];
            setup_mesh(L, UD{});
            L.setAttribute("quarz_convention", std::string("a = Re[(re + i im) exp(-i k0 xi)], xi = t - z"));
            store(L["re"], "a_re", 1.0);
            store(L["im"], "a_im", 1.0);
        }
    }

    for (const Beam* bp : beams) {
        std::vector<double> p = bp->packed();
        const size_t nc = static_cast<size_t>(bp->ncomp());
        if (pstride > 1) {
            std::vector<double> q;
            for (size_t i = 0; i < p.size() / nc; i += static_cast<size_t>(pstride))
                q.insert(q.end(), p.begin() + nc * i, p.begin() + nc * i + nc);
            p.swap(q);
        }
        const long long nloc = static_cast<long long>(p.size() / nc);
        const long long ntot = allreduce_sum(nloc);
        if (ntot == 0) continue;   // same decision on every rank
        const long long poff = exscan_sum(nloc);
        ParticleSpecies sp = it.particles[bp->name()];
        const Dataset ds(Datatype::DOUBLE, {static_cast<uint64_t>(ntot)});
        const Offset o{static_cast<uint64_t>(poff)};
        const Extent e{static_cast<uint64_t>(nloc)};
        auto column = [&](int c, double add, double sign) {
            std::shared_ptr<double> b(new double[std::max<long long>(nloc, 1)], std::default_delete<double[]>());
            for (long long i = 0; i < nloc; ++i) b.get()[i] = add + sign * p[nc * static_cast<size_t>(i) + c];
            return b;
        };
        auto put = [&](RecordComponent rc, std::shared_ptr<double> b, double unit) {
            rc.resetDataset(ds);
            rc.setUnitSI(unit);
            if (nloc > 0) rc.storeChunk(b, o, e);
        };
        auto weighted = [](Record& r, uint32_t macro, double power) {
            r.setAttribute("macroWeighted", macro);
            r.setAttribute("weightingPower", power);
        };
        const double mass = bp->mass();
        {
            Record pos = sp["position"];
            pos.setUnitDimension(UD{{UnitDimension::L, 1}});
            pos.setTimeOffset(0.0);
            weighted(pos, 0, 0.0);
            put(pos["x"], column(0, 0.0, 1.0), I.uL);
            put(pos["y"], column(1, 0.0, 1.0), I.uL);
            put(pos["z"], column(5, t, -1.0), I.uL);   // z = t - xi
        }
        {
            Record po = sp["positionOffset"];
            po.setUnitDimension(UD{{UnitDimension::L, 1}});
            po.setTimeOffset(0.0);
            weighted(po, 0, 0.0);
            for (const char* c : {"x", "y", "z"}) {
                po[c].resetDataset(ds);
                po[c].setUnitSI(I.uL);
                po[c].makeConstant(0.0);
            }
        }
        {
            Record mom = sp["momentum"];
            mom.setUnitDimension(UD{{UnitDimension::L, 1}, {UnitDimension::M, 1}, {UnitDimension::T, -1}});
            mom.setTimeOffset(0.0);
            weighted(mom, 0, 1.0);
            // code momenta are in m c of the species; stored per physical particle in m_e c
            put(mom["x"], column(2, 0.0, mass), I.uP);
            put(mom["y"], column(3, 0.0, mass), I.uP);
            put(mom["z"], column(4, 0.0, mass), I.uP);
        }
        {
            Record w = sp["weighting"];
            w.setUnitDimension(UD{});
            w.setTimeOffset(0.0);
            weighted(w, 1, 1.0);
            put(w[RecordComponent::SCALAR], column(6, 0.0, 1.0), I.uW);
        }
        if (bp->spin()) {   // unit spin vector (T-BMT), not an openPMD base record
            Record sr = sp["spin"];
            sr.setUnitDimension(UD{});
            sr.setTimeOffset(0.0);
            weighted(sr, 0, 0.0);
            sr.setAttribute("quarz_anomalous_moment", static_cast<double>(bp->anomalous()));
            put(sr["x"], column(7, 0.0, 1.0), 1.0);
            put(sr["y"], column(8, 0.0, 1.0), 1.0);
            put(sr["z"], column(9, 0.0, 1.0), 1.0);
        }
        {
            Record q = sp["charge"];
            q.setUnitDimension(UD{{UnitDimension::T, 1}, {UnitDimension::I, 1}});
            q.setTimeOffset(0.0);
            weighted(q, 0, 1.0);
            q[RecordComponent::SCALAR].resetDataset(ds);
            q[RecordComponent::SCALAR].setUnitSI(I.uQ);
            q[RecordComponent::SCALAR].makeConstant(static_cast<double>(bp->charge()));
        }
        {
            Record m = sp["mass"];
            m.setUnitDimension(UD{{UnitDimension::M, 1}});
            m.setTimeOffset(0.0);
            weighted(m, 0, 1.0);
            m[RecordComponent::SCALAR].resetDataset(ds);
            m[RecordComponent::SCALAR].setUnitSI(I.uM);
            m[RecordComponent::SCALAR].makeConstant(static_cast<double>(mass));
        }
        S.flush();
    }
    it.close();
}

#endif

} // namespace quarz
