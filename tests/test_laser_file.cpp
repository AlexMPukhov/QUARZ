// LASY laser files (src/LaserFile.*): the test writes LASY files (layout of lasy 0.6 / 0.7: openPMD 1.1, mesh
// laserEnvelope, complex compound (r, i) as h5py writes complex128) and compares the snapshot a^(r, xi) at t = 0 with an
// independent reference: the spectrally resolved paraxial beam, every frequency omega of the pulse with its own
// q(omega) = dz - i z_R(omega), z_R = omega w0^2 / 2, propagated over dz(xi) = -(file_z + xi - xi0) from its focus at the
// file plane (Gaussian: q0/q e^{i omega r^2 / (2q)}; LG_01: (q0/q)^2 r e^{i omega r^2 / (2q)}), summed numerically.
// Paraxial error ~ 1/(k0 w0)^2 = 1e-3.
#include "LaserFile.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifdef QUARZ_USE_HDF5
#include <hdf5.h>
#endif

using namespace quarz;
using cd = std::complex<double>;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

#ifdef QUARZ_USE_HDF5
namespace {
const double n0 = 1e17, wp = 5.64146e4 * std::sqrt(n0), c_si = 299792458.0, kp = wp / c_si;
const double Ep = 9.1093837015e-31 * c_si * wp / 1.602176634e-19;
const double PI_ = 3.14159265358979323846;

void attr_str(hid_t o, const char* n, const std::string& v) {
    const hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, v.size());
    H5Tset_strpad(t, H5T_STR_NULLPAD);   // as h5py (numpy S)
    const hid_t s = H5Screate(H5S_SCALAR);
    const hid_t a = H5Acreate2(o, n, t, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, t, v.data());
    H5Aclose(a); H5Sclose(s); H5Tclose(t);
}
void attr_strs(hid_t o, const char* n, const std::vector<std::string>& v) {   // fixed-length array, as numpy S2
    size_t sz = 1;
    for (auto& x : v) sz = std::max(sz, x.size());
    std::vector<char> buf(sz * v.size(), 0);
    for (size_t k = 0; k < v.size(); ++k) std::copy(v[k].begin(), v[k].end(), buf.begin() + long(k * sz));
    const hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, sz);
    H5Tset_strpad(t, H5T_STR_NULLPAD);
    const hsize_t d = v.size();
    const hid_t s = H5Screate_simple(1, &d, nullptr);
    const hid_t a = H5Acreate2(o, n, t, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, t, buf.data());
    H5Aclose(a); H5Sclose(s); H5Tclose(t);
}
void attr_dbl(hid_t o, const char* n, const std::vector<double>& v) {
    const hsize_t d = v.size();
    const hid_t s = d == 1 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &d, nullptr);
    const hid_t a = H5Acreate2(o, n, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, v.data());
    H5Aclose(a); H5Sclose(s);
}

// env(comp or 0, u, a, b) in QUARZ units: E [m c omega_p / e] or a; thetaMode: (comp, u, r = b); cartesian: (u, y = a, x = b)
struct LasyFile {
    std::string path, geometry = "thetaMode", field = "normalized_vector_potential";
    double omega = 10;                       // omega_L / omega_p
    std::vector<double> pol = {1, 0, 0, 0};  // (re, im) of p_x, p_y
    int nt = 161, n1 = 1, n2 = 121, ncomp = 1;
    double t0 = -8, dt = 0.1, a0 = 0, da = 1, b0 = 0, db = 0.1;
    std::function<cd(int, double, double, double)> f;
    void write() const {
        const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        attr_str(file, "openPMD", "1.1.0"); attr_str(file, "basePath", "/data/%T/"); attr_str(file, "meshesPath", "meshes/");
        attr_str(file, "software", "lasy"); attr_str(file, "softwareVersion", "0.6.2");
        const hid_t lc = H5Pcreate(H5P_LINK_CREATE);
        H5Pset_create_intermediate_group(lc, 1);
        const hid_t g = H5Gcreate2(file, "data/0/meshes", lc, H5P_DEFAULT, H5P_DEFAULT);
        const bool cart = geometry == "cartesian";
        const hsize_t dims[3] = {hsize_t(cart ? nt : ncomp), hsize_t(cart ? n1 : nt), hsize_t(n2)};
        std::vector<double> v;
        const double unit = field == "electric_field" ? Ep : 1.0;
        for (hsize_t x = 0; x < dims[0]; ++x)
            for (hsize_t y = 0; y < dims[1]; ++y)
                for (hsize_t z = 0; z < dims[2]; ++z) {
                    const cd e = cart ? unit * f(0, t0 + x * dt, a0 + y * da, b0 + z * db) : unit * f(int(x), t0 + y * dt, 0, b0 + z * db);
                    v.push_back(e.real()); v.push_back(e.imag());
                }
        const hid_t ct = H5Tcreate(H5T_COMPOUND, 16);
        H5Tinsert(ct, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(ct, "i", 8, H5T_NATIVE_DOUBLE);
        const hid_t sp = H5Screate_simple(3, dims, nullptr);
        const hid_t ds = H5Dcreate2(g, "laserEnvelope", ct, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, ct, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        attr_str(ds, "geometry", geometry); attr_str(ds, "dataOrder", "C"); attr_str(ds, "envelopeField", field);
        attr_strs(ds, "axisLabels", cart ? std::vector<std::string>{"t", "y", "x"} : std::vector<std::string>{"t", "r"});
        const double T = 1 / wp, L = 1 / kp;
        attr_dbl(ds, "gridSpacing", cart ? std::vector<double>{dt * T, da * L, db * L} : std::vector<double>{dt * T, db * L});
        attr_dbl(ds, "gridGlobalOffset", cart ? std::vector<double>{t0 * T, a0 * L, b0 * L} : std::vector<double>{t0 * T, b0 * L});
        attr_dbl(ds, "position", cart ? std::vector<double>{0, 0, 0} : std::vector<double>{0, 0});
        attr_dbl(ds, "gridUnitSI", {1.0}); attr_dbl(ds, "unitSI", {1.0});
        attr_dbl(ds, "angularFrequency", {omega * wp});
        if (pol[1] == 0 && pol[3] == 0) attr_dbl(ds, "polarization", {pol[0], pol[2]});
        else {
            const hsize_t two = 2;
            const hid_t as = H5Screate_simple(1, &two, nullptr);
            const hid_t a = H5Acreate2(ds, "polarization", ct, as, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, ct, pol.data());
            H5Aclose(a); H5Sclose(as);
        }
        H5Dclose(ds); H5Sclose(sp); H5Tclose(ct); H5Gclose(g); H5Pclose(lc); H5Fclose(file);
    }
};

const double W = 3, Lp = 2, k0 = 10, xi0 = 10, zR = 0.5 * k0 * W * W;
double env_t(double u) { return std::exp(-u * u / (Lp * Lp)); }

// the spectral paraxial reference a^(r, xi) (l = 0 or 1, at its focus at the file plane, file_z ahead of the centre)
cd reference(int l, double fz, double r, double xi) {
    const double dz = -(fz + xi - xi0), u = xi - xi0;
    // e^{-u^2/L^2} e^{-i k0 u} = int dw/2pi G(w) e^{-i w u}, G = sqrt(pi) L e^{-(w - k0)^2 L^2 / 4}
    const int nw = 601;
    const double sw = 2 / Lp, wlo = k0 - 7 * sw, dw = 14 * sw / (nw - 1);
    cd s = 0;
    for (int q = 0; q < nw; ++q) {
        const double w = wlo + q * dw, zr = 0.5 * w * W * W;
        const cd q0(0, -zr), qq(dz, -zr), g = q0 / qq;
        const cd mono = (l ? g * g * (std::sqrt(2.0) * r / W) : g) * std::exp(cd(0, w * r * r) / (2.0 * qq));
        s += std::sqrt(PI_) * Lp * std::exp(-(w - k0) * (w - k0) * Lp * Lp / 4) * mono * std::polar(1.0, -w * u) * dw / (2 * PI_);
    }
    return std::polar(1.0, k0 * xi) * s;
}
} // namespace

static void run() {
    std::vector<double> r, xi;
    for (int j = 0; j <= 150; ++j) r.push_back(0.04 * j + 0.0004 * j * j);   // non-uniform, up to 15
    for (int k = 0; k <= 200; ++k) xi.push_back(0.1 * k);                    // xi0 +- 10
    LasyFile A;                                                             // Gaussian at its focus, vector potential
    A.path = "test_laser_file_a.h5";
    A.f = [](int, double u, double, double b) { return cd(env_t(u) * std::exp(-b * b / (W * W)), 0); };
    A.write();
    LasyFile E = A;                                                         // the same pulse as an electric field
    E.path = "test_laser_file_e.h5"; E.field = "electric_field";
    E.f = [](int, double u, double, double b) {
        const double a = env_t(u) * std::exp(-b * b / (W * W)), dadu = -2 * u / (Lp * Lp) * a;
        return cd(0, k0) * a - dadu;                                        // E = i omega_L a - da/du
    };
    E.write();
    LasyFile V = A;                                                         // LG_01 e^{+i theta}: C_1 = f, S_1 = i f
    V.path = "test_laser_file_lg.h5"; V.ncomp = 3;
    V.f = [](int c, double u, double, double b) {
        const double f = env_t(u) * std::sqrt(2.0) * b / W * std::exp(-b * b / (W * W));
        return c == 1 ? cd(f, 0) : (c == 2 ? cd(0, f) : cd(0));
    };
    V.write();
    LasyFile C;                                                             // cartesian, circular (1, i)/sqrt 2, at the focus
    C.path = "test_laser_file_c.h5"; C.geometry = "cartesian"; C.pol = {std::sqrt(0.5), 0, 0, std::sqrt(0.5)};
    C.n1 = C.n2 = 121; C.a0 = C.b0 = -12; C.da = C.db = 0.2;
    C.f = [](int, double u, double a, double b) { return cd(env_t(u) * std::exp(-(a * a + b * b) / (W * W)), 0); };
    C.write();

    std::mt19937 rng(3);
    std::uniform_int_distribution<int> pj(0, int(r.size()) - 1), pk(0, int(xi.size()) - 1);
    auto compare = [&](const LaserFileResult& R, int l, double fz, double scale) {
        double worst = 0, peak = 0;
        for (int n = 0; n < 1500; ++n) {
            const int j = pj(rng), k = pk(rng);
            const cd ref = scale * reference(l, fz, r[size_t(j)], xi[size_t(k)]);
            peak = std::max(peak, std::abs(ref));
            const double er = std::abs(R.a[size_t(k) * r.size() + j] - ref);
            if (er > worst && std::getenv("QUARZ_TEST_VERBOSE")) std::printf("    worst so far %.3e at xi %.2f r %.3f: code %.5f%+.5fi ref %.5f%+.5fi\n", er, xi[size_t(k)], r[size_t(j)], R.a[size_t(k) * r.size() + j].real(), R.a[size_t(k) * r.size() + j].imag(), ref.real(), ref.imag());
            worst = std::max(worst, er);
        }
        return worst / peak;
    };
    LaserFileInput in;
    in.n0_cm3 = n0; in.xi0 = xi0; in.t_center = 0;
    char b[200];
    for (double fz : {0.0, zR}) {
        in.path = A.path; in.file_z = fz;
        const LaserFileResult Ra = laser_from_file(in, r, xi);
        const double ea = compare(Ra, 0, fz, 1);
        in.path = E.path;
        const LaserFileResult Re = laser_from_file(in, r, xi);
        const double ee = compare(Re, 0, fz, 1);
        std::snprintf(b, sizeof b, "Gaussian, file plane %g ahead (z_R = %g): vector-potential file %.1e, E-field file %.1e of the peak",
                      fz, zR, ea, ee);
        check(ea < 3e-3 && ee < 3e-3 && std::abs(Ra.k0 - k0) < 1e-9 && !Ra.circular, b);
    }
    in.path = V.path; in.file_z = zR; in.l = 1;
    const LaserFileResult Rv = laser_from_file(in, r, xi);
    const double ev = compare(Rv, 1, zR, 1);
    std::snprintf(b, sizeof b, "LG_01 vortex (laser.l = 1), file plane z_R ahead: %.1e of the peak, other modes %.1e", ev, Rv.mode_loss);
    check(ev < 3e-3 && Rv.mode_loss < 1e-12, b);
    in.path = C.path; in.l = 0;
    const LaserFileResult Rc = laser_from_file(in, r, xi);
    const double ec = compare(Rc, 0, zR, std::sqrt(0.5));
    std::snprintf(b, sizeof b, "cartesian file, circular polarization (a^ = a_x amplitude = |a| / sqrt 2): %.1e of the peak", ec);
    check(ec < 5e-3 && Rc.circular, b);
    in.path = A.path; in.l = 0; in.file_z = 0; in.a0 = 2.5;
    const LaserFileResult Rs = laser_from_file(in, r, xi);
    std::snprintf(b, sizeof b, "laser.file_a0 = 2.5 rescales the peak at the file plane: %.6f", Rs.peak_file);
    check(std::abs(Rs.peak_file - 2.5) < 1e-9, b);
}
#endif

int main() {
#ifdef QUARZ_USE_HDF5
    std::printf("LASY laser files (src/LaserFile.*):\n");
    run();
#else
    std::printf("LASY laser files: skipped (built without HDF5)\n");
#endif
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
