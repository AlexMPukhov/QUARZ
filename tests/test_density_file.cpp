// Density profiles from files (plasma.density_file, src/DensityFile.*, format docs/density_file.md).
//   1. the loaded plasma (positions and weights of the macro-particles) with the profile from a file equals the
//      plasma loaded from the built-in tables (plasma.z_table, plasma.radial_table) or the parsed function
//      plasma.density(x,y,z): geometry z and rz, columns and grid layout (both orders), normalized and physical
//      units, a transverse function in the header, a further factor density(x,y,z), rmax, file_z, file_outside
//   2. the reader alone for the geometries of the 3D codes (xz, xyz): a trilinear function is reproduced exactly;
//      columns layout in shuffled order = grid layout
//   3. openPMD meshes (build with HDF5): regular thetaMode mesh, rectilinear 3D mesh in SI = the text files
//   4. files and combinations that must be rejected; the lines and warnings of the run summary
#include "Config.hpp"
#include "DensityFile.hpp"
#include "PlasmaSpecies.hpp"
#include "Profiles.hpp"
#include "RadialGrid.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#ifdef QUARZ_USE_HDF5
#include <hdf5.h>
#endif

using namespace quarz;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
static void write(const std::string& name, const std::string& text) {
    std::ofstream f(name);
    f << text;
}
static std::string num(double x) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", x);
    return b;
}

// positions and weights of a loaded species at the lab positions zs
struct Loaded { std::vector<double> x, y, w; };
static Loaded load(const Config& cfg, const RadialGrid& g, bool m1, const std::vector<double>& zs) {
    DensityProfile prof(cfg, "plasma");
    PlasmaSpecies e(cfg, "e", g, prof, m1);
    Loaded L;
    for (double z : zs) {
        e.load(z, 7);
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace(), e.x_);
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace(), e.y_);
        auto hw = Kokkos::create_mirror_view_and_copy(HostSpace(), e.w_);
        for (int i = 0; i < e.num_particles(); ++i) { L.x.push_back(hx(i)); L.y.push_back(hy(i)); L.w.push_back(hw(i)); }
    }
    return L;
}
// largest difference of the weights relative to the largest weight; 1e300 if the particles differ
static double diff(const Loaded& a, const Loaded& b) {
    if (a.w.size() != b.w.size() || a.w.empty()) return 1e300;
    double d = 0, m = 0;
    for (size_t i = 0; i < a.w.size(); ++i) {
        if (a.x[i] != b.x[i] || a.y[i] != b.y[i]) return 1e300;
        d = std::max(d, std::fabs(a.w[i] - b.w[i]));
        m = std::max(m, std::fabs(a.w[i]));
    }
    return d / m;
}
static Config base() {
    Config cfg;
    cfg.set("e.ppc", "3");
    return cfg;
}
static std::string error_of(const std::function<void()>& f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "";
}
static bool rejects(const std::string& file_text, const std::string& expect) {
    write("df_bad.txt", file_text);
    Config cfg = base();
    cfg.set("plasma.density_file", "df_bad.txt");
    const std::string e = error_of([&] { DensityProfile p(cfg, "plasma"); });
    if (e.find(expect) == std::string::npos) std::printf("    expected '%s', got: %s\n", expect.c_str(), e.empty() ? "(accepted)" : e.c_str());
    return e.find(expect) != std::string::npos;
}

#ifdef QUARZ_USE_HDF5
namespace {
void attr_str(hid_t o, const char* n, const std::string& v) {
    const hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, v.size());
    H5Tset_strpad(t, H5T_STR_NULLPAD);
    const hid_t s = H5Screate(H5S_SCALAR);
    const hid_t a = H5Acreate2(o, n, t, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, t, v.data());
    H5Aclose(a); H5Sclose(s); H5Tclose(t);
}
void attr_strs(hid_t o, const char* n, const std::vector<std::string>& v) {
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
    const hid_t s = H5Screate_simple(1, &d, nullptr);
    const hid_t a = H5Acreate2(o, n, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, v.data());
    H5Aclose(a); H5Sclose(s);
}
hid_t dataset(hid_t f, const std::string& path, const std::vector<hsize_t>& dims, const std::vector<double>& v) {
    const hid_t lc = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(lc, 1);
    const hid_t s = H5Screate_simple(int(dims.size()), dims.data(), nullptr);
    const hid_t d = H5Dcreate2(f, path.c_str(), H5T_NATIVE_DOUBLE, s, lc, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(s); H5Pclose(lc);
    return d;
}
hid_t new_file(const std::string& name) {
    const hid_t f = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    attr_str(f, "openPMD", "1.1.0");
    attr_str(f, "basePath", "/data/%T/");
    attr_str(f, "meshesPath", "meshes/");
    return f;
}
} // namespace
#endif

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    RadialGrid g(RadialGrid::make_uniform(4.0, 0.05));
    const std::vector<double> zs = {0.0, 3.0, 12.5, 40.0, 77.7, 100.0, 180.0};
    const double n0 = 1e18, len_m = 5314.09 / std::sqrt(n0);   // c/omega_p [m]

    std::printf("1. loaded plasma: file against the built-in tables and parsed functions\n");
    {   // geometry z, columns, normalized units: the points of the z table
        write("df_z.txt", "# up-ramp and plateau\n# geometry = z\n# columns = z[kp^-1] n\n0 0\n50 1\n120 0.4\n400 0.4\n");
        Config a = base(), b = base();
        a.set("plasma.z_table", "0:0 50:1 120:0.4 400:0.4");
        b.set("plasma.density_file", "df_z.txt");
        for (int m1 = 0; m1 <= 1; ++m1) {
            const double d = diff(load(a, g, m1, zs), load(b, g, m1, zs));
            std::printf("    n(z) columns vs plasma.z_table (modes %d): %.2e\n", m1, d);
            check(d < 1e-13, "n(z) from a file = plasma.z_table");
        }
        // grid layout with a list axis, values over several lines
        write("df_zg.txt", "# density_profile = 1\n# geometry = z\n# density_unit = 1\n# z = list 0 50 120 400 kp^-1\n0 1\n0.4 0.4\n");
        Config c = base();
        c.set("plasma.density_file", "df_zg.txt");
        check(diff(load(b, g, false, zs), load(c, g, false, zs)) == 0.0, "grid layout (list axis) = columns layout");
        // physical units: mm and cm^-3 with units.n0_cm3
        write("df_zs.txt", "# geometry = z\n# columns = z[mm], n_e[cm^-3]\n0, 0\n" + num(50 * len_m * 1e3) + ", " + num(n0) + "\n" +
                               num(120 * len_m * 1e3) + ", " + num(0.4 * n0) + "\n" + num(400 * len_m * 1e3) + ", " + num(0.4 * n0) + "\n");
        Config s = base();
        s.set("units.n0_cm3", "1e18");
        s.set("plasma.density_file", "df_zs.txt");
        const double d = diff(load(a, g, false, zs), load(s, g, false, zs));
        std::printf("    n(z) in cm^-3 over mm vs plasma.z_table: %.2e\n", d);
        check(d < 1e-12, "absolute density in cm^-3 over mm (units.n0_cm3) = the table in units of n0");
        // file_z and file_outside
        Config sh = base(), ho = base(), ze = base();
        write("df_zshift.txt", "# geometry = z\n# columns = z[kp^-1] n\n1000 0\n1050 1\n1120 0.4\n1400 0.4\n");
        sh.set("plasma.density_file", "df_zshift.txt");
        sh.set("plasma.file_z", "-1000");
        check(diff(load(a, g, false, zs), load(sh, g, false, zs)) < 1e-12, "file shifted with plasma.file_z");
        write("df_zshort.txt", "# geometry = z\n# columns = z[kp^-1] n\n0 0\n50 1\n120 0.4\n");
        ho.set("plasma.density_file", "df_zshort.txt");
        ho.set("plasma.file_outside", "hold");
        check(diff(load(a, g, false, zs), load(ho, g, false, zs)) < 1e-13, "file_outside = hold continues the edge value");
        ze.set("plasma.density_file", "df_zshort.txt");
        DensityProfile pz(ze, "plasma");
        check(pz.eval(0.5, 0, 119.0) > 0 && pz.eval(0.5, 0, 121.0) == 0.0 && pz.eval(0.5, 0, -1.0) == 0.0,
              "outside the file the density is zero (default)");
    }
    {   // geometry rz: f(z) g(r), linear between the grid points -> z_table x radial_table
        const std::vector<double> zt = {0, 50, 400}, fz = {0, 1, 1}, rt = {0, 1, 2.5, 4}, gr = {1, 1.3, 0.6, 0.6};
        std::string cols = "# geometry = rz\n# columns = z[kp^-1] r[kp^-1] n\n";
        std::string gzr = "# geometry = rz\n# density_unit = 1\n# z = list 0 50 400 kp^-1\n# r = list 0 1 2.5 4 kp^-1\n# order = z r\n";
        std::string grz = "# geometry = rz\n# density_unit = 1\n# r = list 0 1 2.5 4 kp^-1\n# z = uniform 0 50 2 kp^-1\n";   // default order: r z
        for (int j = int(rt.size()) - 1; j >= 0; --j)   // rows in another order than the grid
            for (size_t i = 0; i < zt.size(); ++i) cols += num(zt[i]) + " " + num(rt[size_t(j)]) + " " + num(fz[i] * gr[size_t(j)]) + "\n";
        for (size_t i = 0; i < zt.size(); ++i) { for (size_t j = 0; j < rt.size(); ++j) gzr += num(fz[i] * gr[j]) + " "; gzr += "\n"; }
        for (size_t j = 0; j < rt.size(); ++j) grz += num(fz[0] * gr[j]) + " " + num(fz[1] * gr[j]) + "\n";
        write("df_rz.txt", cols);
        write("df_rz_zr.txt", gzr);
        write("df_rz_rz.txt", grz);
        Config a = base(), b = base(), c = base(), d2 = base();
        a.set("plasma.z_table", "0:0 50:1 400:1");
        a.set("plasma.radial", "table");
        a.set("plasma.radial_table", "0:1 1:1.3 2.5:0.6 4:0.6");
        b.set("plasma.density_file", "df_rz.txt");
        c.set("plasma.density_file", "df_rz_zr.txt");
        d2.set("plasma.density_file", "df_rz_rz.txt");
        d2.set("plasma.file_outside", "hold");   // this file ends at z = 50
        const std::vector<double> zin = {0.0, 3.0, 12.5, 40.0, 50.0};
        for (int m1 = 0; m1 <= 1; ++m1) {
            const double d = diff(load(a, g, m1, zs), load(b, g, m1, zs));
            std::printf("    n(r, z) columns vs z_table x radial_table (modes %d): %.2e\n", m1, d);
            check(d < 1e-12, "n(r, z) from a file = plasma.z_table x plasma.radial_table");
        }
        check(diff(load(b, g, false, zs), load(c, g, false, zs)) == 0.0, "rz grid layout, order z r = columns layout");
        check(diff(load(a, g, false, zs), load(d2, g, false, zs)) < 1e-12, "rz grid layout, order r z, uniform z axis, hold");
        (void)zin;
        // further factor density(x,y,z) and rmax
        Config f1 = base(), f2 = base();
        f1.set("plasma.density(x,y,z)", "if(z < 50, z/50, 1) * (1 + 0.2*x) * if(x^2+y^2 < 3^2, 1, 0)");
        write("df_ramp.txt", "# geometry = z\n# columns = z[kp^-1] n\n0 0\n50 1\n400 1\n");
        f2.set("plasma.density_file", "df_ramp.txt");
        f2.set("plasma.density(x,y,z)", "1 + 0.2*x");
        f2.set("plasma.rmax", "3");
        for (int m1 = 0; m1 <= 1; ++m1) {
            const double d = diff(load(f1, g, m1, zs), load(f2, g, m1, zs));
            std::printf("    file x density(x,y,z), rmax vs parsed profile (modes %d): %.2e\n", m1, d);
            check(d < 1e-12, "file x plasma.density(x,y,z) with plasma.rmax = the parsed profile");
        }
    }
    {   // transverse function in the header, with units
        const double w_um = 2.0 * len_m * 1e6;   // 2 c/omega_p in um
        write("df_t.txt", "# geometry = z\n# columns = z[kp^-1] n\n# transverse = \"(1 + 0.3 * x / (" + num(w_um) +
                              " um)) * exp(-(r / (" + num(w_um) + " um))^2)\"\n0 0\n50 1\n400 1\n");
        Config a = base(), b = base();
        a.set("plasma.density(x,y,z)", "if(z < 50, z/50, 1) * (1 + 0.3*x/2) * exp(-(x^2+y^2)/2^2)");
        b.set("units.n0_cm3", "1e18");
        b.set("plasma.density_file", "df_t.txt");
        for (int m1 = 0; m1 <= 1; ++m1) {
            const double d = diff(load(a, g, m1, zs), load(b, g, m1, zs));
            std::printf("    n(z) x transverse function vs parsed profile (modes %d): %.2e\n", m1, d);
            check(d < 1e-12, "transverse function of the header (lengths in um) = the parsed profile");
        }
        Config c = base();
        c.set("plasma.density_file", "df_t.txt");   // um without units.n0_cm3
        check(error_of([&] { DensityProfile p(c, "plasma"); }).find("need units.n0_cm3") != std::string::npos,
              "lengths in um without units.n0_cm3 are rejected");
    }
    {   // species with its own file
        Config a = base(), b = base();
        a.set("plasma.z_table", "0:0 50:1 400:1");
        b.set("e.density_file", "df_ramp.txt");
        check(diff(load(a, g, false, zs), load(b, g, false, zs)) < 1e-13, "<species>.density_file overrides the common profile");
    }

    std::printf("2. reader: geometries of the 3D codes\n");
    {
        densfile::DensityFileUnits u;
        u.length = [](double x, const std::string& unit) { return unit == "mm" ? 1e3 * x : x; };   // code unit: um
        u.density = [](double x, const std::string&) { return x * 1e-18; };
        auto f = [](double x, double y, double z) { return 2 + 0.1 * x - 0.05 * y + 0.01 * z + 0.002 * x * y - 0.0003 * y * z + 0.0001 * x * y * z; };
        const std::vector<double> xs = {-30, -5, 10, 30}, ys = {-20, 0, 25}, zv = {0, 100, 250, 1000};
        std::string grid = "# geometry = xyz\n# density_unit = cm^-3\n# x = list -30 -5 10 30 um\n# y = list -20 0 25 um\n"
                           "# z = list 0 0.1 0.25 1 mm\n# order = z y x\n";
        std::string cols = "# geometry = xyz\n# columns = y[um] n[cm^-3] z[mm] x[um]\n";
        for (double z : zv)
            for (double y : ys) {
                for (double x : xs) grid += num(1e18 * f(x, y, z)) + " ";
                grid += "\n";
            }
        for (int i = int(xs.size()) - 1; i >= 0; --i)
            for (double z : zv)
                for (double y : ys) cols += num(y) + " " + num(1e18 * f(xs[size_t(i)], y, z)) + " " + num(z * 1e-3) + " " + num(xs[size_t(i)]) + "\n";
        write("df_xyz.txt", grid);
        write("df_xyz_cols.txt", cols);
        const densfile::DensityFile A = densfile::readDensityFile("df_xyz.txt", "", -1, u);
        const densfile::DensityFile B = densfile::readDensityFile("df_xyz_cols.txt", "", -1, u);
        double e = 0, eab = 0;
        for (int k = 0; k < 2000; ++k) {
            const double x = -30 + 60 * std::fmod(k * 0.6180339887, 1.0), y = -20 + 45 * std::fmod(k * 0.7548776662, 1.0),
                         z = 1000 * std::fmod(k * 0.5698402910, 1.0);
            e = std::max(e, std::fabs(A.at(z, x, y) - f(x, y, z)));
            eab = std::max(eab, std::fabs(A.at(z, x, y) - B.at(z, x, y)));
        }
        std::printf("    xyz, trilinear function: largest error %.2e; columns - grid layout %.2e\n", e, eab);
        check(A.geometry == "xyz" && A.absolute && e < 1e-12, "xyz grid layout: a trilinear function is reproduced");
        check(eab == 0.0, "xyz columns layout (shuffled rows and columns) = grid layout");
        check(A.at(500, 31, 0) == 0.0 && A.at(500, 0, -21) == 0.0 && A.at(-1, 0, 0) == 0.0 && A.at(500, 31, 0, true) > 0,
              "xyz: zero outside the file, edge value with hold");
        write("df_xz.txt", "# geometry = xz\n# density_unit = 1\n# z = uniform 0 100 3 um\n# x = uniform -10 10 3 um\n# order = x z\n"
                           "1 2 3\n2 3 4\n3 4 5\n");
        const densfile::DensityFile C = densfile::readDensityFile("df_xz.txt", "", -1, u);
        check(C.geometry == "xz" && !C.absolute && std::fabs(C.at(150, 5) - (2 + 0.5 + 1.5)) < 1e-14, "xz grid layout, order x z");
        Config cfg = base();
        cfg.set("plasma.density_file", "df_xyz.txt");
        cfg.set("units.n0_cm3", "1e18");
        check(error_of([&] { DensityProfile p(cfg, "plasma"); }).find("3D profile n(x, y, z) is for the 3D codes") != std::string::npos,
              "QUARZ rejects geometry xyz");
        cfg.set("plasma.density_file", "df_xz.txt");
        check(error_of([&] { DensityProfile p(cfg, "plasma"); }).find("Cartesian 2D profile") != std::string::npos,
              "QUARZ rejects geometry xz");
    }

    std::printf("3. openPMD meshes\n");
#ifdef QUARZ_USE_HDF5
    {
        // regular thetaMode mesh [1][r][z], lengths in c/omega_p (gridUnitSI), a factor: the profile of df_rz.txt on
        // a regular grid that contains its points
        const int nr = 9, nz = 9;   // r = 0, 0.5, ..., 4;  z = 0, 50, ..., 400
        auto gr = [](double r) { return r <= 1 ? 1 + 0.3 * r : r <= 2.5 ? 1.3 - 0.7 * (r - 1) / 1.5 : 0.6; };
        std::vector<double> v(size_t(nr * nz));
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < nz; ++i) v[size_t(j * nz + i)] = (i == 0 ? 0.0 : 1.0) * gr(0.5 * j);
        hid_t f = new_file("df_theta.h5");
        hid_t d = dataset(f, "/data/0/meshes/density", {1, hsize_t(nr), hsize_t(nz)}, v);
        attr_str(d, "geometry", "thetaMode");
        attr_strs(d, "axisLabels", {"r", "z"});
        attr_str(d, "dataOrder", "C");
        attr_dbl(d, "gridSpacing", {0.5, 50.0});
        attr_dbl(d, "gridGlobalOffset", {0.0, 0.0});
        attr_dbl(d, "position", {0.0, 0.0});
        attr_dbl(d, "gridUnitSI", {len_m});
        attr_dbl(d, "unitSI", {1.0});
        attr_dbl(d, "unitDimension", {0, 0, 0, 0, 0, 0, 0});
        H5Dclose(d); H5Fclose(f);
        Config a = base(), b = base();
        a.set("plasma.density_file", "df_rz.txt");
        b.set("plasma.density_file", "df_theta.h5");
        b.set("units.n0_cm3", "1e18");
        const double dd = diff(load(a, g, false, zs), load(b, g, false, zs));
        std::printf("    regular thetaMode mesh vs text file: %.2e\n", dd);
        check(dd < 1e-9, "openPMD thetaMode mesh (regular axes, factor) = the text file");
        // rectilinear mesh [z][r] in SI with gridCoordinates, absolute density, a second mesh beside it
        const std::vector<double> zt = {0, 50, 400}, rt = {0, 1, 2.5, 4}, grv = {1, 1.3, 0.6, 0.6};
        std::vector<double> w, zc, rc;
        for (size_t i = 0; i < zt.size(); ++i)
            for (size_t j = 0; j < rt.size(); ++j) w.push_back((i == 0 ? 0.0 : 1.0) * grv[j] * n0 * 1e6);
        for (double z : zt) zc.push_back(z * len_m);
        for (double r : rt) rc.push_back(r * len_m);
        f = new_file("df_rect.h5");
        H5Dclose(dataset(f, "/data/0/gridCoordinates/z", {3}, zc));
        H5Dclose(dataset(f, "/data/0/gridCoordinates/r", {4}, rc));
        d = dataset(f, "/data/0/meshes/n_gas", {3, 4}, w);
        attr_str(d, "geometry", "other");
        attr_str(d, "geometryParameters", "thetaMode;m=0;imag=+");
        attr_strs(d, "axisLabels", {"z", "r"});
        attr_strs(d, "gridCoordinates", {"/data/0/gridCoordinates/z", "/data/0/gridCoordinates/r"});
        attr_dbl(d, "gridGlobalOffset", {0.0, 0.0});
        attr_dbl(d, "gridUnitSI", {1.0});
        attr_dbl(d, "unitSI", {1.0});
        attr_dbl(d, "unitDimension", {-3, 0, 0, 0, 0, 0, 0});
        H5Dclose(d);
        H5Dclose(dataset(f, "/data/0/meshes/temperature", {3, 4}, w));
        H5Fclose(f);
        Config c = base();
        c.set("plasma.density_file", "df_rect.h5");
        c.set("plasma.file_mesh", "n_gas");
        c.set("units.n0_cm3", "1e18");
        const double dc = diff(load(a, g, false, zs), load(c, g, false, zs));
        std::printf("    rectilinear mesh in SI vs text file: %.2e\n", dc);
        check(dc < 1e-9, "openPMD rectilinear mesh (gridCoordinates, m^-3, file_mesh) = the text file");
        Config e = base();
        e.set("plasma.density_file", "df_rect.h5");
        e.set("units.n0_cm3", "1e18");
        check(error_of([&] { DensityProfile p(e, "plasma"); }).find("give the mesh name") != std::string::npos,
              "two meshes and no plasma.file_mesh: rejected with the list of meshes");
        // 3D mesh [z][y][x] for the reader
        densfile::DensityFileUnits u;
        u.length = [](double x, const std::string&) { return x * 1e6; };   // m -> um
        u.density = [](double x, const std::string&) { return x * 1e-24; };
        auto fn = [](double x, double y, double z) { return 2 + 0.1 * x - 0.05 * y + 0.01 * z + 0.0001 * x * y * z; };
        std::vector<double> v3;
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 5; ++i) v3.push_back(1e24 * fn(-20 + 10 * i, -15 + 15 * j, 100.0 * k));
        f = new_file("df_3d.h5");
        d = dataset(f, "/data/0/meshes/density", {4, 3, 5}, v3);
        attr_str(d, "geometry", "cartesian");
        attr_strs(d, "axisLabels", {"z", "y", "x"});
        attr_dbl(d, "gridSpacing", {100e-6, 15e-6, 10e-6});
        attr_dbl(d, "gridGlobalOffset", {0.0, -15e-6, -20e-6});
        attr_dbl(d, "gridUnitSI", {1.0});
        attr_dbl(d, "unitSI", {1.0});
        attr_dbl(d, "unitDimension", {-3, 0, 0, 0, 0, 0, 0});
        H5Dclose(d); H5Fclose(f);
        const densfile::DensityFile A = densfile::readDensityFile("df_3d.h5", "", -1, u);
        double er = 0;
        for (int k = 0; k < 500; ++k) {
            const double x = -20 + 40 * std::fmod(k * 0.6180339887, 1.0), y = -15 + 30 * std::fmod(k * 0.7548776662, 1.0),
                         z = 300 * std::fmod(k * 0.5698402910, 1.0);
            er = std::max(er, std::fabs(A.at(z, x, y) - fn(x, y, z)));
        }
        std::printf("    3D mesh [z][y][x], trilinear function: largest error %.2e\n", er);
        check(A.geometry == "xyz" && er < 1e-9, "openPMD 3D mesh: geometry xyz from the axes, values reproduced");
    }
#else
    std::printf("    skipped: this build has no HDF5\n");
#endif

    std::printf("4. rejected files and combinations, run summary\n");
    {
        const std::string H = "# geometry = z\n# columns = z[kp^-1] n\n";
        check(rejects("# z[kp^-1] n\n0 0\n100 1\n", "no line '# geometry = ...'"), "no geometry");
        check(rejects("# geometry = z\n0 0\n100 1\n", "neither '# columns = ...' nor the axes"), "no layout");
        check(rejects("# geometry = rz\n# columns = z[kp^-1] n\n0 0\n100 1\n", "does not fit geometry rz"), "columns do not fit the geometry");
        check(rejects(H + "# presure = 1 bar\n0 0\n100 1\n", "unknown header key 'presure'"), "unknown header key");
        check(rejects(H + "0 0\n100 -1\n", "line 4: negative density"), "negative density");
        check(rejects(H + "0 0\n100 nan\n", "not finite"), "nan");
        check(rejects(H + "0 0\n100 1\n100 1\n", "strictly increasing"), "z not increasing");
        check(rejects("# geometry = rz\n# columns = z[kp^-1] r[kp^-1] n\n0 0 1\n0 1 1\n100 0 1\n", "do not fill a rectilinear grid"), "incomplete grid");
        check(rejects("# geometry = z\n# density_unit = 1\n# z = uniform 0 100 3 kp^-1\n0 1\n", "2 values, but the axes of the header give 3"), "wrong number of values");
        check(rejects("# geometry = z\n# columns = z[fs] n\n0 0\n100 1\n", "unknown length unit 'fs'"), "wrong unit");
        check(rejects("# geometry = z\n# columns = z n\n0 0\n100 1\n", "needs a unit"), "no length unit");
        check(rejects("# geometry = z\n# columns = z[kp^-1] n[cm^-3]\n0 0\n100 1e18\n", "needs units.n0_cm3"), "cm^-3 without units.n0_cm3");
        check(rejects("# geometry = rz\n# columns = z[kp^-1] r[kp^-1] n\n# transverse = \"1\"\n0 0 1\n0 1 1\n9 0 1\n9 1 1\n", "only for geometry z"),
              "transverse function with geometry rz");
        check(rejects("# geometry = z\n# columns = z[kp^-1] n\n# transverse = \"exp(-(r/\"\n0 0\n9 1\n", "transverse function"), "faulty transverse function");
        Config c = base();
        c.set("plasma.density_file", "df_ramp.txt");
        c.set("plasma.z_table", "0:0 1:1");
        check(error_of([&] { DensityProfile p(c, "plasma"); }).find("is given by plasma.density_file") != std::string::npos,
              "plasma.z_table together with the file");
        Config c2 = base();
        c2.set("plasma.file_z", "1");
        check(error_of([&] { DensityProfile p(c2, "plasma"); }).find("only used with plasma.density_file") != std::string::npos,
              "plasma.file_z without a file");
        Config c3 = base();
        c3.set("plasma.density_file", "df_nothing.txt");
        check(error_of([&] { DensityProfile p(c3, "plasma"); }).find("cannot open") != std::string::npos, "missing file");

        Config r = base();
        r.set("plasma.density_file", "df_ramp.txt");
        r.set("units.n0_cm3", "1e18");
        DensityProfile p(r, "plasma");
        auto has = [](const std::vector<std::string>& L, const std::string& s) {
            for (const auto& l : L) if (l.find(s) != std::string::npos) return true;
            return false;
        };
        const auto ok = p.file_report(0.0, 300.0, 5.0);
        for (const auto& l : ok) std::printf("    | %s\n", l.c_str());
        check(has(ok, "n(z) on 3 points") && has(ok, "z = 0 ... 400 (run: 0 ... 300)") && has(ok, "peak 1 n0 at z = 50") &&
                  !has(ok, "Warning"), "summary: points, ranges against the run, peak; no warning");
        check(has(p.file_report(0.0, 800.0, 5.0), "Warning: plasma.density_file ends at z = 400 with 100 % of its peak"),
              "warning: the file ends inside the run at a finite density");
        check(has(p.file_report(0.0, 300.0, 20.0), "less than 4 steps"), "warning: gradient shorter than 4 time steps");
        check(error_of([&] { p.file_report(500.0, 800.0, 5.0); }).find("outside the run") != std::string::npos,
              "a file that lies outside the run is an error");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
