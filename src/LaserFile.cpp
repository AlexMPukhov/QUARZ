#include "LaserFile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef QUARZ_USE_HDF5
#include <hdf5.h>
#endif

namespace quarz {

using cd = std::complex<double>;

namespace {
const double PI_ = 3.14159265358979323846;
const double c_si = 299792458.0, e_si = 1.602176634e-19, me_si = 9.1093837015e-31, eps0_si = 8.8541878128e-12;

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char s[256];
    std::snprintf(s, sizeof s, f, a, b, c);
    return s;
}

// a_q <- sum_l a_l exp(sign 2 pi i q l / N), N a power of two
void fft(cd* a, size_t n, int sign) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2 * PI_ / double(len);
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const cd w = std::polar(1.0, ang * double(k));
                const cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
}

// trapezoidal weights on [0, (n-1) h] with Gregory's corrections at the left end (integrands odd at 0)
double gregory_left(int p, int n) {
    double w = (p == 0 || p == n - 1) ? 0.5 : 1.0;
    if (n < 10) return w;
    const double a1 = 1.0 / 12, a2 = -1.0 / 24, a3 = 19.0 / 720, a4 = -3.0 / 160;
    const double c[5] = {-a1 + a2 - a3 + a4, a1 - 2 * a2 + 3 * a3 - 4 * a4, a2 - 3 * a3 + 6 * a4, a3 - 4 * a4, a4};
    if (p < 5) w += c[p];
    return w;
}

// Weights W_j of F(kappa) = int f(r) J_nu(kappa r) r dr = sum_j W_j f_j on the uniform nodes r_j = j dr, exact for
// f = r^nu g(r^2) with g linear in r^2 between the nodes (regular at the axis; GEM-PIC-RZ src/laser_file.cpp)
std::vector<double> hankel_weights(int nu, double kap, double dr, int nr) {
    std::vector<double> W(static_cast<size_t>(nr), 0.0);
    auto rr = [&](int j) { return j * dr; };
    auto B1 = [&](double r) {
        if (kap == 0.0) return nu == 0 ? 0.5 * r * r : 0.0;
        return std::pow(r, nu + 1) * std::cyl_bessel_j(nu + 1.0, kap * r) / kap;
    };
    auto B3 = [&](double r) {
        if (kap == 0.0) return nu == 0 ? 0.25 * r * r * r * r : 0.0;
        return std::pow(r, nu + 3) * std::cyl_bessel_j(nu + 1.0, kap * r) / kap -
               2.0 * std::pow(r, nu + 2) * std::cyl_bessel_j(nu + 2.0, kap * r) / (kap * kap);
    };
    auto addG = [&](int j, double c) {
        const double r = rr(j);
        if (r > 0.0 || nu == 0) { W[size_t(j)] += c / std::pow(r, nu); return; }
        if (nr < 3) { W[1] += c / std::pow(rr(1), nu); return; }
        const double ua = rr(1) * rr(1), ub = rr(2) * rr(2), x = ua / (ub - ua);
        W[1] += c * (1.0 + x) / std::pow(rr(1), nu);
        W[2] -= c * x / std::pow(rr(2), nu);
    };
    double b1a = B1(0), b3a = B3(0);
    for (int j = 0; j < nr - 1; ++j) {
        const double b1b = B1(rr(j + 1)), b3b = B3(rr(j + 1)), I1 = b1b - b1a;
        const double ua = rr(j) * rr(j), du = rr(j + 1) * rr(j + 1) - ua, Iu = (b3b - b3a) - ua * I1;
        addG(j, I1 - Iu / du);
        addG(j + 1, Iu / du);
        b1a = b1b; b3a = b3b;
    }
    return W;
}

// The file's mode e^{i l theta} of the scalar envelope (times the dominant polarization component) on t (u0 + n du) x r (j dr)
struct FileMode {
    int nu = 0, nr = 0;
    double u0 = 0, du = 1, dr = 1;
    std::vector<cd> e;                      // [n * nr + j]
    cd& at(int n, int j) { return e[size_t(n) * nr + j]; }
};

#ifdef QUARZ_USE_HDF5
struct H {                                  // owns an HDF5 id
    hid_t id = -1;
    explicit H(hid_t i) : id(i) {}
    ~H() {
        if (id < 0) return;
        switch (H5Iget_type(id)) {
            case H5I_FILE: H5Fclose(id); break;
            case H5I_GROUP: H5Gclose(id); break;
            case H5I_DATASET: H5Dclose(id); break;
            case H5I_ATTR: H5Aclose(id); break;
            case H5I_DATATYPE: H5Tclose(id); break;
            case H5I_DATASPACE: H5Sclose(id); break;
            default: break;
        }
    }
    H(const H&) = delete;
    H& operator=(const H&) = delete;
};

bool has_attr(hid_t o, const char* n) { return H5Aexists(o, n) > 0; }

std::vector<std::string> attr_strings(hid_t o, const char* n) {
    H a(H5Aopen(o, n, H5P_DEFAULT)), t(H5Aget_type(a.id)), s(H5Aget_space(a.id));
    if (H5Tget_class(t.id) != H5T_STRING) throw std::runtime_error(std::string("attribute ") + n + " is not a string");
    const hssize_t cnt = std::max<hssize_t>(1, H5Sget_simple_extent_npoints(s.id));
    std::vector<std::string> out;
    if (H5Tis_variable_str(t.id) > 0) {
        std::vector<char*> p(static_cast<size_t>(cnt), nullptr);
        H mt(H5Tcopy(H5T_C_S1));
        H5Tset_size(mt.id, H5T_VARIABLE);
        H5Aread(a.id, mt.id, p.data());
        for (char* c : p) { out.push_back(c ? c : ""); }
        H5Dvlen_reclaim(mt.id, s.id, H5P_DEFAULT, p.data());
    } else {
        const size_t sz = H5Tget_size(t.id);
        std::vector<char> buf(sz * size_t(cnt) + 1, 0);
        H mt(H5Tcopy(H5T_C_S1));
        H5Tset_size(mt.id, sz);
        H5Tset_strpad(mt.id, H5T_STR_NULLPAD);   // h5py writes fixed strings without a terminator (numpy S)
        H5Aread(a.id, mt.id, buf.data());
        for (hssize_t k = 0; k < cnt; ++k) {
            std::string v(buf.data() + size_t(k) * sz, sz);
            v.erase(std::find(v.begin(), v.end(), '\0'), v.end());
            out.push_back(v);
        }
    }
    return out;
}
std::string attr_string(hid_t o, const char* n) { return attr_strings(o, n).at(0); }

// numbers (real) or a compound (r, i): (re, im) pairs
std::vector<double> attr_reim(hid_t o, const char* n) {
    H a(H5Aopen(o, n, H5P_DEFAULT)), t(H5Aget_type(a.id)), s(H5Aget_space(a.id));
    const hssize_t cnt = std::max<hssize_t>(1, H5Sget_simple_extent_npoints(s.id));
    std::vector<double> v(2 * size_t(cnt), 0.0);
    if (H5Tget_class(t.id) == H5T_COMPOUND) {
        H ct(H5Tcreate(H5T_COMPOUND, 16));
        H5Tinsert(ct.id, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(ct.id, "i", 8, H5T_NATIVE_DOUBLE);
        H5Aread(a.id, ct.id, v.data());
    } else {
        std::vector<double> r(static_cast<size_t>(cnt));
        H5Aread(a.id, H5T_NATIVE_DOUBLE, r.data());
        for (size_t k = 0; k < r.size(); ++k) v[2 * k] = r[k];
    }
    return v;
}
std::vector<double> attr_vec(hid_t o, const char* n) {
    const std::vector<double> v = attr_reim(o, n);
    std::vector<double> r;
    for (size_t k = 0; k < v.size(); k += 2) r.push_back(v[k]);
    return r;
}
double attr_num(hid_t o, const char* n) { return attr_vec(o, n).at(0); }

herr_t collect(hid_t, const char* name, const H5L_info2_t*, void* op) {
    static_cast<std::vector<std::string>*>(op)->push_back(name);
    return 0;
}

// reads the file's mode l (the envelope times the dominant polarization component) in QUARZ units
FileMode read_mode(const LaserFileInput& in, double& omegaL, cd& px, cd& py, bool& vecpot, double& mode_loss,
                   std::vector<std::string>& info) {
    const double wp = 5.64146e4 * std::sqrt(in.n0_cm3), kp = wp / c_si, Ep = me_si * c_si * wp / e_si;
    (void)eps0_si;
    const std::string where = in.path + ": ";
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    H f(H5Fopen(in.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (f.id < 0) throw std::runtime_error(where + "cannot open the file");
    if (!has_attr(f.id, "openPMD")) throw std::runtime_error(where + "not an openPMD file");
    H data(H5Gopen2(f.id, "data", H5P_DEFAULT));
    if (data.id < 0) throw std::runtime_error(where + "no group data");
    long it = in.iteration;
    if (it < 0) {
        std::vector<std::string> names;
        H5Literate2(data.id, H5_INDEX_NAME, H5_ITER_INC, nullptr, collect, &names);
        for (const std::string& n : names) it = std::max(it, std::atol(n.c_str()));
    }
    std::string mp = has_attr(f.id, "meshesPath") ? attr_string(f.id, "meshesPath") : "meshes/";
    if (mp.empty() || mp.back() != '/') mp += '/';
    const std::string path = std::to_string(it) + "/" + mp + "laserEnvelope";
    if (H5Lexists(data.id, std::to_string(it).c_str(), H5P_DEFAULT) <= 0)
        throw std::runtime_error(where + "no iteration " + std::to_string(it));
    H ds(H5Dopen2(data.id, path.c_str(), H5P_DEFAULT));
    if (ds.id < 0) throw std::runtime_error(where + "no mesh laserEnvelope in iteration " + std::to_string(it));
    const std::string geo = attr_string(ds.id, "geometry");
    const bool cart = geo == "cartesian";
    if (!cart && geo != "thetaMode") throw std::runtime_error(where + "geometry " + geo + " (cartesian or thetaMode)");
    std::vector<std::string> lab = attr_strings(ds.id, "axisLabels");
    std::vector<double> sp = attr_vec(ds.id, "gridSpacing"), off = attr_vec(ds.id, "gridGlobalOffset");
    std::vector<double> pos = has_attr(ds.id, "position") ? attr_vec(ds.id, "position") : std::vector<double>(lab.size(), 0.0);
    if (has_attr(ds.id, "dataOrder") && attr_string(ds.id, "dataOrder") == "F") {
        std::reverse(lab.begin(), lab.end()); std::reverse(sp.begin(), sp.end());
        std::reverse(off.begin(), off.end()); std::reverse(pos.begin(), pos.end());
    }
    const size_t na = cart ? 3 : 2;
    if (lab.size() != na || sp.size() != na || off.size() != na || pos.size() != na)
        throw std::runtime_error(where + "expected " + std::to_string(na) + " axes");
    auto axis = [&](const char* n) {
        for (size_t k = 0; k < lab.size(); ++k)
            if (lab[k] == n) return k;
        throw std::runtime_error(where + "no axis " + n);
    };
    const double gu = has_attr(ds.id, "gridUnitSI") ? attr_num(ds.id, "gridUnitSI") : 1.0;
    const double usi = has_attr(ds.id, "unitSI") ? attr_num(ds.id, "unitSI") : 1.0;
    if (!has_attr(ds.id, "angularFrequency")) throw std::runtime_error(where + "no attribute angularFrequency");
    omegaL = attr_num(ds.id, "angularFrequency") / wp;
    px = 1; py = 0;
    if (has_attr(ds.id, "polarization")) {
        const std::vector<double> p = attr_reim(ds.id, "polarization");
        if (p.size() != 4) throw std::runtime_error(where + "polarization must have 2 entries");
        px = cd(p[0], p[1]); py = cd(p[2], p[3]);
    }
    const double pn = std::sqrt(std::norm(px) + std::norm(py));
    if (!(pn > 0)) throw std::runtime_error(where + "polarization is zero");
    px /= pn; py /= pn;
    const std::string field = has_attr(ds.id, "envelopeField") ? attr_string(ds.id, "envelopeField") : "normalized_vector_potential";
    vecpot = field == "normalized_vector_potential";
    if (!vecpot && field != "electric_field") throw std::runtime_error(where + "envelopeField " + field);
    H space(H5Dget_space(ds.id));
    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(space.id) != 3) throw std::runtime_error(where + "the envelope must have 3 dimensions");
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    std::vector<double> raw(2 * size_t(dims[0]) * dims[1] * dims[2]);
    {
        H ct(H5Tcreate(H5T_COMPOUND, 16));
        H5Tinsert(ct.id, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(ct.id, "i", 8, H5T_NATIVE_DOUBLE);
        if (H5Dread(ds.id, ct.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0)
            throw std::runtime_error(where + "cannot read the envelope (a complex compound r, i expected)");
    }
    auto R = [&](size_t q) { return cd(raw[2 * q], raw[2 * q + 1]); };
    const double unit = usi / (vecpot ? 1.0 : Ep);
    const size_t at = axis("t");
    FileMode m;
    m.du = sp[at] * gu * wp;
    m.u0 = (off[at] + pos[at] * sp[at]) * gu * wp;
    const int L = in.l;
    double all = 0, kept = 0;
    if (!cart) {
        const size_t ar = axis("r");
        const int nc = int(dims[0]), K = (nc - 1) / 2;
        m.nu = int(dims[1 + at]);
        const int nr0 = int(dims[1 + ar]);
        const double dr = sp[ar] * gu * kp, r0 = (off[ar] + pos[ar] * sp[ar]) * gu * kp;
        const size_t st[2] = {size_t(dims[2]), 1};
        // env = sum_k a_k e^{-i k theta}: a_0 = C_0, a_{+-k} = (C_k +- i S_k) / 2; e^{+i l theta} is a_{-l}
        auto mode = [&](int k, int n, int j) -> cd {
            auto C = [&](int comp) { return R(size_t(comp) * dims[1] * dims[2] + n * st[at] + j * st[ar]); };
            if (k == 0) return C(0);
            const int a = std::abs(k);
            if (a > K) return 0;
            return 0.5 * (C(2 * a - 1) + (k > 0 ? cd(0, 1) : cd(0, -1)) * C(2 * a));
        };
        m.dr = dr;
        m.nr = int(std::floor((r0 + (nr0 - 1) * dr) / dr + 1e-9)) + 1;   // nodes j dr from the axis
        m.e.assign(size_t(m.nu) * m.nr, cd(0));
        for (int n = 0; n < m.nu; ++n) {
            for (int j = 0; j < m.nr; ++j) {
                const double r = j * dr, y = (r - r0) / dr;
                cd v;
                if (y < 0) v = mode(-L, n, 0) * (L ? r / r0 : 1.0);
                else {
                    const int a = std::min(int(y), nr0 - 2);
                    const double w = std::min(1.0, y - a);
                    v = (1 - w) * mode(-L, n, a) + w * mode(-L, n, a + 1);
                }
                m.at(n, j) = unit * v;
            }
            for (int j = 0; j < nr0; ++j) {
                const double wr = std::max(r0 + j * dr, 0.25 * dr);
                for (int k = -K; k <= K; ++k) {
                    const double e = std::norm(mode(k, n, j)) * wr;
                    all += e;
                    if (k == -L) kept += e;
                }
            }
        }
        info.push_back(fmt("thetaMode file: %.0f modes, %.0f x %.0f (t, r)", K + 1, m.nu, nr0));
    } else {
        const size_t ax = axis("x"), ay = axis("y");
        const int nx = int(dims[ax]), ny = int(dims[ay]);
        const double dx = sp[ax] * gu * kp, dy = sp[ay] * gu * kp;
        const double x0 = (off[ax] + pos[ax] * sp[ax]) * gu * kp, y0 = (off[ay] + pos[ay] * sp[ay]) * gu * kp;
        const size_t st[3] = {size_t(dims[1] * dims[2]), size_t(dims[2]), 1};
        if (nx < 2 || ny < 2) throw std::runtime_error(where + "QUARZ needs a cartesian file with both transverse axes");
        const double Rm = std::min({x0 + (nx - 1) * dx, -x0, y0 + (ny - 1) * dy, -y0});
        m.dr = std::min(dx, dy);
        if (!(Rm > 2 * m.dr)) throw std::runtime_error(where + "the cartesian grid must contain its origin (the axis) well inside");
        m.nu = int(dims[at]);
        m.nr = int(std::floor(Rm / m.dr + 1e-9)) + 1;
        m.e.assign(size_t(m.nu) * m.nr, cd(0));
        const int nth = std::max(32, 8 * (std::abs(L) + 2));
        for (int n = 0; n < m.nu; ++n)
            for (int j = 0; j < m.nr; ++j) {
                cd s = 0;
                double e2 = 0;
                for (int q = 0; q < nth; ++q) {
                    const double ph = 2 * PI_ * q / nth, fx = (j * m.dr * std::cos(ph) - x0) / dx, fy = (j * m.dr * std::sin(ph) - y0) / dy;
                    const int ix = std::min(std::max(int(std::floor(fx)), 0), nx - 2), iy = std::min(std::max(int(std::floor(fy)), 0), ny - 2);
                    const double wx = std::min(1.0, std::max(0.0, fx - ix)), wy = std::min(1.0, std::max(0.0, fy - iy));
                    auto E = [&](int i, int k) { return R(n * st[at] + i * st[ax] + k * st[ay]); };
                    const cd e = (1 - wy) * ((1 - wx) * E(ix, iy) + wx * E(ix + 1, iy)) + wy * ((1 - wx) * E(ix, iy + 1) + wx * E(ix + 1, iy + 1));
                    s += e * std::polar(1.0, -L * ph);   // the e^{+i l theta} component
                    e2 += std::norm(e);
                }
                m.at(n, j) = unit * s / double(nth);
                const double wr = std::max(j * m.dr, 0.25 * m.dr);
                all += e2 / nth * wr; kept += std::norm(s / double(nth)) * wr;
            }
        info.push_back(fmt("cartesian file: %.0f x %.0f x %.0f (t, y, x), resampled on polar nodes around its origin", m.nu, ny, nx));
    }
    mode_loss = all > 0 ? std::max(0.0, 1 - kept / all) : 0.0;
    return m;
}
#endif
} // namespace

#ifndef QUARZ_USE_HDF5
bool laser_file_available() { return false; }
LaserFileResult laser_from_file(const LaserFileInput& in, const std::vector<double>&, const std::vector<double>&) {
    throw std::runtime_error("laser.file = " + in.path + ": this build has no HDF5 (configure with HDF5 to read LASY files)");
}
#else
bool laser_file_available() { return true; }

LaserFileResult laser_from_file(const LaserFileInput& in, const std::vector<double>& r, const std::vector<double>& xi) {
    if (!(in.n0_cm3 > 0)) throw std::runtime_error("laser.file needs the plasma density units.n0_cm3 (SI units of the file)");
    LaserFileResult out;
    cd px, py;
    bool vecpot;
    FileMode m = read_mode(in, out.omegaL, px, py, vecpot, out.mode_loss, out.info);
    // polarization: a linear (real up to a phase) or circular (|p_x| = |p_y|, 90 degrees) vector; the scalar envelope of
    // QUARZ is the dominant component (circular: a_x = Re[a^ e^{...}], <a^2> = |a^|^2)
    const double cross = std::abs((std::conj(px) * py).imag()), lin = std::abs((std::conj(px) * py).real());
    out.circular = std::abs(std::abs(px) - std::abs(py)) < 1e-6 && cross > 0.5 - 1e-6;
    if (!out.circular && cross > 1e-6 * (lin + 1e-300) && cross > 1e-9)
        throw std::runtime_error(in.path + ": elliptical polarization (QUARZ: linear or circular)");
    const cd pd = std::abs(px) >= std::abs(py) ? px : py;
    for (cd& v : m.e) v *= out.circular ? pd : pd / std::abs(pd);
    out.k0 = in.k0 > 0 ? in.k0 : out.omegaL;
    // peak and centroid at the file plane (the vector potential ~ E / omega_L for the report)
    double peak = 0, w = 0, wt = 0;
    for (int n = 0; n < m.nu; ++n)
        for (int j = 0; j < m.nr; ++j) {
            const double a = std::norm(m.at(n, j));
            peak = std::max(peak, std::sqrt(a));
            const double wr = std::max(j * m.dr, 0.25 * m.dr);
            w += a * wr; wt += a * wr * (m.u0 + n * m.du);
        }
    if (!(peak > 0)) throw std::runtime_error(in.path + ": the envelope (mode l = " + std::to_string(in.l) + ") is zero");
    const double tc = in.t_center < 1e299 ? in.t_center : wt / w;
    // spectrum: F[j][q] = sum_n f_n e^{+i Omega_q (u_n - Us)}, the full field (1/N) sum_q F e^{i Omega_q Us} e^{-i omega_q u}
    const int nt2 = [&] { int p = 1; while (p < 2 * m.nu) p <<= 1; return p; }(), offn = (nt2 - m.nu) / 2;
    const double Us = m.u0 - offn * m.du, dOm = 2 * PI_ / (nt2 * m.du);
    std::vector<cd> F(size_t(m.nr) * nt2, cd(0));
    for (int j = 0; j < m.nr; ++j) {
        cd* s = &F[size_t(j) * nt2];
        for (int n = 0; n < m.nu; ++n) s[offn + n] = m.at(n, j);
        fft(s, size_t(nt2), +1);
    }
    std::vector<double> pw(static_cast<size_t>(nt2), 0.0);
    for (int j = 0; j < m.nr; ++j)
        for (int q = 0; q < nt2; ++q) pw[size_t(q)] += std::norm(F[size_t(j) * nt2 + q]) * std::max(j * m.dr, 0.25 * m.dr);
    const double pmax = *std::max_element(pw.begin(), pw.end());
    int qlo = nt2, qhi = -nt2;
    for (int q = 0; q < nt2; ++q) {
        const int qs = q < nt2 / 2 ? q : q - nt2;
        if (pw[size_t(q)] > 1e-12 * pmax && out.omegaL + qs * dOm > 0) { qlo = std::min(qlo, qs); qhi = std::max(qhi, qs); }
    }
    if (qhi < qlo) throw std::runtime_error(in.path + ": no propagating frequency in the spectrum");
    const int nq = qhi - qlo + 1;
    // the vector potential per frequency: A = E / (i omega) (E-field files); rescale to a0 at the file plane
    std::vector<cd> A(size_t(m.nr) * nq);
    for (int j = 0; j < m.nr; ++j)
        for (int q = 0; q < nq; ++q) {
            const int qi = ((qlo + q) % nt2 + nt2) % nt2;
            const double om = out.omegaL + (qlo + q) * dOm;
            A[size_t(j) * nq + q] = F[size_t(j) * nt2 + qi] * std::polar(1.0, (qlo + q) * dOm * Us) / double(nt2) *
                                    (vecpot ? cd(1) : cd(0, -1) / om);
        }
    // peak |a^| at the file plane from the spectrum (inverse transform on the file's times)
    double apk = 0;
    for (int j = 0; j < m.nr; ++j)
        for (int n = 0; n < m.nu; n += 1) {
            const double u = m.u0 + n * m.du;
            cd s = 0;
            for (int q = 0; q < nq; ++q) s += A[size_t(j) * nq + q] * std::polar(1.0, -(out.omegaL + (qlo + q) * dOm) * u);
            apk = std::max(apk, std::abs(s));
        }
    const double scale = in.a0 > 0 ? in.a0 / apk : 1.0;
    out.peak_file = apk * scale;
    // Hankel transform of order |l| with the inverse weights kappa dk folded in
    const int ord = std::abs(in.l);
    double rmax = 0;
    for (double x : r) rmax = std::max(rmax, x);
    const double Rf = (m.nr - 1) * m.dr, Rn = std::max(Rf, rmax), dk = 2 * PI_ / (4 * (Rf + Rn));   // the images of the discrete inverse transform lie 2 pi / dk away: 4 (Rf + Rn) keeps
                                                                       // them off the radii used (2 (Rf + Rn) left 9e-3 at r = Rn)
    const double wmax = out.omegaL + qhi * dOm, kmax = std::min(PI_ / m.dr, wmax);
    const int nk = std::max(2, int(std::floor(kmax / dk)) + 1);
    std::vector<cd> G(size_t(nq) * nk, cd(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int p = 0; p < nk; ++p) {
        const double kap = p * dk;
        const std::vector<double> W = hankel_weights(ord, kap, m.dr, m.nr);
        for (int q = 0; q < nq; ++q) {
            cd s = 0;
            for (int j = 0; j < m.nr; ++j) s += A[size_t(j) * nq + q] * W[size_t(j)];
            G[size_t(q) * nk + p] = s * kap * dk * gregory_left(p, nk) * scale;
        }
    }
    std::vector<double> beta(size_t(nq) * nk, 0.0);
    std::vector<char> prop(size_t(nq) * nk, 0);
    double e_all = 0, e_kept = 0;
    for (int q = 0; q < nq; ++q) {
        const double om = out.omegaL + (qlo + q) * dOm;
        for (int p = 0; p < nk; ++p) {
            const double kz2 = om * om - p * dk * p * dk;
            const double e = std::norm(G[size_t(q) * nk + p]) / std::max(p * dk, 0.25 * dk);
            e_all += e;
            if (kz2 > 0) { beta[size_t(q) * nk + p] = std::sqrt(kz2) - om; prop[size_t(q) * nk + p] = 1; e_kept += e; }
        }
    }
    // the snapshot: S(xi, kappa) = sum_q G e^{i beta dz(xi)} e^{-i omega_q u(xi)}, a^(r, xi) = e^{i k0 xi} sum_p S J(kappa r)
    const int nx = int(xi.size()), nrq = int(r.size());
    out.a.assign(size_t(nx) * nrq, cd(0));
    std::vector<double> J(size_t(nrq) * nk);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int j = 0; j < nrq; ++j)
        for (int p = 0; p < nk; ++p) J[size_t(j) * nk + p] = std::cyl_bessel_j(double(ord), p * dk * r[size_t(j)]);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int k = 0; k < nx; ++k) {
        const double dz = -(in.file_z + xi[size_t(k)] - in.xi0), u = xi[size_t(k)] - in.xi0 + tc;
        std::vector<cd> S(static_cast<size_t>(nk), cd(0));
        for (int q = 0; q < nq; ++q) {
            const double om = out.omegaL + (qlo + q) * dOm;
            const cd ph = std::polar(1.0, -om * u);
            for (int p = 0; p < nk; ++p)
                if (prop[size_t(q) * nk + p]) S[size_t(p)] += G[size_t(q) * nk + p] * ph * std::polar(1.0, beta[size_t(q) * nk + p] * dz);
        }
        const cd car = std::polar(1.0, out.k0 * xi[size_t(k)]);
        for (int j = 0; j < nrq; ++j) {
            cd s = 0;
            const double* Jj = &J[size_t(j) * nk];
            for (int p = 0; p < nk; ++p) s += S[size_t(p)] * Jj[p];
            out.a[size_t(k) * nrq + j] = car * s;
        }
    }
    out.info.push_back(fmt("omega_L / omega_p = %.6g, ", out.omegaL) + (vecpot ? "vector potential, " : "electric field, ") +
                       std::string(out.circular ? "circular" : "linear") +
                       " polarization" + fmt(", mode e^{i %.0f theta} (other modes: %.2g of the energy, dropped)", in.l, out.mode_loss));
    out.info.push_back(fmt("peak |a^| at the file plane %.4g, t_center %.6g (1/omega_p) of the file at xi0", out.peak_file, tc) +
                       fmt(", file plane %.6g (1/k_p) ahead of the pulse centre", in.file_z));
    out.info.push_back(fmt("exact vacuum propagation per (omega, kappa): FFT %.0f (%.0f frequencies), %.0f radial wavenumbers", nt2, nq, nk) +
                       fmt(", %.2g of the energy evanescent, dropped", e_all > 0 ? 1 - e_kept / e_all : 0.0));
    return out;
}
#endif

} // namespace quarz
