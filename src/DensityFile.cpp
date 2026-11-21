// ============================================================================
//  DensityFile.cpp -- density profile files: text tables and openPMD meshes
//  (format: docs/density_file.md; see DensityFile.hpp)
// ============================================================================
#include "DensityFile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#if defined(DENSITY_FILE_USE_HDF5)
#include <hdf5.h>
#endif

namespace densfile {

namespace {

[[noreturn]] void failAt(const std::string& path, const std::string& m) { throw std::runtime_error(path + ": " + m); }

const char* HEADER_HELP =
    "the header (comment lines '# key = value') must state the geometry and the data layout, e.g.\n"
    "    # geometry = z                        or    # geometry = rz\n"
    "    # columns  = z[mm] n[cm^-3]                 # columns  = z[mm] r[um] n[cm^-3]\n"
    "(see docs/density_file.md; a density without unit, n or n[1], is a dimensionless factor)";

bool isSep(char c) { return c == ' ' || c == '\t' || c == ',' || c == ';' || c == '\r'; }

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t')) ++a;
    return s.substr(a);
}

std::string num(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.10g", x);
    return b;
}

bool dimensionless(const std::string& unit) { return unit.empty() || unit == "1" || unit == "-"; }

// axis names of a geometry in the canonical order (z, a, b); empty: unknown geometry
std::string axesOf(const std::string& geo) {
    if (geo == "z") return "z";
    if (geo == "rz") return "zr";
    if (geo == "xz") return "zx";
    if (geo == "xyz") return "zxy";
    return "";
}

std::vector<double>& axisOf(DensityFile& D, const std::string& names, char c) {
    const size_t k = names.find(c);
    return k == 0 ? D.z : k == 1 ? D.a : D.b;
}

struct Column {
    char kind = '?';  // 'z', 'r', 'x', 'y', 'n'
    std::string unit;
};

// "z[mm]  r [um], n(cm^-3)" -> columns; false if a name is not a coordinate (z, r, x, y) or a density
bool parseColumns(const std::string& s, std::vector<Column>& cols) {
    cols.clear();
    size_t p = 0;
    while (p < s.size()) {
        while (p < s.size() && isSep(s[p])) ++p;
        if (p >= s.size()) break;
        Column c;
        std::string name;
        while (p < s.size() && !isSep(s[p]) && s[p] != '[' && s[p] != '(') name += s[p++];
        size_t q = p;
        while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) ++q;
        if (q < s.size() && (s[q] == '[' || s[q] == '(')) {
            const char close = s[q] == '[' ? ']' : ')';
            const size_t e = s.find(close, q);
            if (e == std::string::npos) return false;
            c.unit = trim(s.substr(q + 1, e - q - 1));
            p = e + 1;
        }
        std::string low;
        for (char ch : name) low += char(std::tolower(static_cast<unsigned char>(ch)));
        if (low == "z" || low == "r" || low == "x" || low == "y") c.kind = low[0];
        else if (low == "density" || (!low.empty() && low[0] == 'n')) c.kind = 'n';
        else return false;
        cols.push_back(c);
    }
    return !cols.empty();
}

// checks of the finished grid, peak and its position
void finish(DensityFile& D) {
    const size_t na = D.na(), nb = D.nb();
    if (D.z.size() < 2 || (!D.a.empty() && D.a.size() < 2) || (!D.b.empty() && D.b.size() < 2))
        failAt(D.path, "at least 2 points per axis are needed");
    for (size_t i = 0; i < D.z.size(); ++i)
        for (size_t j = 0; j < na; ++j)
            for (size_t k = 0; k < nb; ++k) {
                const double x = D.v[(i * na + j) * nb + k];
                if (x > D.peak) {
                    D.peak = x; D.zPeak = D.z[i];
                    D.aPeak = D.a.empty() ? 0.0 : D.a[j];
                    D.bPeak = D.b.empty() ? 0.0 : D.b[k];
                }
            }
    if (!(D.peak > 0.0)) failAt(D.path, "the density is zero everywhere");
}

DensityFile readText(const std::string& path, const DensityFileUnits& u) {
    std::ifstream in(path);
    if (!in) failAt(path, "cannot open the file");
    DensityFile D;
    D.path = path;
    D.format = "text";
    // ---- header: comment lines "# key = value" before the data
    std::string columns, densityUnit, order, transverse;
    bool haveDensityUnit = false;
    struct AxisLine { char name; std::string text; int line; };
    std::vector<AxisLine> axes;
    std::vector<std::string> dataLines;
    std::vector<int> dataNo;
    std::string line;
    int ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        const std::string t = trim(line);
        if (t.empty()) continue;
        if (t[0] != '#') {
            dataLines.push_back(t);
            dataNo.push_back(ln);
            continue;
        }
        if (!dataLines.empty()) continue;           // comments between the data rows
        if (t.size() > 1 && t[1] == '#') continue;  // "## ...": free comment
        const size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(t.substr(1, eq - 1)), val = trim(t.substr(eq + 1));
        bool ident = !key.empty() && (std::isalpha(static_cast<unsigned char>(key[0])) || key[0] == '_');
        for (char c : key) ident = ident && (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
        if (!ident) continue;  // free text with an '=' in it
        const std::string where = "line " + std::to_string(ln) + ": ";
        if (key == "density_profile") {
            if (val != "1") failAt(path, where + "format version '" + val + "' (this reader knows version 1)");
        } else if (key == "geometry") D.geometry = val;
        else if (key == "columns") columns = val;
        else if (key == "density_unit") { densityUnit = val; haveDensityUnit = true; }
        else if (key == "order") order = val;
        else if (key == "transverse") transverse = val;
        else if (key == "z" || key == "r" || key == "x" || key == "y") axes.push_back({key[0], val, ln});
        else
            failAt(path, where + "unknown header key '" + key + "' (keys: density_profile, geometry, columns, "
                             "density_unit, z, r, x, y, order, transverse; write free comments without '=' or "
                             "start them with '##')");
    }
    if (D.geometry.empty()) failAt(path, std::string("no line '# geometry = ...' in the header; ") + HEADER_HELP);
    const std::string names = axesOf(D.geometry);  // canonical order z, a, b
    if (names.empty()) failAt(path, "unknown geometry '" + D.geometry + "' (z | rz | xz | xyz)");
    const size_t nd = names.size();
    if (dataLines.empty()) failAt(path, "no data");
    if (columns.empty() == axes.empty())
        failAt(path, columns.empty()
                         ? std::string("the header gives neither '# columns = ...' nor the axes of a grid; ") + HEADER_HELP
                         : "the header gives both '# columns' and axis lines ('# z = ...'): choose one layout");
    D.detail = "geometry " + D.geometry + (columns.empty() ? ", grid layout" : ", columns");
    auto numbers = [&](size_t k, std::vector<double>& out) {
        const char* p = dataLines[k].c_str();
        for (;;) {
            while (*p && isSep(*p)) ++p;
            if (!*p || *p == '#') break;
            char* end = nullptr;
            const double val = std::strtod(p, &end);
            const std::string where = "line " + std::to_string(dataNo[k]) + ": ";
            if (end == p) failAt(path, where + "not a number: '" + std::string(p).substr(0, 30) + "'");
            if (!std::isfinite(val)) failAt(path, where + "a value is not finite (nan or inf)");
            out.push_back(val);
            p = end;
        }
    };
    const std::string fit = " does not fit geometry " + D.geometry;
    try {
        if (!columns.empty()) {
            // ---- columns layout: coordinates and density per row; the grid is taken from the rows
            if (haveDensityUnit || !order.empty())
                failAt(path, "density_unit / order belong to the grid layout; with '# columns' the units stand in that line");
            std::vector<Column> cols;
            if (!parseColumns(columns, cols)) failAt(path, "cannot read '# columns = " + columns + "'; " + HEADER_HELP);
            int ic[3] = {-1, -1, -1}, in_ = -1;
            for (size_t k = 0; k < cols.size(); ++k) {
                if (cols[k].kind == 'n') {
                    if (in_ >= 0) failAt(path, "'# columns = " + columns + "': two density columns");
                    in_ = int(k);
                    continue;
                }
                const size_t d = names.find(cols[k].kind);
                if (d == std::string::npos || ic[d] >= 0) failAt(path, "'# columns = " + columns + "'" + fit);
                ic[d] = int(k);
            }
            for (size_t d = 0; d < nd; ++d)
                if (ic[d] < 0) failAt(path, "'# columns = " + columns + "'" + fit);
            if (in_ < 0) failAt(path, "'# columns = " + columns + "': no density column (n, n_e, density, ...)");
            D.nUnit = cols[size_t(in_)].unit;
            D.absolute = !dimensionless(D.nUnit);
            struct Row { double c[3]; double n; int line; };
            std::vector<Row> rows;
            std::vector<double> x;
            for (size_t k = 0; k < dataLines.size(); ++k) {
                x.clear();
                numbers(k, x);
                const std::string where = "line " + std::to_string(dataNo[k]) + ": ";
                if (x.size() != cols.size())
                    failAt(path, where + std::to_string(x.size()) + " numbers, '# columns' names " + std::to_string(cols.size()));
                if (x[size_t(in_)] < 0.0) failAt(path, where + "negative density " + num(x[size_t(in_)]));
                Row w{{0, 0, 0}, 0, dataNo[k]};
                for (size_t d = 0; d < nd; ++d) w.c[d] = u.length(x[size_t(ic[d])], cols[size_t(ic[d])].unit);
                w.n = D.absolute ? u.density(x[size_t(in_)], D.nUnit) : x[size_t(in_)];
                if (names.find('r') != std::string::npos && w.c[1] < 0.0) failAt(path, where + "negative radius");
                rows.push_back(w);
            }
            if (nd == 1) {
                for (size_t k = 0; k < rows.size(); ++k) {
                    if (k > 0 && !(rows[k].c[0] > rows[k - 1].c[0]))
                        failAt(path, "line " + std::to_string(rows[k].line) + ": z must be strictly increasing");
                    D.z.push_back(rows[k].c[0]);
                    D.v.push_back(rows[k].n);
                }
            } else {  // the rows fill the grid of the distinct coordinate values
                std::vector<double>* ax[3] = {&D.z, &D.a, &D.b};
                for (size_t d = 0; d < nd; ++d) {
                    for (const auto& w : rows) ax[d]->push_back(w.c[d]);
                    std::sort(ax[d]->begin(), ax[d]->end());
                    ax[d]->erase(std::unique(ax[d]->begin(), ax[d]->end()), ax[d]->end());
                }
                const size_t nz = D.z.size(), na = D.na(), nb = D.nb();
                if (rows.size() != nz * na * nb) {
                    std::string c;
                    for (size_t d = 0; d < nd; ++d)
                        c += (d ? " x " : "") + std::to_string(ax[d]->size()) + " distinct " + names[d];
                    failAt(path, "the rows do not fill a rectilinear grid: " + std::to_string(rows.size()) + " rows, but " +
                                     c + " = " + std::to_string(nz * na * nb) + " grid points (every point once)");
                }
                D.v.assign(nz * na * nb, -1.0);
                for (const auto& w : rows) {
                    size_t idx[3] = {0, 0, 0};
                    for (size_t d = 0; d < nd; ++d)
                        idx[d] = size_t(std::lower_bound(ax[d]->begin(), ax[d]->end(), w.c[d]) - ax[d]->begin());
                    double& cell = D.v[(idx[0] * na + idx[1]) * nb + idx[2]];
                    if (cell >= 0.0) failAt(path, "line " + std::to_string(w.line) + ": this grid point is given twice");
                    cell = w.n;
                }
            }
        } else {
            // ---- grid layout: the axes in the header, the values alone as data
            if (!haveDensityUnit)
                failAt(path, "the grid layout needs '# density_unit = ...' (cm^-3, m^-3; 1: a dimensionless factor)");
            D.nUnit = densityUnit;
            D.absolute = !dimensionless(densityUnit);
            std::string axisOrder;  // names, outer to inner: the last one varies fastest
            for (const auto& a : axes) {
                const std::string where = "line " + std::to_string(a.line) + ": ";
                if (names.find(a.name) == std::string::npos)
                    failAt(path, where + "axis '" + std::string(1, a.name) + "'" + fit);
                if (axisOrder.find(a.name) != std::string::npos) failAt(path, where + "axis given twice");
                axisOrder += a.name;
                // "uniform <origin> <step> <count> <unit>"  or  "list <x1> <x2> ... <unit>"
                std::vector<std::string> tok;
                std::string cur;
                for (char c : a.text + " ") {
                    if (isSep(c)) { if (!cur.empty()) tok.push_back(cur); cur.clear(); }
                    else cur += c;
                }
                const char* form = "'uniform <origin> <step> <count> <unit>' or 'list <x1> <x2> ... <unit>'";
                if (tok.size() < 4) failAt(path, where + "an axis is " + form);
                const std::string unit = tok.back();
                std::vector<double> xs;
                for (size_t k = 1; k + 1 < tok.size(); ++k) {
                    char* end = nullptr;
                    xs.push_back(std::strtod(tok[k].c_str(), &end));
                    if (end == tok[k].c_str() || *end || !std::isfinite(xs.back()))
                        failAt(path, where + "not a number: '" + tok[k] + "' (an axis is " + form + ")");
                }
                std::vector<double>& ax = axisOf(D, names, a.name);
                if (tok[0] == "uniform") {
                    if (xs.size() != 3 || xs[2] < 2 || xs[2] != std::floor(xs[2]) || !(xs[1] > 0))
                        failAt(path, where + "uniform axis: origin, step > 0 and a count >= 2, then the unit");
                    for (long k = 0; k < long(xs[2]); ++k) ax.push_back(u.length(xs[0] + double(k) * xs[1], unit));
                } else if (tok[0] == "list") {
                    if (xs.size() < 2) failAt(path, where + "a list axis needs at least 2 coordinates");
                    for (double x : xs) ax.push_back(u.length(x, unit));
                } else {
                    failAt(path, where + "an axis is " + form);
                }
                for (size_t k = 1; k < ax.size(); ++k)
                    if (!(ax[k] > ax[k - 1])) failAt(path, where + "the coordinates must be strictly increasing");
                if (a.name == 'r' && ax.front() < 0.0) failAt(path, where + "negative radius");
            }
            if (axisOrder.size() != nd) {
                std::string need;
                for (char c : names) need += std::string(need.empty() ? "" : ", ") + "'# " + c + " = ...'";
                failAt(path, "geometry " + D.geometry + " needs the axis lines " + need);
            }
            if (!order.empty()) {
                std::string o;
                for (char c : order)
                    if (!isSep(c)) o += c;
                std::string so = o, sn = names;
                std::sort(so.begin(), so.end());
                std::sort(sn.begin(), sn.end());
                if (so != sn) failAt(path, "'# order = " + order + "'" + fit + " (the axis names, e.g. 'z r')");
                axisOrder = o;
            }
            std::vector<double> x;
            for (size_t k = 0; k < dataLines.size(); ++k) numbers(k, x);
            const size_t n[3] = {D.z.size(), D.na(), D.nb()};
            if (x.size() != n[0] * n[1] * n[2])
                failAt(path, std::to_string(x.size()) + " values, but the axes of the header give " +
                                 std::to_string(n[0] * n[1] * n[2]) + " grid points");
            D.v.resize(x.size());
            size_t od[3] = {0, 0, 0};  // canonical axis index of each position of the order
            for (size_t p = 0; p < nd; ++p) od[p] = names.find(axisOrder[p]);
            size_t idx[3] = {0, 0, 0};
            for (idx[0] = 0; idx[0] < n[0]; ++idx[0])
                for (idx[1] = 0; idx[1] < n[1]; ++idx[1])
                    for (idx[2] = 0; idx[2] < n[2]; ++idx[2]) {
                        size_t flat = 0;
                        for (size_t p = 0; p < nd; ++p) flat = flat * n[od[p]] + idx[od[p]];
                        const double val = x[flat];
                        if (val < 0.0) failAt(path, "negative density " + num(val));
                        D.v[(idx[0] * n[1] + idx[1]) * n[2] + idx[2]] = D.absolute ? u.density(val, densityUnit) : val;
                    }
        }
    } catch (const std::invalid_argument& e) {
        failAt(path, std::string("units of the header: ") + e.what());
    }
    if (!transverse.empty()) {
        if (D.geometry != "z") failAt(path, "a transverse function is only for geometry z (the file is " + D.geometry + ")");
        transverse = trim(transverse);
        if (transverse.size() >= 2 && transverse.front() == '"' && transverse.back() == '"')
            transverse = transverse.substr(1, transverse.size() - 2);
        D.transverse = transverse;
    }
    finish(D);
    return D;
}

#if defined(DENSITY_FILE_USE_HDF5)
// ---------------------------------------------------------------- small read-only HDF5 helper
struct H {  // owning handle
    hid_t id;
    herr_t (*close)(hid_t);
    H(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
    H(const H&) = delete;
    H& operator=(const H&) = delete;
    ~H() { if (id >= 0) close(id); }
    operator hid_t() const { return id; }
};

struct File {
    explicit File(const std::string& p) : path(p), f(H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose) {
        if (f.id < 0) fail("cannot open the file (not HDF5?)");
    }
    std::string path;
    H f;
    [[noreturn]] void fail(const std::string& m) const { throw std::runtime_error(path + ": " + m); }
    bool exists(const std::string& p) const {  // every level (H5Lexists fails on a missing parent)
        size_t q = 1;
        for (;;) {
            q = p.find('/', q);
            const std::string sub = p.substr(0, q);
            if (!sub.empty() && H5Lexists(f, sub.c_str(), H5P_DEFAULT) <= 0) return false;
            if (q == std::string::npos) return true;
            ++q;
            if (q >= p.size()) return true;
        }
    }
    bool isGroup(const std::string& p) const {
        H o(H5Oopen(f, p.c_str(), H5P_DEFAULT), H5Oclose);
        return o.id >= 0 && H5Iget_type(o) == H5I_GROUP;
    }
    bool hasAttr(const std::string& obj, const char* name) const {
        return H5Aexists_by_name(f, obj.c_str(), name, H5P_DEFAULT) > 0;
    }
    std::vector<double> attrD(const std::string& obj, const char* name) const {
        H a(H5Aopen_by_name(f, obj.c_str(), name, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
        if (a.id < 0) fail("missing attribute " + obj + "@" + name);
        H sp(H5Aget_space(a), H5Sclose);
        std::vector<double> v(size_t(std::max<hssize_t>(1, H5Sget_simple_extent_npoints(sp))));
        if (H5Aread(a, H5T_NATIVE_DOUBLE, v.data()) < 0) fail("cannot read attribute " + obj + "@" + name);
        return v;
    }
    double attr1(const std::string& obj, const char* name, double def) const {
        return hasAttr(obj, name) ? attrD(obj, name)[0] : def;
    }
    // string attribute, scalar or array, fixed or variable length
    std::vector<std::string> attrSV(const std::string& obj, const char* name) const {
        std::vector<std::string> out;
        if (!hasAttr(obj, name)) return out;
        H a(H5Aopen_by_name(f, obj.c_str(), name, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
        if (a.id < 0) return out;
        H t(H5Aget_type(a), H5Tclose);
        if (H5Tget_class(t) != H5T_STRING) return out;
        H sp(H5Aget_space(a), H5Sclose);
        const size_t n = size_t(std::max<hssize_t>(1, H5Sget_simple_extent_npoints(sp)));
        if (H5Tis_variable_str(t) > 0) {
            std::vector<char*> s(n, nullptr);
            H mt(H5Tcopy(H5T_C_S1), H5Tclose);
            H5Tset_size(mt, H5T_VARIABLE);
            if (H5Aread(a, mt, s.data()) < 0) return out;
            for (size_t k = 0; k < n; ++k) {
                out.push_back(s[k] ? std::string(s[k]) : std::string());
                if (s[k]) H5free_memory(s[k]);
            }
            return out;
        }
        const size_t len = H5Tget_size(t);
        std::string buf(len * n, '\0');
        if (H5Aread(a, t, &buf[0]) < 0) return out;
        for (size_t k = 0; k < n; ++k) out.push_back(std::string(buf.data() + k * len, len).c_str());
        return out;
    }
    std::string attrS(const std::string& obj, const char* name) const {
        const std::vector<std::string> v = attrSV(obj, name);
        return v.empty() ? std::string() : v[0];
    }
    std::vector<hsize_t> dims(const std::string& p) const {
        H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
        if (d.id < 0) fail("no dataset " + p);
        H sp(H5Dget_space(d), H5Sclose);
        std::vector<hsize_t> v(size_t(std::max(0, H5Sget_simple_extent_ndims(sp))));
        H5Sget_simple_extent_dims(sp, v.data(), nullptr);
        return v;
    }
    std::vector<double> data(const std::string& p) const {
        H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
        if (d.id < 0) fail("no dataset " + p);
        H sp(H5Dget_space(d), H5Sclose);
        std::vector<double> v(size_t(H5Sget_simple_extent_npoints(sp)));
        if (!v.empty() && H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) < 0)
            fail("cannot read " + p);
        return v;
    }
    std::vector<std::string> children(const std::string& p) const {
        std::vector<std::string> out;
        H g(H5Gopen2(f, p.c_str(), H5P_DEFAULT), H5Gclose);
        if (g.id < 0) return out;
        H5G_info_t gi;
        H5Gget_info(g, &gi);
        for (hsize_t k = 0; k < gi.nlinks; ++k) {
            char buf[512];
            if (H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, k, buf, sizeof buf, H5P_DEFAULT) > 0)
                out.push_back(buf);
        }
        return out;
    }
};

DensityFile readOpenPMD(const std::string& path, const std::string& mesh, long iteration, const DensityFileUnits& u) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    File F(path);
    DensityFile D;
    D.path = path;
    D.format = "openPMD";
    if (!F.hasAttr("/", "openPMD")) F.fail("not an openPMD file (no attribute 'openPMD')");
    // iteration: the chosen one, or the last
    std::string base = F.attrS("/", "basePath");
    if (base.empty()) base = "/data/%T/";
    const std::string itRoot = base.substr(0, base.find("%T"));
    std::vector<long> its;
    for (const auto& c : F.children(itRoot.empty() ? "/" : itRoot)) {
        char* end = nullptr;
        const long v = std::strtol(c.c_str(), &end, 10);
        if (end != c.c_str() && *end == '\0') its.push_back(v);
    }
    if (its.empty()) F.fail("no iteration found under " + itRoot);
    std::sort(its.begin(), its.end());
    long it = its.back();
    if (iteration >= 0) {
        if (!std::binary_search(its.begin(), its.end(), iteration))
            F.fail("iteration " + std::to_string(iteration) + " is not in the file");
        it = iteration;
    }
    std::string mp = F.attrS("/", "meshesPath");
    if (mp.empty()) mp = "meshes/";
    if (mp.back() != '/') mp += '/';
    const std::string mroot = itRoot + std::to_string(it) + "/" + mp;
    const std::vector<std::string> all = F.children(mroot);
    std::string name = mesh;
    if (name.empty()) {
        if (F.exists(mroot + "density")) name = "density";
        else if (all.size() == 1) name = all[0];
    }
    if (name.empty() || !F.exists(mroot + name)) {
        std::string l;
        for (const auto& c : all) l += (l.empty() ? "" : ", ") + c;
        F.fail((name.empty() ? std::string("no mesh 'density'") : "no mesh '" + name + "'") + " in iteration " +
               std::to_string(it) + " (meshes: " + (l.empty() ? "none" : l) + "); give the mesh name");
    }
    const std::string p = mroot + name;
    if (F.isGroup(p)) F.fail("mesh '" + name + "' is not a scalar record (a density is one dataset)");
    D.detail = "mesh '" + name + "', iteration " + std::to_string(it);
    const bool theta = F.attrS(p, "geometry") == "thetaMode" || F.attrS(p, "geometryParameters").rfind("thetaMode", 0) == 0;
    std::vector<std::string> lab = F.attrSV(p, "axisLabels");
    const size_t nd = lab.size();
    std::vector<double> sp(nd, 1.0), off(nd, 0.0), pos(nd, 0.0);
    std::vector<std::string> gc(nd);
    auto take = [&](const char* a, std::vector<double>& v, bool required) {
        if (!F.hasAttr(p, a)) {
            if (required) F.fail("mesh '" + name + "' has no attribute " + a);
            return;
        }
        v = F.attrD(p, a);
        if (v.size() != nd) F.fail("mesh '" + name + "': " + a + " must have one entry per axis of axisLabels");
    };
    if (F.hasAttr(p, "gridCoordinates")) {
        gc = F.attrSV(p, "gridCoordinates");
        if (gc.size() != nd) F.fail("mesh '" + name + "': gridCoordinates must have one entry per axis of axisLabels");
    }
    bool anyRegular = false;
    for (const auto& s : gc) anyRegular = anyRegular || s.empty();
    take("gridSpacing", sp, anyRegular);
    take("gridGlobalOffset", off, false);
    take("position", pos, false);
    if (F.attrS(p, "dataOrder") == "F") {
        std::reverse(lab.begin(), lab.end());
        std::reverse(sp.begin(), sp.end());
        std::reverse(off.begin(), off.end());
        std::reverse(pos.begin(), pos.end());
        std::reverse(gc.begin(), gc.end());
    }
    std::string labels;
    for (const auto& c : lab) labels += (labels.empty() ? "" : ", ") + c;
    // geometry from the axes
    for (const char* g : {"z", "rz", "xz", "xyz"}) {
        const std::string names = axesOf(g);
        bool ok = names.size() == nd;
        for (char c : names) ok = ok && std::find(lab.begin(), lab.end(), std::string(1, c)) != lab.end();
        if (ok) D.geometry = g;
    }
    if (D.geometry.empty())
        F.fail("mesh '" + name + "': the axes must be z; r, z; x, z; or x, y, z (axisLabels: " + labels + ")");
    const std::string names = axesOf(D.geometry);
    const std::vector<hsize_t> dims = F.dims(p);
    const size_t lead = (theta && dims.size() == nd + 1) ? 1 : 0;  // thetaMode: the first index is the mode
    if (dims.size() != nd + lead)
        F.fail("mesh '" + name + "': " + std::to_string(dims.size()) + " dimensions for the axes " + labels);
    D.detail += ", geometry " + D.geometry;
    const double gu = F.attr1(p, "gridUnitSI", 1.0);
    size_t dimOf[3] = {0, 0, 0};  // dataset dimension (without the mode index) of each canonical axis
    try {
        for (size_t d = 0; d < nd; ++d) {
            const size_t k = size_t(std::find(lab.begin(), lab.end(), std::string(1, names[d])) - lab.begin());
            dimOf[d] = k;
            const size_t n = size_t(dims[lead + k]);
            std::vector<double>& x = d == 0 ? D.z : d == 1 ? D.a : D.b;
            x.resize(n);
            if (!gc[k].empty()) {  // explicit coordinates, relative to gridGlobalOffset
                const std::vector<double> c = F.data(gc[k]);
                if (c.size() != n)
                    F.fail("coordinates " + gc[k] + ": " + std::to_string(c.size()) + " values for " + std::to_string(n) +
                           " data points along " + lab[k]);
                for (size_t j = 0; j < n; ++j) x[j] = u.length((off[k] + c[j]) * gu, "m");
            } else {
                for (size_t j = 0; j < n; ++j) x[j] = u.length((off[k] + (double(j) + pos[k]) * sp[k]) * gu, "m");
            }
            for (size_t j = 0; j < n; ++j) {
                if (!std::isfinite(x[j])) F.fail("coordinates along " + lab[k] + " are not finite");
                if (j > 0 && !(x[j] > x[j - 1])) F.fail("coordinates along " + lab[k] + " must be strictly increasing");
            }
            if (names[d] == 'r' && x.front() < 0.0) F.fail("mesh '" + name + "': negative radius");
        }
        // unit: number density (L^-3) or dimensionless
        const double uSI = F.attr1(p, "unitSI", 1.0);
        std::vector<double> ud(7, 0.0);
        if (F.hasAttr(p, "unitDimension")) ud = F.attrD(p, "unitDimension");
        ud.resize(7, 0.0);
        bool zero = true, rest = true;
        for (size_t k = 0; k < 7; ++k) {
            zero = zero && ud[k] == 0.0;
            if (k > 0) rest = rest && ud[k] == 0.0;
        }
        if (zero) {
            D.absolute = false;
            D.nUnit = "1";
        } else if (ud[0] == -3.0 && rest) {
            D.absolute = true;
            D.nUnit = "m^-3";
        } else {
            F.fail("mesh '" + name + "': unitDimension is neither a number density (L^-3) nor dimensionless "
                   "(a charge or mass density must be converted to a number density first)");
        }
        const std::vector<double> raw = F.data(p);
        const size_t n[3] = {D.z.size(), D.na(), D.nb()};
        size_t dn[3] = {1, 1, 1};  // extents of the dataset dimensions
        for (size_t k = 0; k < nd; ++k) dn[k] = size_t(dims[lead + k]);
        D.v.resize(n[0] * n[1] * n[2]);
        size_t idx[3] = {0, 0, 0};
        for (idx[0] = 0; idx[0] < n[0]; ++idx[0])
            for (idx[1] = 0; idx[1] < n[1]; ++idx[1])
                for (idx[2] = 0; idx[2] < n[2]; ++idx[2]) {
                    size_t di[3] = {0, 0, 0};  // index along each dataset dimension
                    for (size_t d = 0; d < nd; ++d) di[dimOf[d]] = idx[d];
                    size_t flat = 0;  // mode 0 of a thetaMode record: the leading index 0
                    for (size_t k = 0; k < nd; ++k) flat = flat * dn[k] + di[k];
                    const double val = raw[flat] * uSI;
                    if (!std::isfinite(val)) F.fail("mesh '" + name + "': a value is not finite (nan or inf)");
                    if (val < 0.0) F.fail("mesh '" + name + "': negative density " + num(raw[flat]));
                    D.v[(idx[0] * n[1] + idx[1]) * n[2] + idx[2]] = D.absolute ? u.density(val, "m^-3") : val;
                }
    } catch (const std::invalid_argument& e) {
        F.fail(std::string("units: ") + e.what());
    }
    if (F.hasAttr(p, "transverseFunction")) {
        if (D.geometry != "z") F.fail("attribute transverseFunction is only for a mesh with the axis z");
        D.transverse = trim(F.attrS(p, "transverseFunction"));
    }
    if (lead && dims[0] > 1) D.detail += ", mode m = 0 of " + std::to_string((dims[0] + 1) / 2) + " modes";
    finish(D);
    return D;
}
#endif

// interval and weight of x on the axis; false: outside (and not held). clampLow: below the first point its value
bool locate(const std::vector<double>& ax, double x, bool hold, bool clampLow, size_t& i, double& t) {
    if (x < ax.front()) {
        if (!hold && !clampLow) return false;
        x = ax.front();
    } else if (x > ax.back()) {
        if (!hold) return false;
        x = ax.back();
    }
    if (x >= ax.back()) { i = ax.size() - 2; t = 1.0; return true; }
    i = size_t(std::upper_bound(ax.begin(), ax.end(), x) - ax.begin()) - 1;
    t = (x - ax[i]) / (ax[i + 1] - ax[i]);
    return true;
}

}  // namespace

double DensityFile::at(double zz, double aa, double bb, bool hold) const {
    size_t i = 0, j = 0, k = 0;
    double tz = 0.0, ta = 0.0, tb = 0.0;
    if (!locate(z, zz, hold, false, i, tz)) return 0.0;
    if (a.empty()) {  // the table formula: v_i + (v_i+1 - v_i) t
        if (tz == 1.0) return v[i + 1];
        return v[i] + (v[i + 1] - v[i]) * tz;
    }
    if (!locate(a, aa, hold, geometry == "rz", j, ta)) return 0.0;
    const size_t n1 = a.size(), n2 = nb();
    auto lineA = [&](size_t iz, size_t kb) {
        const double p = v[(iz * n1 + j) * n2 + kb], q = v[(iz * n1 + j + 1) * n2 + kb];
        return p + (q - p) * ta;
    };
    if (b.empty()) {
        const double p = lineA(i, 0), q = lineA(i + 1, 0);
        return p + (q - p) * tz;
    }
    if (!locate(b, bb, hold, false, k, tb)) return 0.0;
    auto plane = [&](size_t iz) {
        const double p = lineA(iz, k), q = lineA(iz, k + 1);
        return p + (q - p) * tb;
    };
    const double p = plane(i), q = plane(i + 1);
    return p + (q - p) * tz;
}

bool densityFileHasHDF5() {
#if defined(DENSITY_FILE_USE_HDF5)
    return true;
#else
    return false;
#endif
}

DensityFile readDensityFile(const std::string& path, const std::string& mesh, long iteration,
                            const DensityFileUnits& units) {
    bool hdf5 = false;  // HDF5 signature \211HDF\r\n\032\n
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) failAt(path, "cannot open the file");
        char sig[8] = {0};
        in.read(sig, 8);
        hdf5 = in.gcount() == 8 && std::string(sig, 8) == std::string("\211HDF\r\n\032\n", 8);
    }
    if (!hdf5) {
        if (!mesh.empty() || iteration >= 0)
            failAt(path, "a mesh name / iteration is for openPMD (HDF5) files; this is a text table");
        return readText(path, units);
    }
#if defined(DENSITY_FILE_USE_HDF5)
    return readOpenPMD(path, mesh, iteration, units);
#else
    (void)mesh; (void)iteration;
    failAt(path, "reading openPMD density files needs a build with HDF5; or give the profile as a text table");
#endif
}

}  // namespace densfile
