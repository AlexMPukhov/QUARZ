// SimulationIOInternal.hpp: helpers of the output groups and checkpoints (SimulationIO.cpp, SimulationCheckpoint.cpp)
#pragma once
#include "Simulation.hpp"
#include "Beam.hpp"
#include "Config.hpp"
#include "Laser.hpp"
#include "OpenPMDWriter.hpp"
#include "RadialGrid.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace quarz {
namespace detail {
namespace fs = std::filesystem;

constexpr uint64_t CHK_MAGIC = 0x314B484352415551ULL;   // "QUARCHK1"
constexpr int32_t CHK_VERSION = 1;

inline std::string step_name(int n) {
    char b[16];
    std::snprintf(b, sizeof(b), "%06d", n);
    return b;
}

// complete checkpoints in dir: step -> path
inline std::map<int, std::string> complete_checkpoints(const std::string& dir) {
    std::map<int, std::string> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string leaf = e.path().filename().string();
        if (leaf.rfind("chk_", 0) != 0 || !e.is_directory()) continue;
        std::ifstream meta(e.path() / "checkpoint.txt");
        std::string key;
        int step = -1, ranks = -1;
        while (meta >> key) {
            if (key == "step") meta >> step;
            else if (key == "ranks") meta >> ranks;
        }
        if (step < 0 || ranks < 1) continue;
        int n = 0;
        for (int r = 0; r < ranks; ++r)
            if (fs::exists(e.path() / ("rank_" + std::to_string(r) + ".bin"), ec)) ++n;
        if (n == ranks) out[step] = e.path().string();
    }
    return out;
}

struct Writer {
    std::ofstream o;
    template <class T> void put(const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    void put(const std::vector<double>& v) {
        put(static_cast<int64_t>(v.size()));
        o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(8 * v.size()));
    }
    void put(const std::string& s) {
        put(static_cast<int32_t>(s.size()));
        o.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
};
struct Reader {
    std::ifstream i;
    std::string fn;
    template <class T> T get() {
        T v;
        i.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!i) throw std::runtime_error("checkpoint file " + fn + " is truncated");
        return v;
    }
    std::vector<double> vec() {
        const int64_t n = get<int64_t>();
        std::vector<double> v(static_cast<size_t>(n));
        i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(8 * n));
        if (!i) throw std::runtime_error("checkpoint file " + fn + " is truncated");
        return v;
    }
    std::string str() {
        const int32_t n = get<int32_t>();
        std::string s(static_cast<size_t>(n), ' ');
        i.read(&s[0], n);
        return s;
    }
};

// one rank's checkpoint file
struct ChkFile {
    int P = 0, rank = 0, step = 0, K = 0, k0 = 0, nloc = 0, M = 0, m1 = 0;
    double t = 0, dxi = 0, xi_min = 0;
    int64_t lost = 0;
    std::vector<std::string> names;
    std::vector<int> started;
    std::vector<std::vector<double>> particles, outbox;
    bool laser = false;
    std::vector<double> env;   // (nloc, M, 2)
};

inline ChkFile read_chk(const std::string& fn) {
    Reader R;
    R.fn = fn;
    R.i.open(fn, std::ios::binary);
    if (!R.i) throw std::runtime_error("cannot open checkpoint file " + fn);
    ChkFile c;
    if (R.get<uint64_t>() != CHK_MAGIC) throw std::runtime_error(fn + " is not a QUARZ checkpoint");
    if (R.get<int32_t>() != CHK_VERSION) throw std::runtime_error(fn + ": unsupported checkpoint version");
    c.P = R.get<int32_t>(); c.rank = R.get<int32_t>(); c.step = R.get<int32_t>();
    c.K = R.get<int32_t>(); c.k0 = R.get<int32_t>(); c.nloc = R.get<int32_t>(); c.M = R.get<int32_t>();
    c.m1 = R.get<int32_t>();
    const int nb = R.get<int32_t>();
    c.laser = R.get<int32_t>() != 0;
    c.t = R.get<double>(); c.dxi = R.get<double>(); c.xi_min = R.get<double>();
    c.lost = R.get<int64_t>();
    for (int b = 0; b < nb; ++b) {
        c.names.push_back(R.str());
        c.started.push_back(R.get<int32_t>());
        c.particles.push_back(R.vec());
        c.outbox.push_back(R.vec());
    }
    if (c.laser) c.env = R.vec();
    return c;
}

} // namespace detail
} // namespace quarz
