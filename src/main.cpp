#include "Config.hpp"
#include "Parallel.hpp"
#include "Simulation.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

// Every string literal of the sources that looks like a key or a part of one ("xi.step", ".sigma_r", "fields"),
// collected by CMake (KnownKeys.inc). The code builds most keys from an object name and such a part, so a key of
// the input is known if it is one of these literals, ends with a literal that starts with '.', or its last
// component is a literal.
static const char* const known_key_literals[] = {
#include "KnownKeys.inc"
};

static bool known_key(std::string key) {
    const auto par = key.find('(');                        // plasma.density(x,y,z) -> plasma.density
    if (par != std::string::npos) key = key.substr(0, par);
    const auto dot = key.rfind('.');
    const std::string last = dot == std::string::npos ? key : key.substr(dot + 1);
    for (const char* lit : known_key_literals) {
        const std::string s = lit;
        if (s == key || s == last) return true;
        if (s[0] == '.' && key.size() > s.size() && key.compare(key.size() - s.size(), s.size(), s) == 0) return true;
    }
    return false;
}

// number of GPUs visible to this process (0 for host backends)
static int visible_devices() {
    int n = 0;
#if defined(KOKKOS_ENABLE_CUDA)
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
#elif defined(KOKKOS_ENABLE_HIP)
    if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
#endif
    return n;
}

int main(int argc, char* argv[]) {
    // MPI first (Kokkos must be initialised after MPI and finalised before it)
    quarz::Comm::init(&argc, &argv);
    quarz::Comm& comm = quarz::Comm::world();
    int rc = 0;
    {
        // one GPU per rank: device = (rank within the node) mod (#GPUs), unless the user
        // chose one (--kokkos-device-id=..., KOKKOS_DEVICE_ID, or CUDA_VISIBLE_DEVICES per rank)
        std::vector<char*> args(argv, argv + argc);
        std::string devarg;
        bool user_dev = std::getenv("KOKKOS_DEVICE_ID") != nullptr;
        for (int i = 1; i < argc; ++i)
            if (std::strncmp(argv[i], "--kokkos-device-id", 18) == 0 || std::strncmp(argv[i], "--kokkos-map-device-id-by", 25) == 0)
                user_dev = true;
        const int ndev = visible_devices();
        if (!user_dev && ndev > 0) {
            devarg = "--kokkos-device-id=" + std::to_string(comm.local_rank() % ndev);
            args.push_back(devarg.data());
        }
        args.push_back(nullptr);
        int kargc = static_cast<int>(args.size()) - 1;
        Kokkos::ScopeGuard kokkos(kargc, args.data());

        if (argc < 2) {
            if (comm.root()) std::cerr << "usage: [mpirun -np P] quarz <input file> [key=value ...] [--check] [--restart <checkpoint dir | latest>] [--kokkos-... options]\n";
            rc = 1;
        } else {
            try {
                quarz::Config cfg(argv[1]);
                // command line: key=value overrides, --check (read the input, set everything up, stop before the
                // first step), --restart <checkpoint dir | latest> (= restart.from)
                bool check = false;
                for (int i = 2; i < argc; ++i) {
                    std::string a = argv[i];
                    if (a == "--check") { check = true; continue; }
                    if (a == "--restart") {
                        if (i + 1 >= argc) throw std::runtime_error("--restart needs a checkpoint directory or 'latest'");
                        cfg.set("restart.from", argv[++i]);
                        continue;
                    }
                    auto eq = a.find('=');
                    if (a.rfind("--", 0) == 0 || eq == std::string::npos) continue;
                    cfg.set(a.substr(0, eq), a.substr(eq + 1));
                }
                if (comm.root() && comm.size() > 1)
                    std::cout << "MPI: " << comm.size() << " ranks (decomposition along xi)"
                              << (ndev > 0 ? ", " + std::to_string(ndev) + " GPU(s) per node" : std::string()) << "\n";
                {
                    quarz::Simulation sim(cfg);
                    // Every key is read while the run is set up. A key that nobody read and that the code does not
                    // know at all is misspelt: stop before the first step (the run would silently use a default).
                    // A known key that this setup does not read (the shape of a bunch that comes from a file, a
                    // species taken out of plasma.species on the command line) is reported, now and not after the run.
                    cfg.n0_cm3();                          // the reference density may serve the units only
                    const bool allow = cfg.get_bool("input.allow_unknown", false);
                    std::string unknown;
                    for (const auto& k : cfg.unused_keys()) {
                        if (known_key(k)) {
                            if (comm.root()) std::cout << "WARNING: input key '" << k << "' was not used\n";
                        } else
                            unknown += " '" + k + "'";
                    }
                    if (!unknown.empty()) {
                        if (!allow) throw std::runtime_error("unknown input key(s):" + unknown + " (misspelt?)");
                        if (comm.root()) std::cout << "WARNING: unknown input key(s):" << unknown << "\n";
                    }
                    if (check) {
                        if (comm.root()) std::cout << "check: input ok (" << cfg.keys().size() << " keys), no step done\n";
                    } else
                        sim.run();
                }
            } catch (const std::exception& e) {
                std::cerr << "ERROR (rank " << comm.rank() << "): " << e.what() << "\n";
                rc = 2;
            }
        }
    }
    if (rc == 2 && comm.size() > 1) quarz::Comm::abort(2);   // other ranks may wait in a receive
    quarz::Comm::finalize();
    return rc;
}
