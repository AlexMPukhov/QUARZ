---
name: quarz-setup
description: Install, build and test the plasma wakefield code QUARZ (quasi-static particle-in-cell, RZ geometry with a non-uniform radial grid, C++17 with Kokkos) and choose how to run it - OpenMP threads, MPI ranks or a GPU. Use when the executable quarz is not installed yet, when a build or a test fails, when the user asks how to run QUARZ on several cores, a cluster or a GPU, or where its output files go. To write or debug an input file use the skill quarz-simulation.
license: BSD-3-Clause
compatibility: Needs git, cmake >= 3.18 and a C++17 compiler (GCC >= 9 or Clang >= 9) on Linux, macOS or WSL; Kokkos is downloaded by cmake if it is not installed. Optional - MPI, HDF5, openPMD-api, a CUDA or HIP toolkit, python3 with numpy.
metadata:
  repository: https://github.com/AlexMPukhov/QUARZ
---

# QUARZ: build, test, run

QUARZ is one executable, `quarz`, that reads one input file of `key = value` lines. There is nothing to install
besides the build. Tell the user what you are going to do before you clone or build. Build and run the code as it
is: do not modify its source; a failing test or a missing feature is reported to the user (and as an issue at
https://github.com/AlexMPukhov/QUARZ/issues), not patched.

## 1. Is it there already?

Look for `quarz` in the PATH or `<repo>/build/quarz`. Started without arguments it prints
`usage: [mpirun -np P] quarz <input file> [key=value ...]`. If it is there, go to section 4 (use the path of that
executable wherever this skill writes `build/quarz`).

## 2. Clone and build

```
git clone https://github.com/AlexMPukhov/QUARZ.git
cd QUARZ
cmake -B build -DKokkos_ENABLE_OPENMP=ON && cmake --build build -j
```

- **Kokkos:** if cmake does not find an installed Kokkos (>= 4.0; `-DKokkos_ROOT=<path>`), it downloads Kokkos
  4.4.01 and builds it with the options given (needs network access; a few minutes).
- **Compiler on clusters:** cmake takes `c++` from the PATH, often an old system compiler even with a newer module
  loaded. Configure a fresh build directory with `CXX=$(which g++) CC=$(which gcc) cmake -B build ...` and check the
  line `The CXX compiler identification is ...`. GCC < 9 is refused.
- **Found automatically, optional:** MPI (`-- QUARZ: MPI enabled`; `-DQUARZ_ENABLE_MPI=OFF` switches it off), HDF5
  (LASY laser files), openPMD-api >= 0.15 (`-DopenPMD_ROOT=<path>`; openPMD output). A missing optional library is
  not an error; the feature is then refused at run time.
- **Windows:** build and run inside WSL (Ubuntu); the commands are the same.
- **GPU:** `tools/build_gpu.sh` finds or installs NVIDIA's compiler in user space, detects the GPU, builds Kokkos
  and QUARZ into `build-gpu/` and runs the tests. By hand: `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<arch>=ON` with
  Kokkos' `nvcc_wrapper` as the compiler; AMD: `-DKokkos_ENABLE_HIP=ON`. Details: section 4 of the repository's
  README.md.

## 3. Test

```
cd build && ctest --output-on-failure                 # unit tests and the agent-skill check, about 1 min
cd ../validation && ./run_validation.sh -q -j 2 -c    # physics regression against references, a few minutes
```

- In containers and WSL set `OMP_PROC_BIND=false`. The validation suite needs python3 with numpy and scipy.
- Validation sections that need a missing optional library (openPMD-api) fail for that reason only; a Monte-Carlo
  section can differ within its stated statistical error with another compiler. Report other failures to the user
  with the output; do not edit the tests or the reference files.

## 4. Run

```
build/quarz run.in time.steps=0 output.dir=check_out    # one quasi-static solve: the quickest check of an input
OMP_NUM_THREADS=8 build/quarz run.in                    # CPU threads
mpirun -np 8 build/quarz run.in                         # MPI: 8 blocks of slices along xi, time steps pipelined
build/quarz run.in grid.rmax=12 output.dir=scan_a       # any input key can be overridden: key=value
build/quarz run.in restart.from=latest time.steps=500   # continue from the newest checkpoint (checkpoint.every)
```

| how | set by | use it for |
|---|---|---|
| OpenMP | `OMP_NUM_THREADS` (build with `-DKokkos_ENABLE_OPENMP=ON`) | a workstation: parallel over the plasma particles of a slice and over the beam |
| MPI | `mpirun -np P` | long boxes with many time steps: P blocks of slices, pipelined over the steps. A single quasi-static solve (`time.steps = 0`) gets no speed-up. MPI runs use `beams.xi_shape = ngp` |
| GPU | a GPU build; one GPU per MPI rank | many particles per slice: fine radial grids, `modes = 1`, ionization |

- **Threads:** in containers, WSL and on shared nodes set `OMP_PROC_BIND=false`; on dedicated cluster nodes
  `OMP_PROC_BIND=close OMP_PLACES=cores` (Slurm: `--cpus-per-task`, `--cpu-bind=cores`). Unpinned or doubly pinned
  threads can make a run several times slower.
- **How many MPI ranks:** the ranks form a pipeline over the time steps, which needs P - 1 steps to fill. With N
  steps the efficiency is about N/(N + P - 1): keep P below about N/4. Under Slurm request `--ntasks` >= P,
  otherwise all ranks share one core.

There is no check-only switch. Unknown keys do not stop a run: they are listed at its end as
`WARNING: input key '...' was not used`. Errors in values stop it with `ERROR (rank 0): ...`. The script
`scripts/check_input.sh` of the skill quarz-simulation runs the zero-step solve and turns both into a failure.

## 5. Where the output goes

Everything is written to `output.dir` (default `out`), relative to the directory where `quarz` is started:
`grid.txt` (the radial grid), `beams.txt` (moments of every beam, one line per step), and every `output.every`
steps `axis_NNNNNN.txt` (on-axis fields) and `fields_NNNNNN.bin` (2D maps; `tools/quarz_read.py` reads them),
every `output.beam_every` steps `beam_<name>_NNNNNN.bin`. With `output.format = openpmd` the maps and particles
go to `openpmd/data_NNNNNN.h5` instead. A field file holds (slices) x (radial nodes) x (components) x 8 bytes:
estimate it before a long run and thin it with `output.fields`, `output.rmax`, `output.xi_stride` or
`output.field_files = 0`.

Other tools in the repository: `python3 tools/plot_fields.py out/fields_000000.bin [rmax]` (maps as PNG; needs
matplotlib), `python3 tools/quarz_gui.py` (input editor, run control and plots in the browser).

## 6. Cluster jobs

Never run simulations on a login node. Build there, then submit. For a wall-time limit use checkpoints:
`checkpoint.every = 100` in the input, and resubmit with `restart.from=latest` and the full `time.steps`.

## Citing

A. Pukhov and T. C. Wilson, *QUARZ: a quasi-static particle-in-cell code with non-uniform radial grids for plasma
wakefields*, arXiv:2610.16910 (2026); software: doi:10.5281/zenodo.23472094.
