# QUARZ — notes for Claude sessions

Development rules for Claude sessions working on QUARZ. README.md is the user documentation
(physics, numerics, inputs, outputs, validation).

## Workflow rules
- **Start of a session:** if the private notes repository (AlexMPukhov/CLAUDE-notes) is in this
  session, read its `quarz/CLAUDE.md` and `quarz/handbook.md` first: they hold the project status,
  plans and working rules. Then `git status` and `git log --oneline | head`.
  Fresh container: `tools/dev/setup_container.sh` rebuilds Kokkos, openPMD-api and QUARZ.
  Uncommitted changes = an earlier session was interrupted: inspect them (`git diff`) and verify
  before building on them.
- **At every milestone** (a feature works, a bug is fixed, validation passes): commit with a
  descriptive message and push (`git push origin main`); update the private notes in the same turn.
- Never report a feature as done without running the relevant validation section.
- This repository is PUBLIC: no unpublished results, collaborator data or plans here. Unsubmitted
  papers that are not directly about the code go to the private repository CLAUDE-notes.

## Project
**Name:** QUARZ = Quasistatic Arbitrary-resolution RZ code (formerly QSRZ), nickname "Q".
Executable `quarz`, namespace `quarz`, macros/CMake options
`QUARZ_*`. Old names still accepted: CMake `-DQSRZ_*` options (deprecation warning),
`tools/qsrz_read.py` (shim), `QSRZ_*` variables in `tools/build_gpu.sh`. Input keys and output
formats unchanged (openPMD attributes renamed `qsrz_*` -> `quarz_*`; no reader depends on them).
Quasi-static axisymmetric (m = 0, optional m = 1) PIC code for plasma wakefield
acceleration, LCODE-like, written in C++17 with Kokkos (OpenMP / CUDA / HIP), MPI
decomposition along xi with pipelined time steps, optional openPMD output.
Distinctive: arbitrary non-uniform radial grid (finite-element Laplacian on hat functions),
explicit B_perp solve, Adams–Bashforth plasma push, several beam pushers. Paper: arXiv:2610.16910.

Code map: README §10. Key files: `src/Simulation.cpp` (sweep, MPI hand-off, output),
`src/PlasmaSpecies.cpp` (deposit, push, ionization kernel), `src/Beam.cpp`,
`src/FieldSolver.cpp`, `src/Laser.cpp` (envelope solver), `src/Ionization.*` (ADK, Bethe,
element table, counter-based RNG).

## Build and test (this container)
```bash
cmake -B build -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ROOT=/home/claude/kokkos-install -DopenPMD_ROOT=/home/claude/openpmd-install
cmake --build build -j2 && (cd build && ctest)
export OMP_PROC_BIND=false            # needed here, otherwise OpenMP runs badly
cd validation && ./run_validation.sh -q -j 2 -c       # quick regression, ~2 min: after EVERY code change
cd validation && ./run_validation.sh -j 2 -c          # full suite (sections 1-25), ~10-15 min: milestones / new physics
```
CUDA compile check (no GPU here):
`PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/bin:$PATH CUDA_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/cu13 make -C build-cuda -j2 quarz`

MPI tests: `mpirun --allow-run-as-root --oversubscribe -np P -x OMP_NUM_THREADS=1 ...`,
compare with `validation/cmp_runs.py serial_dir mpi_dir 1e-6`. Serial reference needs
`beams.xi_shape=ngp` for bit-identity.

Package for delivery (includes .git, so the history survives the container):
`cd /home/claude && zip -qr quarz.zip quarz -x 'quarz/build/*' 'quarz/build-*/*' 'quarz/gpu-deps/*' '*/__pycache__/*' 'quarz/validation/out_*' 'quarz/validation/q_*' 'quarz/validation/d_*'`

## Invariants (do not break)
- MPI results bit-identical to serial (with ngp beam shape) unless beam particles move
  between ranks. Particle creation uses prefix sums for indices; random numbers come from
  a counter-based hash of (seed, step, slice, species, particle, draw), never from
  thread-dependent generators.
- E_z is the exact discrete xi-derivative of psi; Gauss's law holds in the laser region.
- Every new feature gets a validation section in `run_validation.sh` against an independent
  (analytic or Python) reference, a quick variant (`$(q FULL QUICK)`), entries in both
  reference files (`-u` after checking the numbers), and a row in README §7.

## Validation policy (keep it fast)
- After a code change: quick suite with `-c` (2 min). Only the sections the change can affect
  if it is clearly local (laser -> 11, ionization -> 12, MPI/output -> 9 10, m = 1 -> 6 7,
  grid/solver -> 1 2 5, plasma push / sub-slicing -> 13, smoothing -> 14, checkpoints / output groups -> 15, time step -> 16, GUI -> 17, Coulomb scattering -> 18, two-stage runs -> 19, spin -> 20, trapping -> 21). Full suite with `-c` at milestones and before delivering physics results.
- Run long suites in the background (`nohup ... &`) and poll the process, not `pgrep -f`
  with a pattern that matches the polling shell itself.
- CUDA compile check (~10 min) only when device code changed (kernels, Types.hpp, views).
- Never edit `run_validation.sh` while it is running (bash reads scripts incrementally).
