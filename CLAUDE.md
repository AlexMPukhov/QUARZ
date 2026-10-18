# QUARZ — notes for Claude sessions

Read this first in every session. It records the project state, so work does not depend on
conversation history (which can be lost, e.g. when a turn is cut off). README.md is the full
user documentation (physics, numerics, inputs, outputs, validation); this file is the
working state and the rules for developing.

## Workflow rules
- **Start of a session:** read this file, then `git status` and `git log --oneline | head`.
  Uncommitted changes = an earlier session was interrupted: inspect them (`git diff`) and
  verify before building on them; tell Alexander what was found.
- **At every milestone** (a feature works, a bug is fixed, validation passes): commit with a
  descriptive message and update "Status" / "Open issues" below in the same commit.
  Commit unfinished work too, as `WIP: ...`, before long runs or at the end of a turn.
- Never report a feature as done without running the relevant validation section.
- **Remote:** `origin` = https://github.com/AlexMPukhov/QUARZ (PUBLIC; renamed from QSRZ3D on
  2026-10-06, GitHub redirects the old URL).
  Push after every commit (`git push origin main`); a new session starts by cloning it into
  `/home/claude/quarz` (`git clone <url> quarz`). Commit email:
  alex.m.pukhov@gmail.com (GitHub account of Alexander).
- Deliverable to Alexander: `quarz.zip` of the repo (no build dirs, no outputs), see below.

## Project
**Name:** QUARZ = Quasistatic Arbitrary-resolution RZ code (chosen by Alexander and Thomas,
2026-10-06; formerly QSRZ). Nickname "Q" (Alexander, 2026-10-11; matches the avatar). Executable `quarz`, namespace `quarz`, macros/CMake options
`QUARZ_*`. Old names still accepted: CMake `-DQSRZ_*` options (deprecation warning),
`tools/qsrz_read.py` (shim), `QSRZ_*` variables in `tools/build_gpu.sh`. Input keys and output
formats unchanged (openPMD attributes renamed `qsrz_*` -> `quarz_*`; no reader depends on them).
Quasi-static axisymmetric (m = 0, optional m = 1) PIC code for plasma wakefield
acceleration, LCODE-like, written in C++17 with Kokkos (OpenMP / CUDA / HIP), MPI
decomposition along xi with pipelined time steps, optional openPMD output.
Distinctive: arbitrary non-uniform radial grid (finite-element Laplacian on hat functions),
explicit B_perp solve, Adams–Bashforth plasma push, several beam pushers. Alexander plans to
publish it.

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
cd validation && ./run_validation.sh -j 2 -c          # full suite (sections 1-15), ~7-10 min: milestones / new physics
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

## Status (2026-10-11)
Done and validated (README §7, `validation/reference_full.txt`):
- openPMD native grid (2026-10-14): rectilinear meshes, `gridType = "rectilinear"` + `gridCoordinates_r`,
  geometry `other` + geometryParameters `thetaMode;...` (interim), the convention shared with GEM-PIC-RZ and
  proposed to openPMD (`docs/openpmd_rectilinear.md`); shared reader `tools/openpmd_rectilinear.py`, checked
  in section 10 (`check_openpmd.py`). `quarz_r_nodes` still written (deprecated).
  2026-10-15: A. Huebl (openPMD) invited a PR against upcoming-2.0.0; the 2.0 form stores coordinates
  as DATA SETS referenced by the mesh attribute `gridCoordinates` (HDF5 attribute size limit, issue #142).
  QUARZ keeps the attribute form until openPMD-api can write such data sets; the shared reader reads both.
  2026-10-16 (Alexander: no news from openPMD, go on implementing): reference implementations on local
  branches `rectilinear-meshes`, pushed 2026-10-16 to the forks AlexMPukhov/openpmd-api (base dev),
  /openpmd-viewer (dev), /openpmd-validator (1.1.X); no PRs yet (Alexander decides when); patch series also in
  docs/openpmd_patches/: openPMD-api (/home/claude/openpmd-src, off dev:
  Iteration::gridCoordinates container at <basePath>gridCoordinates/, Mesh/MeshRecordComponent::
  setGridCoordinates, Iteration::gridCoordinatesPath/gridCoordinatesOf, Python bindings, openpmd-pipe copies
  coordinates, examples 3c_*, tests HDF5/JSON/ADIOS2 incl. variable-based; full SerialIOTests pass),
  openPMD-viewer (/home/claude/deps/openPMD-viewer: both backends, slicing, 3D thetaMode, pcolormesh, interim
  forms; tests/test_rectilinear.py), openPMD-validator (/home/claude/deps/openPMD-validator: path/length/
  monotonic/position checks). QUARZ: CMake detects the API (QUARZ_OPENPMD_GRID_COORDINATES) and then writes
  the data-set form (<basePath>gridCoordinates/r, rank 0 stores it); otherwise the interim form; native
  gridSpacing is now the smallest spacing. Section 10 checks native output with 2 MPI ranks too (both forms
  bit-identical to serial). Builds: build-mpi (released API + MPI, interim form), build-gc-mpi
  (/home/claude/openpmd-install-rect-mpi, data-set form; needs LD_LIBRARY_PATH incl. hdf5-par/lib).
  Section 10 needs an openPMD-api with MPI: with the default `build` (openpmd-install, no MPI) its MPI lines fail.
  openPMD-api requires equal extents for all components of a record: staggered data with different extents
  go into separate records (proposal text updated).
  2026-10-18 (Alexander: teach ParaView the rectilinear grids): ParaView's openPMD reader
  (Wrapping/Python/paraview/algorithms/openpmd.py, master 1e98e01; sparse clone /home/claude/kitware/paraview,
  branch openpmd-rectilinear) -> vtkRectilinearGrid for cartesian meshes with coordinates, explicit r/z nodes
  in the thetaMode vtkStructuredGrid; coordinate data sets via openPMD-api (Iteration.grid_coordinates) or
  h5py fallback (HDF5); interim forms; geometry other + thetaMode params. Patch docs/openpmd_patches/ParaView/
  (MR on gitlab.kitware.com: Alexander, not reachable from here); plugin tools/paraview/
  openPMDRectilinearReader.py (generated from the patched module + decorators). Tests without ParaView (no
  binary reachable here): tools/paraview/test_openpmd_reader.py with pip vtk 9.7.1 and stubbed paraview
  package, synthetic files make_rectilinear_testfiles.py; in section 10 when vtk is importable. Patched
  openPMD-api rebuilt in /home/claude/openpmd-rect-install (MPI+HDF5, python; system json/toml11).
  NOT tested in the ParaView GUI (decorator XML, two output ports) - Alexander to try.
  Release v1.0.0 (2026-10-18, Alexander: "usable right now"): CMake project VERSION 1.0.0 -> QUARZ_VERSION
  (banner, openPMD software version). This session can neither push tags nor create releases (proxy /
  HTTP 403): Alexander creates the GitHub release v1.0.0 on main (text docs/releases/v1.0.0.md, assets
  openPMDRectilinearReader.py, quarz_paraview_samples.zip). Zenodo DOI: Alexander enables the repo on
  zenodo.org BEFORE creating the release.
- core solver, non-uniform grid, m = 1 mode, mobile ions, parsed profiles, MPI, openPMD,
  CUDA compiles and runs on a Blackwell GB202 GPU (Thomas, 2026-10-06: ~50x vs one thread of a
  Xeon E5620; no systematic benchmark and no GPU-vs-CPU file comparison yet);
- laser envelope solver (Crank–Nicolson in t, trapezoid in xi), ponderomotive force on
  plasma and beams; `pusher.max_qsa_factor` (gamma/Delta > 35 removed as trapped);
- ionization: ADK by plasma/beam fields, period-averaged ADK by the laser (numerical
  quadrature, drift momentum from sampled birth phase), Bethe beam impact ionization 0 -> 1,
  `nsplit` quanta for small fractions; examples `awake_impact_ionization.in`,
  `lwfa_ionization.in`. Immobile ionizable ions do not contribute to chi.
- adaptive sub-slicing of the plasma push (`pusher.max_cells_per_step`, default off): sub-slices
  with full deposit + field solve, beam sources extrapolated with their xi-derivative (guard row
  now carries all 15 beam components), variable-step Adams-Bashforth (history xi positions and
  push counter travel with the particles between ranks; step growth <= 2x), MPI bit-identical.
  Section 13, unit test test_ab. `output.field_files = 0` writes only axis files.
  Finding (2026-10-06): the per-particle sub-stepping first proposed did NOT help (frozen fields
  in xi); the dxi error near the axis is (a) the closure spike = near-singular caustic, not
  convergent at any dxi, and (b) the wake behind it: 0.3 % at dxi 0.005, 2 % at 0.01 -> 0.2 %
  with sub-slicing. Witness region unaffected (1e-5).
- regularization of the bubble-back singularity: `plasma.smooth_length` = a, radial filter
  (1 - a^2 Lap)^-1 on all plasma sources + background (FieldSolver::filter, per angular number,
  n = 0 zero-flux wall, n >= 1 axis/wall rows pass-through: the wall value of S enters the B_theta
  flux condition — forcing it to 0 was a bug). Criterion (Alexander): the fields AFTER the spike
  must converge, the spike height need not. a = 0.005: wake behind converges cleanly (Ez 0.18 ->
  0.05 % for dxi 0.005 -> 0.00125; cold 0.42 -> 0.19 %, noisy); changes vs cold O(a): ~0.5 %
  behind the bubble, 2e-4 at the witness.
  Temperature alone (uth) is not a reliable regularizer (seed dependent). Section 14, test_filter.
  Agreed with Alexander 2026-10-07 (the spike is unphysical: cold plasma, perfect cylinder).
- checkpoints / restart and diagnostic output groups (src/SimulationIO.cpp): `checkpoint.every`,
  `.keep`, `.at_end`, `.dir`, `restart.from` (path or latest). Per-rank files written without
  sync (pipeline keeps running), tmp + rename, complete = all rank files; state = beams in memory
  order incl. dead (packed_all), outbox_, Beam::started_, laser envelope (Laser::current), t, step.
  Restart with same P bit-identical (1 thread), other P redistributes (restrict_to_local, envelope
  slices by global index); logs truncated at the restart step. Groups: main output (output.*) +
  `diag.names` with every/beam_every/fields/beams/rmax/xi_stride/particle_stride/axis/format/
  openpmd_* (OutputGroup; OpenPMDWriter takes the key prefix, writes only present components).
  Section 15 (restart_check.py, group_check.py, group_openpmd_check.py). Note: runs with >1 OpenMP
  thread are not bit-reproducible (atomic deposits, ~1e-11) — any bit-identity test needs 1 thread.
- AWAKE SM benchmark by Thomas (2026-10-08, deck = John's IPAC26 TUP3035 Fig. 2 case; proton
  driver 20 M particles, mobile Rb ions, density step 3.5 % at z = 8750, box 1000 x dxi 1/64,
  grid to R = 30, plasma r <= 5, ppc 128): QUARZ CPU and GPU, uniform (3840 cells) and stretched
  (278 nodes, regions 0.1:0.005 1:0.01 5:0.05 30:1.0) agree with HiPACE++ (E_z amplitude, phases
  of E_z(0), F_r(0.1), F_r(1)); small near-axis deviation in F_r(0.1) phase, probably ion motion
  (off in HiPACE++). Stretched E_z amplitude ~3 % above uniform/HiPACE at xi < -170: check grid
  convergence (0.05 cells at 1 < r < 5 and at the plasma edge). Timings: 16 GH200 GPUs uniform
  2560 s / stretched 391 s; 240 CPU ranks x 2 threads 8520 / 1970 s; 1 GH200 ~ 118 s per sweep.
  Fixed after his report (b6bb3b7): every rank held the whole beam (OOM with 48 ranks/node);
  zero-weight plasma rings outside r = 5 were loaded (6x particles) and all live rings went to one
  OpenMP thread (scaling 1.06x -> 1.83x on 2 threads). Advice: ppc 128 is far more than needed
  for m = 0; pipeline efficiency N/(N+P-1) (240 ranks, 300 steps: 56 %); field output of the
  uniform case is 20 GB per file -> diagnostic groups.
- Thomas' scaling benchmark (2026-10-09, inputs_nodiags = AWAKE deck without output; per-sweep s):
  JUWELS node 1 core 1516, 48 MPI ranks 33 (96 %), OpenMP 6/12/24 threads 442/332/274 (5.5x),
  2x24 hybrid 83; GH200 1/2/4/8/16 GPUs 108/55/27/13/7 (ideal). Production 16 ranks: uniform CPU
  (16x24 threads) 8960 s, GPU 2310 s; stretched (now 5:0.02, 403 cells) CPU 7240, GPU 359.
  Diagnosis: GPU time ~ alpha*N_nodes (uniform/stretched fit: ~87 % of the uniform GPU time grid
  bound) -> the one-block PCR (log2 M passes over global memory). New TridiagMethod::Partition
  (Laszlo-Giles-Appleyard Thomas/PCR hybrid, P = min(256, M/8) chunks, reduced 2P system by PCR in
  team scratch; result independent of thread count) is now the GPU default (`auto`); host default
  stays Thomas (partition on 2 vCPUs here: 57 us 1 thread vs Thomas 43 us). Unit test M = 16..9001,
  full runs agree with Thomas to 1e-10; CUDA compiles. NOT yet measured on a GPU.
  CPU threading: kernel profile here (1 thread) is 97 % in plasma.push/deposit/deposit_S, serial
  Thomas 0.8 %; Amdahl fit of Thomas' numbers gives a constant ~3.5 ms per slice that does not
  shrink with threads -> suspect unpinned threads (we told him OMP_PROC_BIND=false in
  paper/inputs/README; now corrected to close/cores) and fork/join wake-up (~25 regions per slice).
  Asked for kernel-timer profiles (1 vs 24 threads, GPU uniform vs stretched).
  `grid.type = stretched` accepted as alias of regions (Thomas' deck uses it).
  JUPITER = GH200 (not H200). Timings = averages of 50 sweeps (1 core: 10). HiPACE++ 24.08: dx=dy=1/128
  (same as QUARZ uniform), dxi 1/64, box 16x16x251, 5x5 ppc, 1.2e8 beam particles, immobile ions.
  Paper: Sec. awake (Fig. fig_awake.pdf from Tom) + Sec. perf (Table tab:perf); JSC project LPQED.
  F_r(0.1) phase differs up to 0.4 rad (xi -210..-130), same transverse cell -> not resolution;
  likely non-axisymmetric noise in 3D acting on the tearing (m = 0 excludes it).
  RESULT with partition (Thomas, 2026-10-09 afternoon): GH200 1/2/4/8/16 GPUs 25/13/6.5/3.4/1.8
  s/sweep (4.3x faster, still ideal scaling; 1 GH200 ~ 1.4 JUWELS nodes of 48 MPI ranks, 35 s).
  Production on 16 GPUs: uniform 2310 -> 667 s, stretched 359 -> 319 s (now particle bound).
  CPU 384 ranks pure MPI (8 JUWELS nodes): uniform 3900 s, stretched 1530 s (was 8960 / 7240 with 16x24 threads).
  Thomas ALREADY pinned threads (close/cores, srun --cpu-bind=cores): binding hypothesis wrong;
  OpenMP scaling (24 threads 277 s, 5.5x) still unexplained, kernel profiles pending. Measured here:
  ~0.65 ms per slice of non-kernel host time on 1 thread (~40 s per full sweep, does not scale).
  Description of his benchmark says density step at 75 cm, but the deck has z < 8750 k_p^-1
  = 1.76 m (k_p^-1 = 201 um at 7e14); sigma_z written as 253.74 um, should be k_p^-1 (5.1 cm).
- adaptive time step (2026-10-09, Tom's suggestion, like HiPACE++): `time.adaptive`, nt_per_betatron
  (20), dt_max/dt_min, adaptive_gamma_min (2), adaptive_density, adaptive_lag (default P+1, serial 1),
  `time.t_end` (lands exactly; also without adaptive), `<beam>.adaptive_dt`. Rank 0 chooses dt_n and
  the last-step flag, sent in the pipeline message header [n, dt, last]. gamma_eff = min max(gamma,
  gmin) m/|q| over non-rigid beams; each rank isend_small's its min after the push of step m to rank 0
  (tag 31000 + m % 1500); rank 0 uses m = n - lag (blocking recv, normally no wait), extrapolates a
  decreasing gamma linearly. n_max = max density at the BOX HEAD over the step (the QS model gives
  the whole box the head density: z_front = t - xi_min; found when the ramp ODE matched only with
  z = t - xi_min). Variable-step leapfrog (kick (dt_prev+dt)/2, drift dt), identical bit for bit for
  constant dt (quick suite unchanged). Logs timestep.txt, gamma_records.txt; checkpoint adds
  adapt_rank_*.txt. Laser + adaptive refused. Section 16 (adaptive_ramp.in, adaptive_decel.in,
  adaptive_check.py): dt formula 6e-10, ramp ODE 1.0 %, 2304 vs 4220 steps; lag 6 overshoot 5 %
  only in the first lag steps; restart and 2-rank runs bit-identical.
- Paper, John Farmer's comments (2026-10-09): abstract rewritten without number lists; the pinched-
  witness result reframed as SAME axis resolution -> same answer, 30x faster: uniform dr = 5e-4
  (16000 cells, deck paper/inputs/conv_u0.0005.in) vs stretched 512 cells: n_i(0), W_r agree 1-4e-5,
  E_z 1e-3 (sheath resolution 0.01); 36.6 s vs 1.2 s per sweep on 2 cores. Row added to Table pinch.
- Plasma density at the box head for the whole box (z = t - xi_min) is CORRECT, not an
  approximation (Alexander, 2026-10-09; see his GEM-PIC paper): the QS method takes z as the slow
  coordinate (variables xi = t - z, z); n(z) then belongs to the box front. Density varying inside
  the box only makes sense with slow time tau = t AND the slow-time derivative kept (GEM-PIC);
  once it is dropped, as in QS, the difference disappears. Do not "fix" this.
- Pitfall (2026-10-09): disk allowance full -> runs died (SIGBUS) and left /dev/shm/sem.OMPIO_*;
  afterwards every run hung in MPI_File_open (sharedfp sem_wait). Fix: rm /dev/shm/sem.OMPIO_*.
  Old scratch runs in /tmp/claude-0 (sub, rst, mem: 16 GB) were deleted.
- GUI (2026-10-10, Alexander: optional GUI for desktop/notebook runs): tools/quarz_gui.py (stdlib
  http.server + numpy) + tools/quarz_gui.html (no external JS; canvas plots). Deck editor, Run/Stop
  (threads, MPI ranks), log, live frames via an extra diag group 'gui' passed as command-line
  overrides (deck unchanged; old gui frames deleted at run start), 2D map (r resampled to uniform,
  xi thinned by block max |f|, clip 99.5 %, mirror, head right), xi cut at r, radial cut at xi (click
  map), frame slider/play, beams.txt plot (stale file hidden during a run: MPI merges at the end).
  127.0.0.1 only + token. MPI frames shown when rank-0 step >= frame + P. Checked with Playwright
  screenshots (serial, 2 ranks, error deck). Validation section 17 (gui_check.py, headless API).
  No tkinter in this container -> browser approach on purpose.
- Multiple Coulomb scattering of beam particles (2026-10-10, Alexander): `<beam>.scattering`,
  scattering.Z / ion_charge / A / coulomb_log_ions / coulomb_log_electrons / factor / seed; needs
  units.n0_cm3. In Beam::advance after the Lorentz kick: Gaussian dp_x, dp_y with variance
  kappa [Z^2 n_i L_i + n_e L_e + (Z-zeta) n_i L_b] dt_kick, kappa = k_p r_e (q/m)^2; local n_e, n_i
  from diag (D_NE0, D_NI0; ni = number density of positive species + neutralizing background);
  |p| kept. RNG key = hash(seed, step) x hash(bits of x, y, xi, pz before the push): independent of
  memory index -> MPI particles bit-identical, restart bit-identical. Section 18: free diffusion
  <p^2> = D t within statistics, ion-channel emittance growth D/sqrt(2 gamma) to 0.8 %. Section 18
  runs with 1 thread (threaded deposits change state bits -> other random numbers). CUDA compiles
  (struct Scatter must be public: extended lambdas cannot capture private member types).
  DEFAULT ON (Alexander 2026-10-10: on if cheap) for non-rigid beams when units.n0_cm3 is given;
  cost: beam.push 2.0x -> 1.9x after one-log optimisation (1e6 particles, 1 thread), ~0.2 % of an
  AWAKE-size run. Full suite 18/18 with the default on; CUDA compiles.
  Observation: scatter_channel.in (pure ion channel, no electrons) has MPI-vs-serial field diffs of
  3.6e-16 in bth even WITHOUT scattering (particles identical) - pre-existing, not investigated.
- Two-stage runs (2026-10-10, Alexander: stiff driver with large dt, soft witness far behind):
  boundary.write (stage 1: last rank also pushes through its last slice and appends per step the
  downstream pipeline message minus beams/guard: mobile species pack_state + B+ 6M; header with grid,
  dxi, xi_head, xi_next, species/ab_order; record t, step, len, max|rho_b| of the last slice; restart
  truncates records >= start step) and boundary.read (stage 2, rank 0 unpacks the state interpolated
  linearly in t, mask: x,y,px,py,dl,w + AB history + B+ interpolated, counts/flags/lev/xi bookkeeping
  from the nearer record; Np mismatch -> nearer record). xi_head_ = stage-1 xi.min for z_front (and
  adaptive dt density). Guard = 0 in stage 2. Section 19: same dt bit-identical (serial and 2+2 ranks),
  dt/4 with rigid driver bit-identical, evolving driver: witness gamma 2e-5 vs fine single run while a
  coarse single run is 3 % off.
- Beam spin, Thomas-BMT (2026-10-10, Alexander): `<beam>.spin = sx sy sz`, `<beam>.anomalous_moment`
  (defaults by mass: e+-, mu, p). Packed particle = 10 doubles with spin (Beam::ncomp(); pipeline
  messages, checkpoints, dumps, merges, openPMD record `spin` all use it; readers infer 7/10 from
  the size). Spin at half steps like p; W from fields of the step and mean of old/new u, Boris
  rotation; startup rotates by -dt/2. Scattering kicks also turn s by kappa*theta about the kick
  axis, kappa = g beta^2 (a + 1/(g+1)). spin.txt (<s>, P = |<s>|; MPI parts merged, restart-truncated).
  Section 20: ion channel s_r = sin(kappa theta) 1e-6 (e, mu, pbar kappa 90), gamma-3 witness in a
  wake with a = 2 vs Python RK4 second order (1.3e-3 -> 3.1e-4), scattering slope 0.999999, restart
  and 2 ranks bit-identical. Cost: beam-only run 17 % longer. Not included: Stern-Gerlach force,
  Sokolov-Ternov. Full suite 20/20, CUDA compiles.
- Trapped plasma particles -> beam particles (2026-10-11, Alexander: Trojan-horse / ionization
  injection): `<species>.trap = 1` (mobile, non-ionizable), `.trap_beam` (default <species>_trapped,
  created automatically with the species' q, m; must NOT be in `beams`), `.trap_nphi` (m = 0 copies).
  Push kernel: at the QS limit (Delta < delta_min or gamma/Delta > max_qsa_factor) records tw = w,
  txi = xi_cur before the push, tpz = gamma - Delta (incl. laser A); the dead particle keeps its OLD
  state (x, y, px, py) -> beam particle at the slice where the limit was reached (stays local to the
  rank). Simulation::convert_trapped after step_fields: weight w * (dt_prev + dt)/2, xi = xi_head_ +
  txi, m = 0 random azimuth from a hash of the state; Beam::add_new appends and back-kicks by
  -dt_prev/2 (only if the beam is started; else the start-up does it). `<beam>.passive = 1`: no
  deposit (test particles). Also fixed: lost_ counter was never reset on ranks > 0 (display only).
  Section 21 (trap.in, trap_check.py; Trojan horse: rigid driver, 0.1 % He, laser a0 0.06 in the
  bubble, passive trapped beam): H = gamma - p_z - psi = 1 - psi_birth, min H = 1 - max psi to 9e-6,
  conserved per particle 4e-4 rms over 30 steps; charge rate dt 0.2 vs 0.1 1.4e-3; max_qsa 20/35/70
  identical charge; with beam loading restart and 2/3 ranks bit-identical. Self-injection at the
  closure of a strong bubble (blowout n_b = 8) is NOT clean: H far from 1 (closure caustic).
  Example examples/trojan_horse.in (19 pC, <gamma> 25 at t = 40). Full suite 21/21, CUDA compiles.
- Paper v1 (arXiv submission planned Fri 2026-10-09): new subsection "Bubble closure on a fine
  axial mesh" (sec:closure, Fig. fig_closure from scripts/closure.sh + fig_closure.py), framed
  around the grid: uniform grids regularize the caustic implicitly over one cell, the fine axis
  mesh exposes it, explicit smoothing a decouples regularization from the mesh. One sentence each
  in abstract, sec:disc, conclusions. Sub-slicing deliberately NOT in the paper.
- Logo (2026-10-08, chosen by Alexander): docs/logo (self-modulated proton bunch through the
  tilted radial-grid disk inside the Q ring, amber sparks = made with Claude; NOT Anthropic's
  logo). SVG with outlined text, PNG, avatar, favicon; make_logo.py; README header.
- `paper/tex/` (LaTeX source, figures, PDF of the PRAB draft), `paper/scripts/` (figure and
  error scripts, path-independent).
- `paper/inputs/`: self-contained decks of all paper cases (generated by make_decks.py,
  verified to reproduce the convergence errors exactly), `run_paper_cases.sh`, README with
  CPU timings; given to Thomas Wilson for GPU performance tests.

## Open issues
- Build pitfall (solved 2026-10-05): on Alexander's cluster CMake took /usr/bin/c++ = GCC 8.5 while
  the gcc/15.3 module's libstdc++ was loaded at run time -> crash in std::filesystem::path.
  CMake now refuses GCC < 9; use CXX=$(which g++). Branch wip/posix-io (POSIX writes,
  output.io switch) was a wrong lead; delete it on GitHub (session cannot delete branches).
- Wake-T 0.9.1 disagrees with QUARZ by up to 11 % (a0 = 2) and 27 % (a0 = 4) for a single
  laser wake; QUARZ satisfies Gauss's law, Wake-T does not in the laser region. Needs a
  full-PIC comparison (e.g. FBPIC) before trusting a0 >~ 2.
- Bethe M^2, C tabulated only for Ar; Rb (AWAKE) needs literature values.
- Laser: m = 0 envelope only, no d^2/dt^2, no phase correction for strong red-shift,
  no ionization energy loss.
- No collisional ionization by plasma electrons, no recombination; impact only level 0 -> 1.
- GPU: with the partition solver 1 GH200 (25 s/sweep) ~ 1.4 JUWELS nodes (35 s). Next: fused
  slice kernels, laser.advance is still a serial complex
  Thomas (Range(0,1)) on the GPU. CPU OpenMP scaling poor (5.5x on 24 threads): wait for profiles.

- Paper: v1 on arXiv as arXiv:2610.16910 (announced 2026-10-14; README header + "Citing QUARZ",
  CITATION.cff). Later edits in paper/tex (S+ explicit, w defined, Sec. VII stretched grid vs mesh
  refinement, Table III per step, Fig. 2e label) go into v2 / the PRAB submission; referee
  round used for items 1-3 below. The manuscript cites https://github.com/AlexMPukhov/QUARZ
  (public). Everything pushed is visible to everyone: no unpublished results in the repo.

## Next steps (agreed 2026-10-06, before the revised manuscript)
1. Cross-code benchmarks: beam-driven cases vs LCODE 2D and QPAD/HiPACE++; laser vs FBPIC.
2. DONE (see Status): adaptive sub-slicing + error analysis. Candidate for the revised paper:
   a short paragraph/figure on the closure spike vs the converged wake behind it, and the
   smoothing regularization (plasma.smooth_length) with its O(a) cost.
3. Long-term stability/noise behind long drivers (AWAKE-length).
Later: m >= 2 modes (POSTPONED, agreed 2026-10-07: only for generality; flat beams not a goal.
For AWAKE e-seed misalignment m = 1 suffices to O((d/sigma)^2): check by an offset scan,
response/d constant = linear regime, plus one QPAD/HiPACE++ case at the largest offset.
Estimate if ever done: 3-5 days, refactor to mode-indexed arrays first); phase-corrected / m = 1 laser
envelope; beams from openPMD files, warm plasma, Python/PICMI interface;
GPU benchmark vs a full current CPU node, fused slice kernels; Bethe data beyond Ar.

## Earlier proposals
- Benedetti phase-corrected envelope; m = 1 laser envelope.
