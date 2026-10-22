# Handbook for Claude sessions on QUARZ

CLAUDE.md is the project *state* (what exists, what is validated, what is open). This file is the
*experience* behind it: how the work with Alexander runs, why things are the way they are, what
was tried and failed, and the traps that cost time. It was written on 2026-10-22 by the session that
built QUARZ from the first line (2026-09 to 2026-10), for every later session. Read it once at the
start of a session, together with CLAUDE.md; update it when you learn something a successor needs.

The repository is public: nothing private about people goes here.

## 1. Working with Alexander

- **Who.** Alexander Pukhov leads a theoretical plasma physics group (HHU Düsseldorf). He is an
  expert in PIC codes and laser/beam-plasma physics (author of VLPL and of the GEM-PIC scheme).
  Talk physics and numerics at expert level; do not explain basics. He writes C++, not much Python.
- **Zero-touch code.** Alexander directs, Claude writes *all* code, tests, documentation and the
  manuscript; he has not edited a single line, and this is a published claim (paper, README:
  "zero-touch code"). So never hand him code to edit or patch to apply. Ask him only for what the
  container cannot do (see §6), and then give exact click paths or commands.
- **Requests are short** ("please add X", "case 3", "ok"). A one-line request means the whole job:
  design, implement, validate against an independent reference, document (README + §7 row),
  update CLAUDE.md, commit, push, and report in a few lines. Don't ask permission for the obvious
  steps; ask only when a choice changes the physics or is expensive to undo, and then offer
  numbered options with a recommendation (he answers "case 3", "1 and 2").
- **He asks for recommendations** ("what is your recommendation?"): give one clear answer with
  the reason, not a menu.
- **Honesty over success.** He values being told that something did not work (per-particle
  sub-stepping did not help; self-injection at the bubble closure is not clean; Wake-T disagrees;
  the timing in Table III was per step, not total). State limitations and untested parts plainly
  ("not tested in the ParaView GUI"). Never report a feature done before its validation ran.
- **He corrects physics; check, then follow.** Example: the plasma density at the box head for
  the whole box is *correct* in the quasi-static model, not an approximation (CLAUDE.md). When he
  states a criterion, it becomes the acceptance test (e.g. "the fields AFTER the closure spike
  must converge, the spike height need not"; "default on if cheap").
- **Terminology he insists on:** GEM-PIC(-RZ) is *not* quasi-static (it keeps slow-time
  derivatives; quasi-static = all slow-time derivatives dropped). The code is QUARZ, nickname "Q".
- **Answers:** concise, structured, numbers instead of adjectives, no flattery, no recap of
  steps. He reads on the phone too: lead with the result. He writes English in the QUARZ work
  (German in some other chats); answer in the language he uses.
- **Reports from collaborators** arrive pasted (logs, `module list`, `ldd`). Debug from those,
  ask for one specific command at a time on his cluster, and bisect by commit when "it worked
  before" (that is how the GCC 8 / libstdc++ crash was found, 2026-10-05).
- **Deliverables:** code pushed to GitHub; zips when asked (`quarz.zip`, Overleaf zip of the
  paper); figures as PNG/PDF via the file-send tool. Paper edits go to `paper/tex/` and are
  recompiled; he uploads to arXiv / journals himself.

## 2. People and roles (public roles only)

- **Thomas C. Wilson** (co-author; same group): runs QUARZ on JUWELS (CPU) and JUPITER (GH200
  GPUs), JSC project LPQED; benchmarks vs HiPACE++, scaling. His reports are precise; take his
  numbers seriously, but check his deck descriptions against the decks (it happened that the
  text said 75 cm and the deck said 1.76 m).
- **John Farmer** (AWAKE; commented on the paper): no number lists in the abstract; frame results
  physically (same axis resolution -> same answer, 30x faster).
- **Lars**: tested the ParaView plugin in the GUI (ParaView 6.0.1, 2026-10-19: works). ParaView itself is
  not reachable from the container, so GUI checks go to him.
- **Axel Huebl** (openPMD): invited a PR for rectilinear meshes against upcoming-2.0.0. PRs to
  openPMD / Kitware are opened by Alexander when he decides; the code is on his forks.

## 3. How a feature is built here (the checklist that worked 20 times)

1. Read the relevant code paths fully before designing; write the physics/numerics in a few
   lines first (equations, where in the step, what is conserved).
2. Implement with the invariants in mind (CLAUDE.md): MPI bit-identity, restart bit-identity,
   counter-based random numbers keyed on *state*, not memory index (scattering key = hash of
   x, y, xi, pz bits), prefix sums for new particles.
3. **Touchpoints for anything that changes per-particle beam data** (spin made this list):
   `Beam::ncomp()`, pipeline messages and outbox (`Simulation.cpp`), checkpoints and restart
   (`SimulationIO.cpp`), particle dumps and their MPI merge, openPMD particle records
   (`OpenPMDWriter.cpp`), readers (`tools/quarz_read.py` infers the column count),
   `validation/restart_check.py`. New log files: MPI merge at the end and truncation on restart.
4. **CUDA:** extended lambdas cannot capture private member types (make such structs public);
   no `std::` containers or virtual calls in kernels; run the CUDA compile check when device code
   changes (~10 min).
5. **Validation section** in `validation/run_validation.sh` against an *independent* reference
   (analytic solution or a separate Python integrator: RK4 for spin, ODE for the adaptive-dt ramp,
   h5py reader for openPMD), with a quick variant; bit-identity checks with 1 OpenMP thread;
   check the numbers, then `-u` for both reference files; README §7 row; CLAUDE.md status entry.
6. Quick suite `-c` after every code change; full suite at milestones; commit + push.

## 4. Physics and numerics: decisions and their reasons

- Non-uniform radial grid = the point of the code (paper). Finite elements on hat functions keep
  the Laplacian symmetric on any grid; geometric spacing with `grid.max_ratio` (default 1.05,
  Alexander's choice) between region points.
- **Bubble closure spike** is a near-singular caustic of the cold, perfectly axisymmetric model,
  not convergent in dxi. A uniform grid hides it (implicit regularization over one cell); the fine
  axis mesh exposes it. Fix = explicit radial smoothing `plasma.smooth_length` (decouples
  regularization from the mesh); acceptance = convergence of the wake *behind* the spike.
  Temperature is not a reliable regularizer (seed dependent).
- Sub-slicing: the per-particle sub-stepping first proposed did not help (fields frozen in xi);
  full sub-slices with field solves do. Not in the paper (deliberately).
- Laser envelope: BDF2 / backward differences in xi were unconditionally unstable, and there was
  a sign error in the transverse operator; the working scheme is Crank-Nicolson in t, trapezoid
  (box) in xi. Strong closure spikes made gamma/Delta diverge and killed ionization runs ->
  `pusher.max_qsa_factor` (particles beyond the QS limit are removed, or trapped -> beam).
- Wake-T disagrees at a0 >~ 2; QUARZ satisfies Gauss's law, Wake-T does not in the laser
  region. Do not "fix" QUARZ towards Wake-T; a full-PIC comparison (FBPIC) decides.
- Self-injection at the closure of a strong bubble is not clean (Hamiltonian far from 1);
  ionization / Trojan-horse injection is (section 21).
- m >= 2: postponed by decision (flat beams are not a goal); m = 1 covers small AWAKE seed
  offsets to O((d/sigma)^2).
- Performance: GPU time was grid bound (one-block PCR) -> Thomas/PCR partition solver (4.3x on
  GH200). CPU OpenMP scaling (5.5x on 24 threads) is still unexplained; thread pinning was not
  the cause. ppc 128 is far more than needed for m = 0.

## 5. Traps in this container

- `export OMP_PROC_BIND=false`, otherwise OpenMP runs badly here.
- The disk allowance is small: delete old run outputs. When it filled, runs died with SIGBUS and
  left `/dev/shm/sem.OMPIO_*`, after which every MPI-IO run hung (remove those files).
- Never edit `run_validation.sh` while it runs (bash reads scripts incrementally). Poll background
  jobs by PID, not with `pgrep -f` patterns that match the polling shell.
- Bit-identity tests need 1 OpenMP thread (atomic deposits differ at ~1e-11).
- Stale output directories make comparisons look identical (happened with Wake-T): delete per run.
- A stale build writes old formats: rebuild before testing output changes.
- openPMD-api FetchContent downloads fail here: build it with system nlohmann-json and toml11;
  only parallel HDF5 is in apt, so openPMD-api is built with MPI.
- Python `openpmd_api` is the pip wheel. Never `cmake --install` an openPMD-api build with Python
  into site-packages: it overwrites the wheel's module, which then cannot find its library.
- Network: only package registries and GitHub. zenodo.org, paraview.org, gitlab.kitware.com are
  not reachable. Web pages: use the web tools, not curl.
- Headless Chromium for GUI screenshots: Python `playwright` with the preinstalled browser;
  the Node playwright version mismatches the preinstalled build.
- The container is not persistent across sessions: `tools/dev/setup_container.sh` rebuilds
  Kokkos, openPMD-api and QUARZ (vtk included; `WITH_EXTRAS=1` for playwright/Slurm/sshd, `WITH_CUDA=1` for
  the CUDA compile check). Step logs go to `$PREFIX/setup-logs/`; a failure names the step
  and prints the log tail. Run it in the foreground or with `setsid nohup` (two plain `nohup` runs died
  silently on 2026-10-22). The patched openPMD-api with coordinate data sets
  (`rectilinear-meshes` branch on the fork AlexMPukhov/openpmd-api) is needed only for the
  data-set form in section 10.

## 6. What the session cannot do (ask Alexander, with exact steps)

- Push tags, create GitHub releases, delete branches (proxy returns 403). `git push` of commits needs
  credentials in the container; without them (2026-10-22) push the changed files with the GitHub
  connector (`push_files`; it writes mode 100644, so update executable scripts with
  `create_or_update_file`) and check a fresh clone against the local commit.
- Anything on Zenodo, Kitware GitLab, arXiv, journal systems, his cluster, his email.
- Run ParaView or a real GPU. CUDA is compile-checked only; GPU runs are Thomas'.

## 7. Paper

- `paper/tex/quarz_paper.tex` (PRAB style), figures from `paper/scripts/`, decks in
  `paper/inputs/` (they reproduce the published errors exactly). v1 = arXiv:2610.16910; edits
  since then go into v2 / the journal submission (CLAUDE.md lists them).
- Claude writes the manuscript like a co-worker would; Thomas supplies his runs and figures.
  References numbered in order of first citation. Abstract without lists of numbers. Every
  number in the text must come from a run in the repo or from Thomas; say "per time step" vs
  "total" explicitly. The zero-touch statement stays in abstract, introduction, implementation,
  conclusions and acknowledgments.
- Software citation: Zenodo concept DOI 10.5281/zenodo.23472094; license BSD-3-Clause.
