<!-- Generated from README.md (section 6) by tools/sync_skills.sh. Do not edit this copy. -->
# QUARZ output files

Section numbers (§) refer to the README.md of https://github.com/AlexMPukhov/QUARZ.

- **`fields_NNNNNN.bin`** (format version 3). Header: int32 version = 3,
  int32 M, int32 K, int32 mode1, float64 t, r[M], ξ[K], int32 ncomp. Then
  ncomp blocks of {char name[16], float64 [K][M]}. ξ = z − t increases with the slice index:
  the head of the box is the last slice. Version 2 = QUARZ v1.0.0 (ξ = t − z, head first):
  `tools/quarz_read.py` and the GUI read both and convert version 2 (slices reversed, ξ negated).
  - m = 0 components: `psi ez er eth br bth bz ne ni rhob`.
  - m = 1 components (when present): the same names with `_c` / `_s`, the
    cos / sin amplitudes in f = f₀ + f_c cos θ + f_s sin θ.
- **`axis_NNNNNN.txt`**: ξ (increasing, head last), E_z(0), ψ(0), n_e(0), ρ_b(0), and the focusing
  slope (E_r − B_θ)/r at the first node. With m = 1 it adds W_x(0) and W_y(0),
  the transverse force on the axis. Fixed-width columns (11 significant digits).
- **`beams.txt`**, one line per beam per step, with columns:
  step, t, alive, N, charge, ⟨γ⟩, σ_γ, ⟨ξ⟩, σ_ξ, r_rms, ε_n,x, ⟨x⟩, ⟨y⟩, ε_n,y,
  σ_x, σ_y. The emittances are about the centroid. With `modes = 0` a macro-particle stands for a ring,
  and the x and y moments are those of the axisymmetric beam (azimuthal averages: ⟨x²⟩ = ⟨r²⟩/2,
  ⟨p_x²⟩ = ⟨p⊥²⟩/2, ⟨x p_x⟩ = ⟨r·p⊥⟩/2, means 0). Before 2026-10-30 they were the projections of the
  representative particles; for the parsed loader, which puts all of them at y = 0, σ_x was r_rms and
  ε_n,x about √2 too large and not conserved under radial forces.
- **`slices_<beam>_NNNNNN.txt`**: per-ξ-bin charge, ⟨x⟩, ⟨y⟩, σ_x, σ_y, ⟨γ⟩
  and ε_n,x, for hosing studies. Bins are `output.beam_slices` equal bins over
  `output.beam_slices_range` (empty bins are skipped).
- **Laser runs** add `laser.txt`, one line per step: step, t, max |â|,
  w_rms = (2⟨r²⟩)^{1/2}, ⟨ξ⟩, L_rms (all weighted with |â|²) and ∫|â|² dV.
  ∫|â|² dV is not conserved in a plasma: during depletion the photon number is
  conserved and the frequency drops. The field files contain the envelope as
  `a_re`, `a_im`, with a = Re[(a_re + i a_im) e^{i k₀ ξ}] in the user ξ = z − t (openPMD: mesh
  `laserEnvelope`, components `re`, `im`).
- **Ionization runs** add `ionization.txt`, one line per step, with four columns per ionizable species:
  - the weight of the electrons born in the sweep, in units of n₀(c/ω_p)² per c/ω_p;
  - the same as electrons per metre of plasma;
  - their mean p⊥² at birth;
  - the mean charge state of the column at the end of the box.

  The field files contain the ion charge density `nz_<species>` = Σ z n (m = 0;
  openPMD: mesh `n_z_<species>`).
- **MPI runs** write the same files. Field and axis files are written by all
  ranks directly into one file (each rank at its own byte offset). `beams.txt`,
  the slice files and the particle dumps are first written per rank
  (`*.partN`) and merged into the usual files by rank 0 at the end of the run;
  if a run is killed, the `.partN` files hold the data.
- **`beam_<name>_NNNNNN.bin`**: a 24-byte header, char magic[8] = `QUARZDMP`, int32 version = 1, int32 columns
  (7, or 10 for a beam with `<beam>.spin`), int64 n; then n × (x, y, p_x, p_y, p_z, ξ, w), or
  n × (…, w, s_x, s_y, s_z). Code units: x, y, ξ in c/ω_p (ξ = z − t), p in m_s c of the species, w in
  n₀(c/ω_p)³. Dumps of QUARZ v1.0.0 have no header (int32 n, then the records; ξ = t − z): the beam import
  refuses them, `tools/convert_deck.py --beam-file OLD NEW` converts them, and `tools/quarz_read.py`
  reads them converted, with a warning. Particles removed from the beam (w = 0) are not written.
  A dump can be read back as a beam (`<beam>.profile = file`, `<beam>.file = beam_<name>_NNNNNN.bin`). The
  momenta of a dump are half a time step behind the positions (leapfrog), so a run started from a dump is not
  an exact continuation (use checkpoints for that); with `particle_stride` > 1 the dump holds a subset
  with unchanged weights.
- **openPMD** (`output.format = openpmd` or `both`): one file per output step,
  `openpmd/data_NNNNNN.h5` (or `.bp`, `.json`), following the
  [openPMD standard 1.1](https://github.com/openPMD/openPMD-standard).
  - Field meshes `E`, `B` (components `r`, `t`, `z`), `psi`, `n_e`, `n_i`,
    `rho_beam`, in geometry `thetaMode`, axes (r, z), shape [modes][r][z].
    The modes are (f₀) or (f₀, f_c, f_s) with f = f₀ + f_c cos θ + f_s sin θ
    (geometryParameters `m=2;imag=+`), the convention of openPMD-viewer, FBPIC and WarpX.
  - **Longitudinal axis:** co-moving, as in other quasi-static codes: the mesh axis `z` is ξ = z_lab − ct (the
    same values and order as the native files; `gridGlobalOffset` = ξ_min). The lab position is z_lab = ξ + ct
    with the iteration time t (series attribute `quarz_coordinates`). QUARZ v1.0.0 files are in the lab frame
    (z = t − ξ_v1 = ξ + t, `quarz_coordinates` = "z = t - xi (lab frame); …"); the beam import
    (`<beam>.profile = openpmd`) recognises them and converts the positions exactly with the iteration time.
  - **Radial grid:** openPMD requires uniform spacing, so by default the fields are
    interpolated (piecewise linearly, consistent with the hat functions) onto a
    uniform radial grid. The default spacing is h_min, which can make the files much
    larger than the native ones; the code prints the size per output and a hint.
    `output.openpmd_grid = native` writes the exact data on the non-uniform nodes
    instead, as *rectilinear meshes* proposed for openPMD 2.0
    ([`docs/openpmd_rectilinear.md`](docs/openpmd_rectilinear.md)); the ξ axis stays regular.
    Built against an openPMD-api with coordinate data sets (our proposed extension of
    openPMD-api, see the same document; CMake detects it and says so), the radial nodes are the data set
    `<basePath>gridCoordinates/r`, referenced by the mesh attribute `gridCoordinates`,
    as GEM-PIC-RZ writes them. With a released openPMD-api QUARZ writes the interim
    form: `gridType = "rectilinear"` and the nodes in the mesh attribute
    `gridCoordinates_r`. Until the extension is part of the standard the meshes carry
    geometry `other` with geometryParameters `thetaMode;m=…;imag=+`, so readers of
    regular grids skip them. `tools/openpmd_rectilinear.py` (shared with GEM-PIC-RZ)
    reads both forms (`read_mesh`, `theta_slice`), and so does our proposed extension of
    openPMD-viewer. **ParaView:** load the plugin `tools/paraview/openPMDRectilinearReader.py`
    (Tools → Manage Plugins) and open a `.pmd` file with the reader "openPMD Reader (rectilinear
    meshes)": the fields appear on the true non-uniform nodes (see `tools/paraview/README.md`).
    **Grid size limit of the interim form:** HDF5 stores an attribute in 64 KB, so with a released
    openPMD-api (attribute `gridCoordinates_r`) and HDF5 files the native grid can have at most 8181 radial
    nodes; beyond that the run stops at the start with a message. The data-set form has no limit; otherwise use
    `output.openpmd_grid = uniform`, the native output, or `output.openpmd_backend = bp` if the openPMD-api has
    ADIOS2. The old per-mesh copy `quarz_r_nodes` (same limit, in every build) is no longer written.
  - **Particles:** every beam is a species with `position` (x, y, z = ξ, co-moving like the meshes),
    `positionOffset` (0, so openPMD-viewer also shows ξ), the species attribute `quarz_particle_stride`, `momentum` (per physical particle, in m_e c × `unitSI`, unlike the m_s c of the native files), `weighting`
    (physical particles per macro-particle), `charge` and `mass`.
  - **Units:** SI when `units.n0_cm3` is given, otherwise normalised units with
    `unitSI` = 1 (the series attribute `quarz_units` says which).
  - **MPI:** openPMD writes are collective. In the ξ pipeline the ranks reach a
    step at different times, so each openPMD output step drains the pipeline
    (about P − 1 block sweeps). Example, 2 ranks, output every 2nd of 10 steps:
    10.1 s (native) vs 14.0 s (openPMD). For frequent output with many ranks the
    native format, which needs no synchronisation, is cheaper.
  - Reading: `openpmd_viewer.OpenPMDTimeSeries('out/openpmd/')`, e.g.
    `ts.get_field('E', 'z', iteration=n, m='all', theta=0)`.
- **Diagnostic groups** (`diag.names`) write into `<output.dir>/<name>/`, with the same file
  names and formats as the main output (`fields_*.bin`, `beam_*_*.bin`, `axis_*.txt`,
  `openpmd/…`), restricted to their selection. A field file with `rmax` and `xi_stride` holds
  the written nodes and slices in its r and ξ header arrays, so `tools/quarz_read.py` reads it
  unchanged. The main output (`output.*`) is a group too; set `output.every = 0` to have only
  the named groups. Example: full data rarely, the accelerating field and the witness often:

  ```
  output.every       = 200            # everything every 200 steps
  output.beam_every  = 200
  diag.names         = light
  diag.light.every   = 5
  diag.light.fields  = ez psi ne
  diag.light.rmax    = 1.0
  diag.light.xi_stride = 2
  diag.light.beams   = witness
  checkpoint.every   = 1000
  ```

  In the pinched-witness case such a group (ez, ψ, n_e, r ≤ 1, every 3rd slice, every 7th
  witness particle) is 6 % of the size of the full output.
- **Checkpoints** (`checkpoint.every`): `checkpoints/chk_NNNNNN/` with `checkpoint.txt`
  (step, t, number of ranks, the full deck) and one `rank_<r>.bin` per rank. A checkpoint
  `chk_N` holds the state at the beginning of step N:
  - all beam particles in memory order (removed ones included), the particles in transit to
    the downstream rank, and the leapfrog start flag of each beam;
  - the laser envelope;
  - t and the step.

  The plasma needs no state, because it is loaded anew at the head of the box in every step.
  Every rank writes its own file as soon as it has finished step N − 1, so checkpoints do not
  stall the ξ pipeline. A checkpoint is complete when all rank files exist (each is written to
  a temporary file and renamed). `checkpoint.keep` deletes older checkpoints only after a newer
  one is complete.
- **Restart:** `quarz deck.in restart.from=out/checkpoints/chk_000500` (or `restart.from=latest`),
  with the same deck; `time.steps` is the last step of the whole run. The radial grid, the ξ
  box, the modes, the beams and the laser must be those of the checkpoint. The run continues
  at step N, and the restarted part is bit-identical to an uninterrupted run (same number of
  ranks, one OpenMP thread per rank; with several threads the atomic deposits are not
  bit-reproducible in any run). With a different number of ranks, the particles and envelope
  slices are redistributed. Restarting into the same `output.dir` truncates `beams.txt`,
  `laser.txt` and `ionization.txt` at step N and continues them; field and particle files of
  steps ≥ N are overwritten. Diagnostics with a single-file openPMD series (pattern without
  `%T`) are refused on restart, because the file would be overwritten.
- **Python helpers.** `tools/quarz_read.py` reads all of these;
  `xz_plane()` builds the field in the (ξ, x) plane from the modes.
  `tools/plot_fields.py` makes a quick-look PNG.
- **GUI** (optional, `tools/quarz_gui.py`): an input-deck editor with run control and live
  plots in the web browser, for runs on a desktop or notebook. Needs only Python 3 and numpy
  (no GUI toolkit; the page has no external dependencies and works offline):

  ```bash
  python3 tools/quarz_gui.py examples/blowout_basic.in     # opens http://127.0.0.1:8765/?token=...
  python3 tools/quarz_gui.py run.in --no-browser --port 9000  # remote: ssh -L 9000:127.0.0.1:9000 host
  ```

  - *Deck*: edit, open, save (Ctrl-S); examples from `examples/`, `paper/inputs/`, `validation/`;
    extra `key = value` overrides that are passed on the command line but not saved.
  - *Run*: OpenMP threads, MPI ranks (`mpirun`), Run/Stop, progress bar and the log. The deck is
    saved before every run and run in its own directory.
  - *Live frames*: the GUI adds a diagnostic group `gui` on the command line
    (`diag.names = … gui`, `diag.gui.every`, `.fields`, `.rmax`, `.xi_stride`; the deck file is not
    changed), so the live output stays small whatever the main output does. Its frames appear in
    `<output.dir>/gui/` while the run goes on (with MPI a frame is shown once the last rank has
    written it).
  - *Plots*: 2D map of a field in (ξ, r) (optionally mirrored to ±r, head on the right, colour
    scale clipped to 99.5 % of the values so that a closure spike does not hide the rest), a cut
    along ξ at any r and a radial cut at any ξ (click on the map to set both), a frame slider with
    playback, and the beam evolution from `beams.txt` (⟨γ⟩, σ_γ, r_rms, ε_n, …). With `modes = 1`
    the fields are shown in the plane θ = 0. *Browse output* shows a finished run.
  - *Machines*: the run can also go to a cluster or another computer. The usual setup: the GUI
    runs on the head (login) node, the browser on the desktop via an ssh tunnel
    (`ssh -L 9000:127.0.0.1:9000 headnode`, then `python3 tools/quarz_gui.py run.in --no-browser
    --port 9000` there and open the printed URL). Machines are described in a JSON file
    (`--machines FILE`, else `$QUARZ_GUI_MACHINES`, else `~/.config/quarz/machines.json`; template
    `tools/quarz_machines.example.json`); `local` always exists. Two types:
    - `slurm`: the run is submitted with `sbatch` (keys `exe`, `partition`, `account`, `time`,
      `nodes`, `ranks`, `threads`, `gpus_per_node`, `ntasks_per_node`, `mem`, `setup` = shell lines
      such as `module load …`, `launcher` (default `srun --cpus-per-task={threads}`), `sbatch` =
      extra `#SBATCH` options, `env`). The GUI shows the queue state (pending/running), follows the
      job's output file and the live frames on the shared file system; Stop = `scancel`. Partition,
      account, time, nodes and GPUs can be changed in the GUI; *Job script* shows the script
      without submitting it (it is also saved next to the deck as `quarz_gui_job.sh`).
    - `ssh`: the run is started on another computer (e.g. a GPU workstation) with ssh (key login,
      no password prompt; `host`, `exe`, `setup`, `launcher`, `launcher_mpi` (default
      `mpirun -np {ranks}`), `env`, `ssh_options`). Default: the run directory is on a shared file
      system (`path_map: [local prefix, remote prefix]` if it is mounted elsewhere there). With
      `remote_dir` there is no shared file system: the deck is copied to `remote_dir/<deck dir name>`
      and the output is pulled back with rsync while the run goes on. The run survives a lost
      connection (it is started with `setsid nohup`); Stop kills its process group.
    The placeholders `{ranks}`, `{threads}`, `{nodes}` are filled in `setup`, `launcher`, `sbatch`
    and `env`. The GUI itself does not need to stay up for the run to finish, but it follows only
    runs it started (a restarted GUI can still *Browse output*).
  - The server listens on 127.0.0.1 only, and every request needs the random token of the URL.

