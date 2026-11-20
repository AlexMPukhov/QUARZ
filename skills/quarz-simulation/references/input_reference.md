<!-- Generated from README.md (section 5) by tools/sync_skills.sh. Do not edit this copy. -->
# QUARZ input reference

Section numbers (§) refer to the README.md of https://github.com/AlexMPukhov/QUARZ.

Input files are plain `key = value` lines; `#` starts a comment. Keys that are
never read are reported at the end of the run, which catches typos.

**Radial grid**

| key | default | meaning |
|---|---|---|
| `grid.type` | `uniform` | `uniform` \| `regions` (alias `stretched`) \| `file` |
| `grid.rmax` | — | outer radius R (conducting wall) |
| `grid.dr` | — | spacing for `uniform` |
| `grid.regions` | — | `r_end:h` pairs, e.g. `0.05:0.0005 2.5:0.01 8:0.05`. Spacing h is used for r < r_end. Transitions are geometric. A coarse→fine transition starts early enough to reach the fine spacing at the boundary. In boxes longer than about 300 c/ω_p keep the cells inside the plasma at 0.02–0.03 or below and coarsen only outside it: the field noise of clustering plasma macro-particles grows with the cell size. QUARZ prints a note at the first step when a box longer than 300 has plasma in cells above 0.03. |
| `grid.max_ratio` | 1.05 | maximum ratio of neighbouring cells |
| `grid.file` | — | text file with one node radius per line (first 0, last R) |

The generated grid is written to `out/grid.txt` (j, r_j, h_j, V_j).

**Box, time, pusher, solver**

| key | default | meaning |
|---|---|---|
| `modes` | 0 | 0 = axisymmetric (m = 0), 1 = modes m = 0 and 1 |
| `solver.picard` | 3 (modes=1) | Picard iterations for the χ₁ mode coupling of B⊥ |
| `xi.min`, `xi.max`, `xi.step` | 0, —, — | box and slice spacing Δξ |
| `time.dt`, `time.steps`, `time.start` | 0, 0, 0 | beam time step and number of steps (0 = one quasi-static solve). With `time.adaptive`, `time.dt` (if given) is the largest step |
| `time.t_end` | — | end the run at this time instead of after `time.steps` steps (then only an upper bound); the last step is shortened to land on it exactly, without a tiny final step |
| `time.adaptive` | 0 | adaptive beam time step from the betatron period, see below |
| `time.nt_per_betatron` | 20 | steps per betatron period, 2π/(ω_β Δt) |
| `time.dt_max`, `time.dt_min` | `time.dt` or ∞, 0 | bounds of the adaptive step |
| `time.adaptive_gamma_min` | 2 | particles with smaller γ count as this γ (slow particles do not stall the run) |
| `time.adaptive_density` | from the profiles | density n for ω_β (overrides the profiles) |
| `time.adaptive_lag` | P + 1 (1 serial) | steps by which rank 0 knows γ late (≥ number of ranks P) |
| `<beam>.adaptive_dt` | 1 | include this beam in the adaptive step (rigid beams never count) |
| `pusher.ab_order` | 3 | Adams–Bashforth order 1–5 for plasma rings. 2–3 is most robust at bubble closure; 5 can blow up there. |
| `pusher.delta_min` | 10⁻³ | rings with γ − p_z below this are removed (trapped) |
| `pusher.max_qsa_factor` | 35 | rings with γ/(γ − p_z) above this are removed (trapped; the quasi-static weight diverges), or become beam particles with `<species>.trap` |
| `plasma.smooth_length` | 0 | a > 0: radial smoothing of all plasma sources, f ← (1 − a²∇⊥²)⁻¹ f, which regularizes the singular density/E_z spike at the closure of a bubble (§1, plasma push). 0.005 is a good value; the axis cell and Δξ should be ≤ a. |
| `pusher.max_cells_per_step` | 0 | adaptive sub-slicing: split a ξ step into sub-slices if a plasma particle would cross more radial cells than this (0 = off; 1 is a good value). The step line of the log reports the extra sub-slices (rank 0). |
| `pusher.substep_max` | 64 | largest number of sub-slices per step |
| `solver.tridiag` | `auto` | `auto` (Thomas on host, partition on device) \| `thomas` \| `pcr` \| `partition` |
| `beams.xi_shape` | `linear` (1 rank), `ngp` (MPI) | longitudinal shape of beam particles for deposit and field gather: `linear` (between two slices) or `ngp` (nearest slice). MPI runs need `ngp`; a serial run with `ngp` gives the same results as an MPI run. |
| `pusher.beam` | `vay` | beam momentum pusher: `vay` \| `hc` \| `imp` \| `imp_rr` \| `boris`. Can be overridden per beam with `<beam>.pusher`. |
| `pusher.rr_n0_cm3` | 0 | plasma density in cm⁻³. With `imp_rr`, a value > 0 switches on radiation reaction. |
| `pusher.rr_scale` | 1 | multiplies the radiation-reaction strength (testing only) |
| `<beam>.scattering` | 1 if `units.n0_cm3` is given | multiple Coulomb scattering on the plasma ions and electrons (needs `units.n0_cm3`), see §1. The default Z = 1 is hydrogen: set `scattering.Z`, `ion_charge` for other gases |
| `scattering.Z`, `scattering.ion_charge`, `scattering.A` | 1, Z, 1 (2Z for Z > 1) | nuclear charge, ion charge state, mass number of the background ions |
| `scattering.coulomb_log_ions`, `scattering.coulomb_log_electrons` | computed | fixed Coulomb logarithms instead of the per-particle formulas |
| `<beam>.spin` | — | three numbers: initial spin direction (normalised); switches on the Thomas–BMT spin precession, see §1 |
| `<beam>.anomalous_moment` | by mass (e±, μ, p) | a = (g − 2)/2; required for other masses |
| `scattering.factor`, `scattering.seed` | 1, 1 | multiplies the scattering rate (testing only); random seed |

**Function parser (HiPACE++ style).** Profiles can be given as formulas.
- **Compilation.** Each expression is compiled once into a small stack
  program, which runs on the CPU and inside GPU kernels.
- **Constant folding.** Parts without variables are evaluated at compile
  time.
- **Syntax and names.** Quotes around expressions are optional. The variable
  names are taken from the key, e.g. `plasma.density(x,y,z)`.

```
my_constants.L    = 200            # user constants; may use each other, in any order
my_constants.kp   = 2*pi/L
plasma.density(x,y,z)   = "if(z < L, sin(0.5*pi*max(z,0)/L)^2, 1) * (1 + 0.2*(x^2+y^2)/4)"
driver.profile           = parsed
driver.density(x,y,xi)   = "6*exp(-(x^2+y^2)/(2*0.25^2)) * (xi > 1)*(xi < 4)"
witness.uz_mean(x,y,xi)  = "2000 - 40*(xi - 6.3)"      # energy chirp
witness.uz_std(x,y,xi)   = "2 + 10*abs(xi - 6.3)"       # slice energy spread varying along the bunch
witness.ux_std(x,y,xi)   = "0.05*(1 + (x^2+y^2)/0.05^2)" # transverse temperature growing with radius
electrons.uth(x,y,z)     = "0.001*(1 + z/1000)"         # plasma temperature
```

| feature | details |
|---|---|
| operators | `+ - * /`, `^` or `**` (right-assoc.), unary `- + !`, `< > <= >= == !=`, `&&`, `\|\|` (true = 1, false = 0) |
| functions | `sqrt exp log log10 sin cos tan asin acos atan sinh cosh tanh abs floor ceil erf heaviside(=step)`, `pow atan2 min max mod`, `if(cond, a, b)` |
| constants | `pi`, `e`, and every `my_constants.<name>` |

Errors name the position, e.g. `parser: unknown symbol 'q' … at position 2 in "q*2"`.

**Plasma**

| key | default | meaning |
|---|---|---|
| `plasma.density(x,y,z)` | — | parsed density / n₀. It replaces `plasma.radial`, `rmax`, `channel_*` and the tables. z is the lab position of the box head (the plasma entering the box). With `modes = 0` its azimuthal average is used; with `modes = 1` its value at each particle, so its m = 1 part is kept. |
| `<species>.density(x,y,z)` | — | the same, for one species only (e.g. a different ion profile) |
| `plasma.species` | `electrons` | list of species names |
| `plasma.radial` | `uniform` | `uniform` \| `channel` (n = 1 + depth·(r/r_c)²) \| `table` |
| `plasma.channel_depth`, `plasma.channel_radius` | | for `channel` |
| `plasma.radial_table` | | `r:n` pairs, piecewise linear |
| `plasma.rmax` | ∞ | plasma column radius |
| `plasma.z_table` | | `z:n` pairs; longitudinal profile at the lab position of the box front |
| `plasma.neutralize` | yes if no positive species | add an immobile neutralising background |
| `plasma.seed` | 1 | RNG seed (thermal plasma) |
| `<species>.charge` | −1 | charge in units of e |
| `<species>.mass` | 1 | mass in units of m_e |
| `<species>.mobile` | 1 | 0 = static background |
| `<species>.ppc` | 4 | rings per radial cell. Fine cells automatically get more rings per unit volume. 0 = initially empty (mobile species, e.g. to receive ionization electrons). |
| `<species>.ntheta` | 1 (modes=0), 8 (modes=1) | particles per ring, at equidistant azimuths (≥ 4 with modes = 1) |
| `<species>.uth` or `<species>.uth(x,y,z)` | 0 | isotropic thermal momentum spread (m_s c): a number or a function of position (z = lab position). With `modes = 0`, the rms over the ring is used. |
| `<species>.u_drift` | 0 0 0 | momentum (m_s c) added at loading (tests, e.g. mono-energetic electrons); the stream density is `density`, the lab density density·γ/Δ |
| `<species>.tail_dump` | 0 | 1 = write the particles at the end of the box to `tail_<species>_<step>.txt` (x y p_x p_y Δ w; last rank) |
| `<species>.density` | 1 | density multiplier relative to the profile (e.g. 1/Z for ions) |

**Laser** (envelope model; switched on by `laser.a0`)

| key | default | meaning |
|---|---|---|
| `laser.a0` | — | peak normalised vector potential at focus |
| `laser.k0` | — | ω₀/ω_p; or `laser.lambda0_um` (wavelength in µm) together with `units.n0_cm3` |
| `laser.w0` | — | spot size at focus: â ∝ exp(−r²/w0²) |
| `laser.L0` | — | length: â ∝ exp(−(ξ − ξ0)²/L0²) (intensity FWHM = 1.177 L0) |
| `laser.xi0` | — | centre of the pulse in ξ (keep ≳ 3 L0 from the head of the box) |
| `laser.focus` | 0 | distance from the initial position to the focal plane (> 0: the pulse is still converging) |
| `laser.polarization` | `linear` | `linear` (⟨a²⟩ = \|â\|²/2) or `circular` (⟨a²⟩ = \|â\|²) |
| `laser.l` | 0 | azimuthal index ℓ of a Laguerre–Gaussian vortex LG₀ℓ: â = a0 N_ℓ (√2 r/w0)^ℓ (1 + is)^{−(ℓ+1)} exp(−r²/(w0²(1 + is))) e^{iℓθ} × the ξ profile, s = −focus/Z_R; N_ℓ = (e/ℓ)^{ℓ/2}, so `laser.a0` is the peak of \|â\| (on the ring r = w0 (ℓ/2)^{1/2}). 0 = Gaussian |
| `laser.p` | 0 | radial index of the LG mode; only 0 is implemented |
| `laser.xi_scheme` | `bdf2` | ξ discretization of the envelope march: `bdf2` (second order, weakly dissipative, as in QuickPIC; default since 2026-11-10), `trapezoid` (neutral, second order; the earlier default), `euler` (first order) |
| `laser.file` | — | LASY file (instead of `laser.a0`, `w0`, `L0`, `focus`, `polarization`); needs HDF5 and `units.n0_cm3` |
| `laser.file_z` | 0 | distance from the pulse centre at t = 0 to the file's plane (1/k_p; > 0: the pulse still travels to it) |
| `laser.file_t_center_fs` | centroid | file time (fs) placed at `laser.xi0` |
| `laser.file_a0` | — | rescale the peak \|â\| at the file plane |
| `laser.file_iteration` | last | iteration in the file |
| `units.n0_cm3` | — | plasma density (needed with `laser.lambda0_um`; also gives SI units in openPMD output) |

**Adaptive time step** (`time.adaptive = 1`). Before step n, rank 0 sets
Δt_n = (2π/N) / ω_β with N = `time.nt_per_betatron`, ω_β² = n_max/(2γ_eff),
γ_eff = min over the particles of all non-rigid beams of max(γ, `adaptive_gamma_min`)·m/|q|
(ω_β² = (|q|/m) n/(2γ)), and n_max the largest plasma density at the box head during the step
(the quasi-static model takes the position z as the slow coordinate, so the plasma density of the whole box is the one at its head; electron species, or the
charge density of the positive species in a pure ion channel), clamped to [`dt_min`, `dt_max`].
With MPI, γ_eff is known globally only with a delay: every rank sends its minimum after the push
of step m to rank 0 (small messages against the pipeline), which uses step m = n − lag with
lag ≥ P (default P + 1, so that rank 0 does not wait for the tail). A decreasing γ_eff is
extrapolated linearly to t_{n+1}; an increasing one is not (the lagged value is then the
conservative one). During the first lag steps no rate is known yet. The step is sent down the
pipeline with the step-n message, so all ranks use the same Δt_n and the result does not
depend on timing: a serial run with the same `time.adaptive_lag` is bit-identical to an MPI run.
The beam leapfrog takes variable steps (kick (Δt_{n−1} + Δt_n)/2, drift Δt_n); with a constant
step it is the usual scheme, bit for bit. Output: `timestep.txt` (step, t, Δt, γ_eff, n_max) and
`gamma_records.txt` (true global γ_eff after every push, for checking). Checkpoints carry the
state (`adapt_rank_*.txt`). Not available with a laser (fixed step of the envelope solver).

The time step `time.dt` is shared with the beams. The laser accuracy is
controlled by Δt relative to the Rayleigh length k₀w0²/2 and to the
dephasing/depletion length. In the tests below, Δt ≤ Z_R/8 was sufficient.

**Ionization** (per ionizable plasma species; needs `units.n0_cm3`)

| key | default | meaning |
|---|---|---|
| `<species>.element` | — | H, He, Li, C, N, O, Ne, Ar (all levels), Kr (8), Rb (3), Xe (3), Cs (2): ionization energies and mass |
| `<species>.ionization_energies_eV` | from the element | list of ionization energies (overrides the table) |
| `<species>.z0`, `<species>.z_max` | 0, all levels | initial and highest charge state |
| `<species>.mass` | from the element | in electron masses |
| `<species>.mobile` | 1 | 0: ions stay at their positions (still macro-particles: their charge changes) |
| `<species>.ionization.product` | `electrons` | species that receives the new electrons (mobile, negative; `ppc = 0` for an initially empty one) |
| `<species>.trap` | 0 (1 if it receives delta electrons) | 1 = particles of this (mobile, non-ionizable) species that reach the quasi-static limit become beam particles (§1) |
| `<species>.trap_beam` | `<species>_trapped` | name of that beam (created automatically; set its options under this name) |
| `<species>.trap_nphi` | 1 | `modes = 0`: beam particles per converted ring (equidistant azimuths, random offset) |
| `<species>.ionization.field` | 1 | ADK by the plasma, beam and laser fields |
| `<species>.ionization.impact` | 1 if a cross-section is known | beam impact ionization 0 → 1 |
| `<species>.impact.M2`, `<species>.impact.C` | He, Ne, Ar, Kr, Xe, Rb: built in (§1) | Bethe parameters |
| `<species>.impact.sigma_cm2` | — | constant impact cross-section instead |
| `<species>.impact.secondary_energy` | 0 | 1 = delta electrons: impact-born electrons get energy and direction (§1) |
| `<species>.impact.J_eV` | He, Ne, Ar, Kr, Xe: built in (§1) | width J of the delta-electron spectrum 1/(J² + W²) |
| `<species>.impact.secondary_angle` | `kinematic` | `kinematic` (binary collision) or `isotropic` |
| `<species>.impact.inner_shells` | 1 | free-electron tail of the inner subshells (built in: Kr n = 1, 2; Xe n = 1–3) |
| `<species>.impact.inner_shells_eV`, `.inner_shells_n` | — | own subshells: binding energies (eV) and electron numbers (lists) |
| `<species>.ionization.born_dump` | 0 | 1 = write the electrons born in every sweep to `born_<species>_<step>.txt` (last column: 0 field, 1 beam impact, 2 electron impact) |
| `<species>.ionization.electron_impact` | 0 | 1 = ionization of all levels by the plasma electrons (§1) |
| `<species>.eimpact.table_<z>` | Lotz (§1) | file `E_eV sigma_cm2` for the transition z → z+1 |
| `<species>.eimpact.q` | from the configuration | custom gases: electrons in the outermost subshell, one number per level |
| `<electrons>.electron_impact_projectile` | 1 | 0 = this electron species does not ionize |
| `<species>.excitation` | 1 where data exist (Xe) | excitation loss of the projectiles on the neutral atoms |
| `<species>.excitation.table`, `.excitation.energy_eV` | — | own total excitation cross-section (`E_eV sigma_cm2`) and its energy loss |
| `ionization.electron_loss` | `discrete` | `discrete` collisions (straggling) or `continuous` slowing down of the projectile electrons |
| `<species>.ionization.quiet` | 0 | 1 = quiet ionization: accumulator emission with interleaved phases, low-discrepancy draws of the new electrons (see Ionization) |
| `<species>.ionization.nsplit` | 1 | > 1: split the first ionization into quanta of w₀/nsplit. Use it when the ionized fraction is small; choose nsplit ≳ 1/P_total. The particle count grows up to nsplit-fold where the gas is fully ionized. |

The `<species>.density` of an ionizable species is the atom density. Neutral gas does not
count as a positive species, so a pre-ionized background electron species still gets its
neutralizing background. A partly pre-ionized gas (`z0 > 0`) needs its electrons as a
separate species.

**Beams** (`beams = name1 name2 …`)

| key | default | meaning |
|---|---|---|
| `<beam>.charge`, `<beam>.mass` | −1, 1 | e.g. protons: 1, 1836.15 |
| `<beam>.profile` | `gaussian` | `gaussian` \| `flattop` \| `file` \| `parsed` |
| `<beam>.n0` | — | peak density / n₀ (of the uncut bunch) |
| `<beam>.sigma_r` | — | rms size per transverse coordinate (σ_x = σ_y) |
| `<beam>.sigma_xi` / `<beam>.length` | — | gaussian rms length / flat-top length |
| `<beam>.xi0` | — | centre position |
| `<beam>.xi_cut_head` | — | drop particles with ξ < value (seeded SMI) |
| `<beam>.gamma` | — | mean γ |
| `<beam>.espread` | 0 | rms relative energy spread |
| `<beam>.emit_n` | 0 | normalised rms emittance per plane (c/ω_p) |
| `<beam>.cut` | 3 | truncation in σ (radial and longitudinal) |
| `<beam>.nparticles` | 100000 | macro-particles |
| `<beam>.seed` | 12345 | RNG seed |
| `<beam>.rigid` | 0 | 1 = frozen beam |
| `<beam>.passive` | 0 | 1 = test particles: pushed, but no charge or current deposited |
| `<beam>.analytic` | 0 | 1 = noise-free rigid beam deposited analytically on the grid (gaussian/flattop, offsets and slopes allowed) |
| `<beam>.x0`, `<beam>.y0` | 0 | transverse centroid offset (modes = 1) |
| `<beam>.x_slope`, `<beam>.y_slope` | 0 | centroid tilt: x_c(ξ) = x0 + x_slope (ξ − ξ0) (modes = 1) |
| `<beam>.xp0`, `<beam>.yp0` | 0 | centroid angle, p_x = p_z·xp0 (modes = 1) |
| `<beam>.nsym` | 1 | quiet start: replicate every particle at nsym azimuths (nparticles must be a multiple) |
| `<beam>.mirror_y` | 0 | quiet start: also add the mirror image y → −y (keeps an x-only offset/tilt exactly y-symmetric) |
| `<beam>.file` | | for `profile = file`: columns x y p_x p_y p_z ξ w |

**Parsed beams** (`<beam>.profile = parsed`)

| key | default | meaning |
|---|---|---|
| `<beam>.density(x,y,xi)` | — | beam density / n₀; the third variable is the co-moving ξ |
| `<beam>.xi_range` | — | ξ interval that is sampled |
| `<beam>.r_max` | — | radius that is sampled |
| `<beam>.ppc` | `1 1` | sub-rings per radial grid cell, sub-slices per ξ cell |
| `<beam>.ntheta` | 1 (modes=0), 8 (modes=1) | particles per ring |
| `<beam>.rods` | 0 | > 0 (even): *rods*: that many momentum samples per radial position (antithetic pairs ±g, scaled to exact second moments per shell), each kept at every ξ. The transverse mixing of the rods then stays smooth along ξ: no macro-particle noise along the beam, also after the bunch has evolved (the default loader, with independent momenta per particle, is quiet at t = 0 but 10⁴–10⁵ times above the real shot noise after 1 m of drift; section 25). |
| `<beam>.dr` | grid cells | radial shell width of the loader (default: the cells of the radial grid) |
| `<beam>.shot_noise` | 0 | amplitude factor of the shot noise of the real particles (1 = physical, needs `units.n0_cm3`): the weight of every macro-particle (N = w n₀ (c/ω_p)³ real particles) is redrawn from Poisson(N) (Gaussian above N = 100). Each phase-space cell then has the count noise of the real beam: the right noise power at every k up to 1/dξ and for every radial kernel (section 25). For the self-modulation instability without a seed. |
| `<beam>.ux_mean(x,y,xi)`, `uy_mean`, `uz_mean` | 0, 0, √(γ²−1) | mean momenta p/mc (a plain number is accepted too); `uz_mean` or `gamma` is required |
| `<beam>.ux_std(x,y,xi)`, `uy_std`, `uz_std` | 0 | Gaussian spread of p_x, p_y, p_z. Each may depend on position, e.g. `ux_std(x,y,xi) = "emit/sr * (1 + 0.5*xi)"`; a plain number is accepted. Negative values count as 0. |
| `<beam>.u_std` | `0 0 0` | shorthand: three constant spreads (overridden component-wise by `ux_std`, etc.) |
| `<beam>.analytic = 1` | | rigid and noise-free: the parsed density is deposited directly on the grid (32-point azimuthal quadrature for the m = 0, 1 parts); needs `gamma` |

The particles are placed deterministically (quiet start), so no sampling
noise is created:
- **radial:** on the radial cells of the simulation grid, which means a fine
  grid near the axis also gives fine sampling there;
- **ξ:** on sub-slices of the ξ grid;
- **azimuth:** at equidistant angles.

The weight of each particle is density × volume, and particles with zero
density are skipped.

**Output**

| key | default | meaning |
|---|---|---|
| `output.dir` | `out` | output directory |
| `output.every` | 1 | write fields every n steps (0 = never) |
| `output.field_files` | 1 | 0: do not write the 2D field files `fields_*.bin` at these steps, only the on-axis data `axis_*.txt` (long runs, fine Δξ) |
| `output.beam_every` | 0 | write beam particles every n steps |
| `output.fields` | `all` | field components of the main output: `all`, `none`, or a list of `psi ez er eth br bth bz ne ni rhob nz_<species> a` (`a` = laser envelope; with m = 1 a name includes its `_c`, `_s` parts) |
| `output.beams` | `all` | beams whose particles are written: `all`, `none` or a list of names |
| `output.rmax` | R | write the fields only for r ≤ rmax (up to the first node ≥ rmax); also limits the uniform openPMD grid |
| `output.xi_stride` | 1 | write every n-th slice (ξ = ξ_min, ξ_min + nΔξ, …) |
| `output.particle_stride` | 1 | write every n-th beam particle (per rank) |
| `output.axis` | 1 | write `axis_*.txt` with the field output |
| `output.beam_slices` | 0 | number of ξ-bins for per-slice beam diagnostics (centroid, size, γ, ε) written at field outputs |
| `output.beam_slices_range` | box | `lo hi`: ξ-range of these bins (default: the whole box); the bins are fixed, so files at different steps line up |
| `output.format` | `native` | `native` (QUARZ binary/text files), `openpmd`, or `both`. With `openpmd`, the native field files and particle dumps are not written; `axis_*`, `beams.txt` and `slices_*` always are. |
| `output.openpmd_backend` | `h5` | `h5` (HDF5), `bp` (ADIOS2) or `json` |
| `output.openpmd_file` | `openpmd/data_%06T` | file name pattern inside `output.dir` (`%T` = step: one file per output) |
| `output.openpmd_grid` | `uniform` | `uniform`: fields interpolated onto a uniform radial grid (openPMD requires uniform spacing; readable by openPMD-viewer, yt, …); `native`: the code's non-uniform nodes, exact, geometry `other` |
| `output.openpmd_dr`, `output.openpmd_rmax` | h_min, R | spacing and extent of the uniform radial grid (default h_min, at most 8192 points) |
| `output.openpmd_options` | `{}` | JSON/TOML options passed to openPMD-api (compression, ADIOS2 engine, …) |
| `units.n0_cm3` | — | plasma density for SI units in the openPMD files; without it the data are in normalised units (all `unitSI` = 1) |

**Diagnostic groups.** Additional outputs, each with its own period and selection, so that
frequent small dumps and rare full dumps can be combined (§6):

| key | default | meaning |
|---|---|---|
| `diag.names` | — | names of the groups, e.g. `diag.names = light` |
| `diag.<name>.every` | 0 | field output period of the group (steps; 0 = no fields) |
| `diag.<name>.beam_every` | `every` | particle output period of the group |
| `diag.<name>.fields` | `all` | as `output.fields` |
| `diag.<name>.beams` | `all` | as `output.beams` |
| `diag.<name>.rmax`, `.xi_stride`, `.particle_stride` | R, 1, 1 | as `output.rmax`, … |
| `diag.<name>.axis` | 0 | also write `axis_*.txt` into the group directory |
| `diag.<name>.format` | `output.format` | `native`, `openpmd` or `both` |
| `diag.<name>.field_files` | 1 | as `output.field_files` |
| `diag.<name>.openpmd_*` | `output.openpmd_*` | openPMD settings of the group (backend, file pattern, grid, dr, rmax, options) |

**Checkpoints and restart**

| key | default | meaning |
|---|---|---|
| `checkpoint.every` | 0 | write a checkpoint every n steps (0 = never) |
| `checkpoint.at_end` | 1 if `every` > 0 | write a checkpoint at the end of the run (to continue it with a larger `time.steps`) |
| `checkpoint.keep` | 2 | keep the newest n complete checkpoints, delete older ones (0 = keep all) |
| `checkpoint.dir` | `<output.dir>/checkpoints` | directory of the checkpoints |
| `restart.from` | — | restart from a checkpoint directory (`.../chk_NNNNNN`) or `latest` (newest complete checkpoint in `checkpoint.dir`) |
| `boundary.write` | — | stage 1 of a two-stage run: file for the plasma state leaving the box, every step (see "Two-stage runs") |
| `boundary.read` | — | stage 2: take the plasma entering the box from this file instead of fresh plasma |

**Physical units.** At plasma density n₀ [cm⁻³]:

- c/ω_p = 5.31·10⁵ / √n₀ cm
- E₀ = 0.96 √n₀ V/m
- a beam weight w corresponds to w·n₀(c/ω_p)³ particles.

