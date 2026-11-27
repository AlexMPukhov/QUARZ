---
name: quarz-simulation
description: Write, check, run and debug input files for the plasma wakefield code QUARZ (quasi-static particle-in-cell, RZ geometry, non-uniform radial grid, azimuthal modes m = 0 and 1). Use for beam-driven plasma wakefields (PWFA) with electron, positron or proton drivers, long proton bunches and self-modulation (AWAKE-like runs), witness bunches and beam loading, ion motion, hosing, laser wakefields in the envelope model, ionization, plasma density profiles and ramps over metres of plasma, and for choosing the grid and time step, reading the output (beams.txt, axis files, field files, openPMD) or finding out why a run gives wrong or noisy results. Not for full-PIC codes or for GEM-PIC-RZ. If quarz is not built yet, use the skill quarz-setup first.
license: BSD-3-Clause
compatibility: Needs a built quarz executable (skill quarz-setup) and a POSIX shell; python3 for the helper scripts (numpy for quick_look.py, matplotlib only for its figure).
metadata:
  repository: https://github.com/AlexMPukhov/QUARZ
---

# QUARZ: from a physics problem to a checked input file

## Is QUARZ the right code?

QUARZ is quasi-static: the plasma response is computed for a driver that is frozen while the plasma passes it,
and the driver then moves with a large time step. That is what makes metres of plasma cheap, and it sets the limits.

| the problem | code |
|---|---|
| beam-driven wakes over long distances (electron, positron, proton drivers; self-modulation of long bunches; witness, beam loading, ion motion, hosing with small offsets), laser wakes where the envelope model holds | **QUARZ** |
| the laser carrier matters (few-cycle pulses, strong red-shift and depletion, structured light beyond the envelope model, flying focus), radiation or FEL, undulators, dielectric or metal structures | **GEM-PIC-RZ** (github.com/AlexMPukhov/GEM-PIC-RZ): resolves the carrier, keeps the forward radiation; it is not quasi-static |
| self-injection at the bubble closure, down-ramp injection, backward waves, reflection, solid targets | a full PIC code (FBPIC, WarpX, Smilei, EPOCH) |
| strongly non-axisymmetric beams (m >= 2: elliptical beams, offsets comparable to the beam size) | a 3D quasi-static code (HiPACE++, QuickPIC) or 3D PIC |

Inside QUARZ, plasma electrons that become trapped cannot be followed as plasma: they are removed, or (with
`<species>.trap`, ionization injection) handed over to a beam. A trapped charge from QUARZ is a statement about
that hand-over, not a full-PIC result.

## What this skill is for, and how to work with the user

The purpose is to help the user **run the code for their physics problem**: turn the problem into an input file,
explain the choices, run it and read the result. Work in this order:

1. **Talk about the physics first.** Before writing an input, make sure you know, and ask in one short message
   for what is missing (offer a default for each item, so that the user can answer "go ahead"):
   - the driver: beam (species, energy, number of particles or charge, rms sizes, emittance, a cut or seed) or
     laser (wavelength, a0, spot, duration, focus);
   - a witness, if any (charge, sizes, position, energy);
   - the plasma: density, profile along z (ramps, gradients, length), column radius, gas species if ionization or
     ion motion matters;
   - what is to be measured: wake amplitude and phase along z, energy gain and quality of a witness, the
     micro-bunch train, hosing;
   - the propagation length, and first look or production accuracy.
   If the user wants no questions, or the request leaves only numerical choices open, go on with stated assumptions.
2. **Say what you assume and discuss the choices, in physics terms:** the box (what it holds, where the head is),
   the resolution along xi and r and what it resolves, the wall radius, the time step against the betatron period,
   which physics is on or off (mobile ions, ionization, m = 1, smoothing of the bubble closure) and what the
   quasi-static model cannot give (table above). Name the numbers (c/omega_p, E0, n_b/n0, k_p sigma_r, betatron
   period).
3. **Then the input file,** with a comment per group of keys, the estimated run time and output volume, and the
   checks to make on the first output ("Verify before delivering" below).
4. **Run it if asked,** look at the result with the user, and propose the next step (convergence run, longer
   propagation, parameter scan).

**Hard rule: do not modify the QUARZ source code.** Building and running the code are allowed; patching it is not.
If the problem needs something the code cannot do, say so, name the limit and the code that fits (table above),
and suggest opening an issue at https://github.com/AlexMPukhov/QUARZ/issues. Debugging means debugging the input
and the run (resolution, time step, box, noise), not the code.

## Start here

Paths are relative to this skill's directory.

1. `python3 scripts/quarz_numbers.py --density <n0 in cm^-3> ...` converts the problem to code units and prints
   starting values with the key they belong to (`--help` lists the options).
2. Copy the closest file of `examples/` and change it. Do not write an input from memory: the keys are in
   `references/input_reference.md` (the complete list, generated from the code's README).
3. `sh scripts/check_input.sh <quarz> my.in`: one quasi-static solve; fails on errors and on keys the code never
   read (QUARZ itself only warns about those at the end of a run).

More: `references/physics_rules.md` (model, resolution, time step, box, cost), `references/diagnostics.md`
(output files, units, what to look at), `references/debugging.md` (symptoms and causes),
`references/output_reference.md` (file formats, generated from the README).

## Examples

| file | what | cost |
|---|---|---|
| `examples/awake_density_gradient.in` | AWAKE-like: 4e11 protons, 400 GeV, sigma_z = 5 cm, cut one sigma ahead of the centre, 7e14 cm^-3 rising by 5 % over 10 m; whole bunch in the box, cells of 0.02 over the plasma (rule 9) | 250 steps, 24 min on 32 threads |
| `examples/awake_seeded_smi.in` | seeded self-modulation of a long proton bunch cut at its centre, uniform plasma | set to 3 steps; 250 steps = 10 m |
| `examples/blowout_basic.in` | blowout of a dense electron driver: one quasi-static solve (rigid beam) | seconds |
| `examples/pwfa_ion_motion.in` | driver and a strongly pinched, low-emittance witness, mobile hydrogen ions, 5e-4 cells on the axis | 100 steps, minutes |
| `examples/lwfa_laser.in` | laser wake in the envelope model (0.8 um, a0 = 4, matched spot) with a witness in the first bucket | 100 steps, about 1 min |

Further inputs in the repository's `examples/`: hosing (`modes = 1`), profiles from formulas, ionization, Trojan
horse.

## The input in one page

`key = value` statements, one per line or separated by `;`; `#` starts a comment. Keys are written flat
(`xi.step = 0.01`) or in blocks (`xi { step = 0.01; min = -12 }`, `bunch driver { ... }`, `species electrons { ... }`,
`diag light { ... }`: the named blocks add their name to `beams`, `plasma.species`, `diag.names`); a key given twice
is an error. A number without a unit is in plasma units of the reference density n0:
lengths in c/omega_p, times in 1/omega_p, densities in n0, momenta in m c, fields in E0 = m c omega_p / e. A number
may carry a unit (`8 um`, `90 fs`, `5e16 cm^-3`, `2 mrad`, `1 %`) once `units.n0 = 1e17 cm^-3` is given; mind that
the reference density also switches on the Coulomb scattering of the beams (`<beam>.scattering`). Functions stay
in plasma units. The names common to the VLPL family are accepted too and rewritten into the keys below
(`references/input_reference.md`, "Common names"): `grid.m_max`, `grid.r_max`, `laser.wavelength`, `laser.tau_fwhm`,
`laser.waist`, `laser.z_focus`, and for a bunch `species`, `charge` with a unit (= total charge), `energy`,
`energy_spread`, `sigma_x`, `sigma_y`, `sigma_z`, `emittance_x`, `emittance_y`, `zeta_center` (from the rear end of the
box), `macroparticles`, `file_species`, `file_iteration`. The
co-moving coordinate is **xi = z - ct: the beams move towards +xi, the head of the box is at `xi.max` (often 0)
and the box extends to negative xi**; the plasma is swept from the head (large xi) to the tail. A beam's head is at
its largest xi, and a witness behind a driver is at smaller xi. (QUARZ v1.0.0 used xi = ct - z, head at small xi:
convert old inputs with `tools/convert_deck.py` of the repository. Every input states `xi.convention = z-t`;
inputs without it are refused.)

```quarz
grid.type      = regions                # radial grid: fine near the axis, coarser outside
grid.rmax      = 8.0                    # conducting wall
grid.regions   = 1.0:0.01  8.0:0.05     # spacing 0.01 for r < 1, growing to 0.05 for r < 8
grid.max_ratio = 1.05

xi.convention = z-t                     # required (since v2.0.0): states the convention xi = z - t
xi.min  = -12                           # tail of the box
xi.max  = 0                             # head of the box
xi.step = 0.01

time.dt    = 20                         # beam time step; a step moves the box by dt c/omega_p
time.steps = 100                        # 0 = one quasi-static solve

plasma.species = electrons              # an immobile neutralising background is added automatically
electrons.ppc  = 4

beams = driver
driver.n0         = 2.0                 # peak density / n0
driver.sigma_r    = 0.3
driver.sigma_xi   = 1.0
driver.xi0        = -4.0                # centre; keep the bunch (cut = 3 sigma) inside the box
driver.gamma      = 20000
driver.emit_n     = 0.1
driver.nparticles = 200000

output.dir   = out
output.every = 25
```

Species and beams are named by the user: `plasma.species = electrons ions` makes the keys `electrons.ppc`,
`ions.mass`, ...; `beams = driver witness` makes `driver.n0`, `witness.xi0`, ... (written `<species>.key` and
`<beam>.key` in the reference). Protons: `<beam>.charge = 1`, `<beam>.mass = 1836.15`. A long bunch with a sharp
front: `<beam>.xi_cut_head`. Plasma profile along z: `plasma.z_table` (z:n pairs) or a formula
`plasma.density(x,y,z)`. Every key can be overridden on the command line: `quarz my.in time.steps=10`.

## Rules that decide whether the result is right

1. **Resolve the plasma wave and the beam along xi.** `xi.step` <= 0.05 for linear wakes (126 slices per plasma
   wavelength), 0.005-0.01 for a blowout; at least 10 slices per sigma_xi of a short bunch.
2. **Radial cells: fine over the beam, and fine enough over the whole plasma.** Cells <= sigma_r/10 over the
   beam (a pinched witness needs cells well below its matched size), 0.01-0.02 where the sheath of a bubble moves.
   Outside the beam the cells may grow (`grid.type = regions`, `grid.max_ratio` 1.05), to 0.05 in a short box.
   In a box longer than about 300 c/omega_p keep them <= 0.02-0.03 everywhere inside the plasma (rule 9) and let
   them grow to 0.1 only beyond `plasma.rmax`, where there are no plasma particles and cells cost little.
3. **The wall is a conductor:** `grid.rmax` several plasma skin depths beyond the beam and the bubble (the
   examples use 8-14). A finite plasma column is set with `plasma.rmax`, inside the wall.
4. **Time step against the betatron period** of the lowest-energy beam particles, 2 pi sqrt(2 gamma m/|q|)
   in an ion channel of density n0: `time.dt` <= period/20 (`scripts/quarz_numbers.py` prints it). Mobile ions or a
   laser need less (laser: Rayleigh length / 8). Beams whose energy changes much: `time.adaptive = 1`.
5. **The box must hold what matters:** the whole driver (or its part behind the cut) plus the region of interest
   behind it. The plasma ahead of `xi.max` is unperturbed, so the box may start just ahead of a cut, but a laser
   must stay about 3 `laser.L0` behind the head.
6. **The density of the whole box is the density at its head** (`plasma.z_table`, `plasma.density(x,y,z)` are
   evaluated there). That is the quasi-static model, not an error of the code: a ramp is seen by the beam with
   the box as it passes.
7. **The spike of E_z at the closure of a bubble does not converge** in a quasi-static code; the fields behind
   it must. Use `plasma.smooth_length = 0.005` with the axis cell and `xi.step` not larger than that.
8. **Noise:** a long, weak bunch seeds its own instability from macro-particle noise. Use enough particles
   (millions for a self-modulation run), a seed that dominates the noise (a cut, `<beam>.xi_cut_head`), and for
   unseeded runs the quiet loaders described in the reference (`<beam>.profile = parsed` with `<beam>.rods`,
   `<beam>.shot_noise`).
9. **Long boxes: the plasma macro-particles cluster, and coarse cells turn that into field noise.** Behind the
   first tens of c/omega_p the cold plasma rings gather into clusters on the scale of a cell (Kargapolov et al.,
   Phys. Plasmas 32, 023905 (2025)). The density noise saturates at a level that falls only as 1/sqrt of
   `<species>.ppc`; the field it makes grows about as the square of the cell size, because a density ripple of
   wavelength h makes a potential proportional to h^2. Hundreds of c/omega_p behind the head a wake on coarse
   cells is noise. Measured with `examples/awake_density_gradient.in` at the first step (box of 1000, 4 rings per
   cell, the wake averaged over r < 1 in the last tenth of the box, where the seed wake is 0.001 E0): cells of
   0.02 inside the plasma 0.001, 0.03: 0.001, 0.05: 0.002, 0.1: 0.016-0.023. With 0.1 cells, 16 rings per cell
   (twice the time) give 0.0024; with 0.02 cells even one ring per cell gives 0.0013. So for a box longer than
   about 300: **first make the cells inside the plasma column fine (<= 0.02-0.03), then raise `<species>.ppc`**
   (4 -> 16 helps on cells of 0.03-0.05), and always look at the wake of the first step along xi
   (`scripts/quick_look.py` prints it in ten windows). Judge it off the axis (E_z at r = 0.5, or averaged over
   r < 1): on a fine mesh a single ring at the axis node makes a density spike of 10-100 n0 and a jump of the
   on-axis E_z that overstates the damage. Tell the user where the box stops being reliable. `plasma.smooth_length`
   also removes the noise of the first step, but it filters all plasma sources, also the fine radial structure
   around a self-modulating beam: not without a convergence check in the smoothing length.
10. **m = 1 costs about ten times m = 0** and covers small offsets and tilts only; keep `modes = 0` unless the
   question is hosing or misalignment.

## Verify before delivering

1. `sh scripts/check_input.sh <quarz> my.in` passes; read the summary it prints (slices, radial cells, particles
   per slice, what is written).
2. A short run (`time.steps=3 output.every=1` on the command line, so that every step writes its axis file) for the time per step (`sweep ... s` in the log): estimate the
   full run and the output volume (a field file is slices x radial nodes x components x 8 bytes; thin it with
   `output.fields`, `output.rmax`, `output.xi_stride`).
3. `python3 scripts/quick_look.py <output.dir> --density <n0>` after the first output steps: is the wake where
   and as large as expected (linear theory, the numbers from step 1), does the beam keep its charge, and is the
   wake of the first step free of a jump along xi (rule 9)?
4. For a result that will be used: repeat with `xi.step`, the radial cells and `time.dt` halved (one at a time)
   and report how much the measured quantity moved.
