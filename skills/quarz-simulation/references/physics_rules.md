# QUARZ: model, resolution, time step, box, cost

## The model in a few lines

- Quasi-static: for every beam time step the plasma is swept slice by slice from the head of the box
  (`xi.max`) to its tail, with the beams frozen. Then the beams are pushed by `time.dt` in the stored fields.
  All slow-time derivatives of the plasma response are dropped.
- Geometry: (r, theta, xi) with azimuthal modes m = 0 (`modes = 0`, axisymmetric) or m = 0 and 1 (`modes = 1`).
- Radial grid: any monotone set of nodes (`grid.type = regions`), a conducting wall at `grid.rmax`.
- Plasma particles are rings (with `modes = 1`: several particles per ring); beams are 3D macro-particles.
- Lasers: envelope model (ponderomotive force on the plasma, envelope equation for the pulse).
- Units: lengths c/omega_p, times 1/omega_p, densities n0, momenta m c of the species, fields m_e c omega_p / e.
  xi = z - ct: the beams move towards +xi, the head is at the largest xi.

## What it can and cannot give

| can | cannot |
|---|---|
| wakes of electron, positron, proton, muon beams; several beams | trapped plasma electrons as plasma (removed, or handed to a beam with `<species>.trap`) |
| self-modulation, hosing (m = 1), beam loading, ion motion | m >= 2: elliptical beams, offsets comparable to the beam size |
| plasma profiles in r and z, formulas, finite columns | radiation, the laser carrier, backward waves, reflection |
| laser envelope, Laguerre-Gaussian vortex, LASY files | few-cycle pulses, strong red-shift of a depleted pulse |
| field and impact ionization, ionization injection into a beam | recombination; laser energy loss to ionization |
| radiation reaction, Coulomb scattering, spin of the beam | a converged height of the E_z spike at the bubble closure |

## Step by step: a beam-driven run

1. `python3 scripts/quarz_numbers.py --density ... --particles ... --species ... --energy-gev ... --sigma-r-um ...
   --sigma-z-um ... --emit-n-um ... --length-m ...`: code units, `<beam>.n0`, betatron period, `time.t_end`.
2. Regime from n_b/n0 and k_p sigma_r: n_b/n0 << 1 linear (wake of the order n_b/n0 E0 or below; a long bunch
   drives little until it self-modulates); n_b/n0 > 1 and k_p sigma_r < 1 blowout (bubble radius about 2 sqrt(Lambda),
   Lambda = n_b/n0 (k_p sigma_r)^2).
3. Box: `xi.max` just ahead of the head (3 sigma, or the cut), `xi.min` behind the region of interest.
   `xi.step` 0.05 (linear), 0.005-0.01 (blowout).
4. Radial grid: `grid.regions` with cells <= sigma_r/10 over the narrowest beam, 0.01-0.02 over the bubble,
   up to 0.05 outside in a short box; in a box longer than about 300 not more than 0.02-0.03 inside the plasma,
   0.1 only beyond `plasma.rmax` (rule 9 of SKILL.md); `grid.rmax` 8-14.
5. `time.dt` <= betatron period / 20 of the lowest-energy particles that matter; `time.steps` or `time.t_end`
   from the length.
6. Plasma: `plasma.z_table` for ramps and gradients (z in c/omega_p from the start of the run), `plasma.rmax`
   for a finite column; mobile ions as a second species when the beam can move them within its own length:
   k_p sigma_xi sqrt((n_b/n0) m_e/m_i) approaching 1 or more.
7. Beam particles: 1e5-1e6 for short bunches, millions for long bunches whose instability must grow from the
   seed and not from the noise.
8. Output: `output.every` for the maps, `output.beam_every` for particles; thin the maps for long boxes.

## Long proton bunches (AWAKE-like)

- **Radial cells first** (rule 9 of SKILL.md): far behind the head the clustering plasma macro-particles make
  field noise that grows with the square of the cell size. Keep the cells inside the plasma column at 0.02-0.03
  or below, e.g. `grid.regions = 2.0:0.01 7.3:0.02 10.0:0.1` for a column of radius 7, and check the wake of the
  first step off the axis along the whole box before the long run.
- Bunch length hundreds of c/omega_p: the box is long (10^4 slices and more), the sweep takes seconds, the run
  a few hundred steps of some centimetres. See `examples/awake_density_gradient.in`.
- Seed: a sharp front (`<beam>.xi_cut_head`) stands for the ionization front of a co-propagating laser. `<beam>.n0`
  is the peak density of the uncut bunch; the particles ahead of the cut are dropped.
- A density gradient changes the plasma wavelength along z and with it the phase of the wake relative to the
  micro-bunches; give it with `plasma.z_table` relative to the reference density n0 (the density at z = 0 if the
  table starts with 1).
- What to look at: the largest on-axis E_z against z (growth, saturation, decay), its position in xi, the
  micro-bunch train in rho_beam on the axis, the charge that stays within the plasma radius.
- The wall: the defocused protons leave radially; `grid.rmax` must be well outside the plasma column, and
  particles that reach it are lost (see the charge in `beams.txt`).

## Step by step: a laser-driven run (envelope)

1. `python3 scripts/quarz_numbers.py --density ... --wavelength-um ... --a0 ... --w0-um ... --tau-fs ...`.
2. `units.n0_cm3` and `laser.lambda0_um` (or `laser.k0`), `laser.a0`, `laser.w0`, `laser.L0`, `laser.xi0`
   (>= 3 L0 from the head of the box), `laser.focus`.
3. `xi.step` 0.01-0.02 for a nonlinear wake; radial cells 0.005-0.02 inside the bubble.
4. `time.dt` <= Rayleigh length / 8, and small against the dephasing and depletion lengths.
5. If the pulse is short against its wavelength count (a few cycles), red-shifts strongly, or the question is
   about the carrier: this is outside the envelope model, use GEM-PIC-RZ.

## Cost and output volume

- Time per step = one sweep (slices x plasma particles per slice) + the beam push. The log prints
  `sweep ... s` for every step: measure it with 3 steps, multiply.
- Guide values measured with the examples (308 radial cells): 6000 slices, 4 rings per cell: 0.6 s per sweep
  on 8 threads; 20000 slices: 5 s on 16 threads; 64 rings per cell instead of 4 cost 5 times as much;
  20000 slices with 512 radial cells, 4 rings per cell: 5.8 s on 32 threads.
  `modes = 1`: about ten times more.
- A field file: slices x radial nodes x components x 8 bytes (all 10 components of a 20000 x 308 grid: 490 MB).
  `output.fields`, `output.rmax`, `output.xi_stride` reduce it; `output.field_files = 0` keeps only the on-axis
  files.
- MPI helps runs with many steps (pipeline over the steps), not a single solve.

## Convergence

Halve one of `xi.step`, the radial cells (`grid.regions`), `time.dt`, and double the particles
(`<species>.ppc`, `<beam>.nparticles`), one at a time; compare the quantity the user asked for, not a spike.
Report the change. A result that moves by more than the accuracy the user needs is not finished.
