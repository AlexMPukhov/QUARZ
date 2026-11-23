# QUARZ: when a run fails or looks wrong

Debug the input and the run. Do not patch the code: if the input is right and the code still fails, report it to
the user with the input and the log, and suggest an issue at https://github.com/AlexMPukhov/QUARZ/issues.

## The check fails

| message | cause |
|---|---|
| `UNKNOWN KEY: input key '...' was not used` | misspelt key; a key that does nothing in this setup (e.g. `time.nt_per_betatron` without `time.adaptive = 1`); or a `<beam>.key` whose beam is not listed in `beams` (the same for `plasma.species`) |
| `ERROR (rank 0): ...` | wrong value or missing key; the message names it (e.g. `required key 'driver.length' is missing` for `<beam>.profile = flattop`). Parsed formulas: `parser: unknown symbol ...` names the position |
| a laser key is refused | `laser.lambda0_um` needs `units.n0_cm3`; `laser.file` needs a build with HDF5 |

Look up every key in `input_reference.md`; do not guess names from other codes (QUARZ has no blocks, no `--check`
switch, and its lengths are in c/omega_p, not in metres).

## The run stops or blows up

| symptom | cause, cure |
|---|---|
| NaN, or fields growing from slice to slice behind a bubble | plasma particles cross many radial cells per slice at the closure: smaller `xi.step`, `pusher.max_cells_per_step = 1`, `plasma.smooth_length = 0.005`; keep `pusher.ab_order` at 2 or 3 |
| the witness emittance grows in the first steps | not matched, axis cell too coarse for the pinched beam (cells <= sigma_r/10), or `time.dt` too large (<= betatron period / 20) |
| the beam loses charge at once | the bunch does not fit into the box (`xi.min`, `xi.max` against `<beam>.xi0` and 3 sigma), or `grid.rmax` cuts it |
| results change with the number of MPI ranks | compare with a serial run using `beams.xi_shape = ngp`: MPI runs use the nearest-slice shape |
| segmentation fault right after the start-up summary | compiler/runtime mismatch of the build (skill quarz-setup, README section 4) |

## The result looks wrong

| symptom | cause, cure |
|---|---|
| wake much smaller or larger than the estimate | `<beam>.n0` is the peak density in units of n0 (not a charge); recompute with `scripts/quarz_numbers.py`; check the particle number from the `charge` column of `beams.txt` |
| the wake has the wrong sign or the beam sits at the wrong end | xi = z - ct: the head is at the largest xi (`xi.max`). A witness goes behind the driver, i.e. at smaller xi. An input written for QUARZ v1.0.0 (xi = ct - z) runs mirrored: convert it with `tools/convert_deck.py` |
| the wake jumps far behind the head, at a place that moves with `<species>.ppc` or the grid | the log has `Note: the box is ... long and the radial cells inside the plasma ... reach h = ...`: clustering of the plasma macro-particles in a long box, turned into field noise by coarse radial cells (rule 9 of SKILL.md): cells <= 0.02-0.03 everywhere inside the plasma, then more `<species>.ppc`; judge the wake off the axis; say where the box stops being reliable |
| a self-modulation run differs from run to run, or grows without a seed | macro-particle noise seeds it: more particles, a real seed (`<beam>.xi_cut_head`), or the quiet loaders (`<beam>.profile = parsed`, `<beam>.rods`) |
| spike of E_z at the end of the bubble changes with resolution | expected in a quasi-static code; compare the fields behind it, use `plasma.smooth_length` |
| a density ramp seems to act on the whole box at once | that is the model: the density of the box is the one at its head (rule 6 of SKILL.md) |
| wake reflected or distorted near the wall | `grid.rmax` too small for the plasma response; enlarge it (coarse outer cells are cheap) |
| laser diffracts or focuses differently from a full PIC run | envelope model: check `time.dt` <= Rayleigh length / 8 and the resolution of the pulse; few-cycle or strongly depleted pulses are outside the model |

## Slow or too much output

- Time per step is the sweep: fewer slices (`xi.step` not finer than needed, a shorter box), fewer radial cells
  where nothing happens, `<species>.ppc` 4, `modes = 0`.
- Many steps: MPI (`mpirun -np P`), which pipelines the steps.
- Output: `output.every` larger, `output.fields` with the few components needed, `output.rmax`,
  `output.xi_stride`, `output.field_files = 0`.

## Pitfalls: defaults and habits that surprise

Each line gives the reason, so that an unusual case can be judged.

| what happens | why | what to do |
|---|---|---|
| a beam scatters, and its emittance grows, although nobody asked for collisions | multiple Coulomb scattering of non-rigid beams is on by default as soon as `units.n0_cm3` is given | say so when comparing with a code without it; `<beam>.scattering = 0` switches it off; for gases other than hydrogen set the scattering charge (reference) |
| a neutral gas is ionized by the beam itself | beam impact ionization is on by default for the gases with built-in cross-sections | `<species>.ionization.impact = 0` if only field ionization is wanted |
| the wake of a weakly ionized gas is noisy | the newborn electrons are few: their sampling noise falls only as 1/sqrt of their number. `<species>.ionization.quiet = 1` removes the bias of the random draws, not this noise | `<species>.ionization.nsplit` > 1 (about 1/ionized fraction) or more `<species>.ppc` |
| two runs of the same input differ in the 10th digit | with more than one OpenMP thread the order of the deposits is not fixed (round-off) | not physics. Bit-identical comparisons (also MPI against serial) need one thread and `beams.xi_shape = ngp` |
| a smoothing length that looked harmless changes the result later | `plasma.smooth_length` filters all plasma sources. At the first step a smooth beam shows no difference, but a growing instability lives on fine radial structure | use it at the scale of the axis cell (0.005 at a bubble closure, rule 7); for an evolving beam repeat the run with half the length before trusting it |
| a stretched grid and a uniform grid disagree by a few per cent | the outer cells are too coarse where the return current and the wake still live | a convergence test halves the outer cells too, not only the axis cells (rules 2 and 9) |
| sub-slicing does not remove the noise of a long box | `pusher.max_cells_per_step` cures particles that cross many cells per slice at a bubble closure; the noise of long boxes has another cause (rule 9) | refine the radial cells inside the plasma |
| `time.adaptive = 1` is refused with a laser | the envelope solver needs a fixed step | fixed `time.dt` <= Rayleigh length / 8 |
| hundreds of plasma rings per cell "to be safe" | the cost is linear in `<species>.ppc`, the gain is small once the cells are fine enough | 4-16 with converged cells is the normal range for `modes = 0` |
| the laser wake behind the bubble closure, or of a strongly depleted pulse, differs from a full PIC code | envelope model: validated in front of the closure (README section 7) | for the region behind the closure, strong red-shift or depletion use GEM-PIC-RZ or a full PIC code |

## Pitfalls of running

| what happens | why | what to do |
|---|---|---|
| a run with many threads is slower than with few | threads are not pinned, or pinned onto each other | containers, WSL, shared nodes: `OMP_PROC_BIND=false`. Dedicated nodes: `OMP_PROC_BIND=close OMP_PLACES=cores`, with Slurm `--cpus-per-task` and `--cpu-bind=cores` |
| all MPI ranks sit on one core | the Slurm allocation has one task | request `--ntasks` >= the number of ranks |
| many MPI ranks give little for a short run | the ranks form a pipeline over the time steps that needs P - 1 steps to fill: with N steps the efficiency is about N/(N + P - 1) | keep P below about N/4; a single solve (`time.steps = 0`) gains nothing from MPI |
| the disk is full after a few output steps | a field file is slices x radial nodes x components x 8 bytes; a fine uniform grid in a long box reaches gigabytes per step | `output.fields`, `output.rmax`, `output.xi_stride`; `output.field_files = 0` for scans; a light diagnostic group (`diag.names`) written often and full maps rarely |
| after a run died on a full disk, MPI runs hang when they open a file | the dead run left semaphore files of the MPI-IO layer in `/dev/shm` (Open MPI: `sem.OMPIO_*`) | free the disk, delete your own leftover files there |
| crash right after the start-up summary on a cluster | built with the old system compiler under a newer runtime library | configure a fresh build directory with `CXX=$(which g++) CC=$(which gcc)`; GCC >= 9 (skill quarz-setup) |
