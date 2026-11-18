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
| the wake has the wrong sign or the beam sits at the wrong end | xi = ct - z: the head is at small xi. A witness goes behind the driver, i.e. at larger xi |
| the wake jumps far behind the head, at a place that moves with `<species>.ppc` or the grid | clustering of the plasma macro-particles in a long box (rule 9 of SKILL.md): more `<species>.ppc`, a shorter box; say where the box stops being reliable |
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
