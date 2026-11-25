# QUARZ: reading a run

## Console

```
Radial grid: N = 308 cells, R = 10, h_min = 0.0100298 (at axis h_0 = 0.0100298), h_max = 0.100298, ...
Box: xi in [-1000, 2] (head at 2), dxi = 0.05, 20041 slices;  beam shape in xi: linear
Plasma species 'electrons': q = -1, m = 1, mobile, up to 1232 macro-particles per slice ...
Beam 'protons': 4000000 macro-particles
Output: fields ez er bth ne rhob every 25, beams protons every 250, r <= 3.01525 (237 nodes), every 5. slice, native
step      0  t =          0   sweep 5.066 s
```

- Read the summary: is the grid what you meant (axis cell, number of cells), the box, the species, the beams?
- `sweep ... s` is the time of one step. Trapped or removed plasma particles and extra sub-slices are reported
  on the step lines when they occur.
- `WARNING: input key '...' was not used` at the end: the key was never read (misspelt, or not valid in this
  setup). `ERROR (rank 0): ...` stops the run.

## Files in `output.dir`

| file | content |
|---|---|
| `grid.txt` | the radial grid: j, r_j, h_j, V_j |
| `beams.txt` | one line per beam per step: name, step, t, alive, npart, charge, gamma_mean, gamma_rms, xi_mean, xi_rms, r_rms, emit_nx, x_mean, y_mean, emit_ny, sigma_x, sigma_y |
| `axis_NNNNNN.txt` | on the axis, one line per slice (increasing xi: the head of the box last): xi, E_z, psi, n_e, rho_beam, focusing slope (E_r - B_theta)/r at the first node |
| `fields_NNNNNN.bin` | 2D maps (xi, r) of psi, ez, er, eth, br, bth, bz, ne, ni, rhob (binary; `tools/quarz_read.py` of the repository reads it, `tools/plot_fields.py` plots it) |
| `beam_<name>_NNNNNN.bin` | beam particles: x, y, p_x, p_y, p_z, xi, w |
| `slices_<beam>_NNNNNN.txt` | per-slice beam moments (`output.beam_slices`), for hosing and micro-bunch studies |
| `laser.txt` | laser runs: step, t, max \|a\|, w_rms, mean xi, L_rms, energy integral |
| `openpmd/data_NNNNNN.h5` | with `output.format = openpmd` or `both` |

The exact formats are in `output_reference.md`.

## Units

| quantity | unit | at n0 [cm^-3] |
|---|---|---|
| length, xi, r | c/omega_p | 5.314e5 / sqrt(n0) cm |
| time t | 1/omega_p | the box moves by t c/omega_p: z = t c/omega_p |
| fields | E0 = m_e c omega_p / e | 96.16 sqrt(n0) V/m |
| density | n0 | |
| beam `charge` in `beams.txt` | n0 (c/omega_p)^3 particles | `scripts/quarz_numbers.py` prints the number |
| emittance | c/omega_p (normalised) | |

## Quick look

```
python3 scripts/quick_look.py out --density 7e14
```

prints, per output step, the largest on-axis E_z (in E0 and GV/m), where it is, and the largest on-axis electron
density; and the first and last line of every beam. With matplotlib it writes `out/quicklook.png`.

## What to look at

- **Wake amplitude:** linear regime: of the order n_b/n0 E0 or below. Blowout: the focusing slope
  (E_r - B_theta)/r is 0.5 inside the bubble (last column of the axis file), E_z falls linearly through it.
- **Beam charge:** the column `charge` of `beams.txt` should stay constant unless particles are expected to leave
  (defocused protons of a self-modulating bunch). A sudden loss: particles reach the wall or leave the box
  through its tail.
- **Energy:** gamma_mean of a witness against t; gamma_rms for the spread.
- **Emittance:** emit_nx of a matched witness should stay constant; growth at the first steps means a mismatch,
  too coarse an axis cell or too large a time step.
- **Self-modulation:** max |E_z| against z grows, saturates and decays; rho_beam on the axis shows the
  micro-bunches with the period of the plasma wave.
- **Noise:** compare two runs with different `<beam>.seed`; what differs is noise.
