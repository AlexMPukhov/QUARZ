# Agent skills for QUARZ

QUARZ ships two skills in the open [Agent Skills](https://agentskills.io/specification) format: a directory with a
`SKILL.md` (YAML frontmatter `name`, `description`; instructions in Markdown) and optional `references/`,
`scripts/`, `examples/`. Agents that read the format (Claude Code, Codex, OpenCode, Cursor, Gemini CLI, Cline,
Goose and others) load the short description at start-up and the rest when a request fits.

| skill | content |
|---|---|
| `skills/quarz-setup` | clone, build (Kokkos, MPI, GPU), tests, how to run, where the output goes |
| `skills/quarz-simulation` | which code for which problem; dialogue with the user; the input in one page; rules for grid, box, time step, noise; `references/` (input and output reference, physics rules, diagnostics, debugging); `examples/` (five inputs); `scripts/` (`quarz_numbers.py`, `check_input.sh`, `quick_look.py`) |

**Purpose.** The skills help the user run the code for a physics problem: the agent asks for what is missing,
states its assumptions, writes the input, checks it, makes a short run and reads the first output with the user.
Both skills tell the agent not to modify the QUARZ source: if the code cannot do something, the agent says so,
names the limit and the code that fits, and points to the issue tracker.

## Install

```
npx skills add AlexMPukhov/QUARZ                          # all agents the tool finds
npx skills add AlexMPukhov/QUARZ --skill quarz-simulation -a claude-code
```

(`skills` CLI of github.com/vercel-labs/skills; it looks for `skills/*/SKILL.md` in the repository.) Without
Node.js: copy the two folders into the agent's skills directory, e.g. `~/.claude/skills/` (Claude Code) or
`.agents/skills/` in a project (Codex, OpenCode, Cursor, Gemini CLI).

## Checking an input without a check switch

QUARZ has no check-only mode, and unknown keys only give a warning at the end of a run. The skill therefore
ships `scripts/check_input.sh <quarz> <input>`: one quasi-static solve (`time.steps=0`) into a scratch directory,
which fails if the code reports an error or a key that was never read. One solve costs seconds even for 20000
slices.

## Keeping the skills in step with the code

- `sh tools/sync_skills.sh` regenerates the parts that are copies: `references/input_reference.md` (README
  section 5), `references/output_reference.md` (section 6) and four inputs of `examples/`. Run it after editing
  those.
- `ctest -R agent_skill` (`tests/agent_skill.sh`):
  1. format of the specification (name = directory, description length, SKILL.md length, named files exist);
  2. the copies equal their sources;
  3. every input key named in the hand-written skill files (`grid.rmax`, `<beam>.sigma_r`, ...) occurs as a string
     literal in `src/` and is documented in README section 5;
  4. every ```` ```quarz ```` snippet passes the check, and the check rejects a misspelt key and a wrong value;
  5. every input shipped in a skill passes the check and runs 2 steps (finite numbers in the output);
  6. `quarz_numbers.py` against the beam charge the code reports; `quick_look.py` runs.
- A new input key that the skills should know: write it as `prefix.key` (or `<beam>.key`, `<species>.key`) in the
  skill files; the test then insists that the code and the README know it.

The same pair of skills, sync script and test exists in GEM-PIC-RZ; the tests differ only in a block
"code-specific settings".

## What the work on the skills found

- **Clustering of plasma macro-particles in long boxes, and the radial cell size.** The AWAKE-like example (4e11
  protons, sigma_z = 5 cm, cut one sigma ahead of the centre, 7e14 cm^-3) was first set up with a box of 1000
  c/omega_p, 4 rings per cell and the grid `2.0:0.01 10.0:0.1`. At t = 0 the on-axis wake is 0.005 E0 behind the
  cut, as linear theory gives, but at xi = 350 it jumps to 0.1 E0 with electron density spikes on the axis: the
  clustering instability of quasi-static plasma macro-particles (Kargapolov et al., Phys. Plasmas 32, 023905
  (2025)). First-step runs with other numerics (wake averaged over r < 1 in the last tenth of the box, seed level
  0.001 E0) showed what decides: the radial cell size inside the plasma column. Cells of 0.02-0.03: 0.001; 0.05:
  0.002; 0.1: 0.016-0.023, all with 4 rings per cell and nearly the same run time. More rings help slowly (16 per
  cell on the 0.1 grid: 0.0024 at twice the time; one ring per cell on a uniform 0.02 grid: 0.0013). The cell-scale
  density noise itself saturates at the same level on all these grids (about 0.25 with 4 rings, 0.1 with 16); what
  changes is the field it makes. Half the xi step, sub-slicing and a soft plasma edge change nothing. The on-axis
  values overstate the damage (a ring at the axis node), so `quick_look.py` also prints the wake at r = 0.5 and
  averaged over r < 1 when a field file is there. Rules 2 and 9 of the skill and the example were changed
  accordingly: the example now holds the whole bunch on `2.0:0.01 7.3:0.02 10.0:0.1` with 4 rings per cell.
  The `plasma.declustering` option in the source is not a general remedy and is not documented in the README, so
  the skill does not name it.
- The full 10 m run of the example (24 min on 32 threads) against runs with uniform cells of 0.02 (4 and 8 rings
  per cell) and 0.01: the r.m.s. of the on-axis E_z in the front half of the bunch (xi = 100-500), and in the
  whole bunch from 5 m on (after saturation), agree within 10 %, the phases within 0.3 rad, the remaining charge
  (31 % at 10 m) within 2 %. In the rear half during the growth (2-4 m) no two of these runs agree in the phase
  of the wake, also not 4 against 8 rings on the same grid: that part is sensitive to noise and is not converged.
  With the first grid (0.1 cells) the front half is the same; the rear quarter after saturation is 20-45 % lower
  and up to 1.3 rad off. Not compared with another code.
- **What seeds the rear half:** repeating the example with another `<beam>.seed` changes the wake of the first
  step behind xi = 300 (phase by up to 2.6 rad): there the wake of the cut has decayed to the level of the
  macro-particle noise of the beam (4e6 particles), which then seeds the self-modulation of the rear half. With
  16e6 particles the first-step wake there is 2-3 times smaller, and two seeds still disagree during the growth.
  The quiet loader (rule 8) with the physical shot noise does not remove this: in eight runs of the example that
  differ only in the beam seed, the rear half (xi 500-900) has no common phase at 2-3 m (phase spread up to
  0.8 rad on the axis, amplitude spread 11-19 %), while from 5 m on all runs are one wave (phase spread
  0.1-0.2 rad, amplitude 2-10 %, charge 0.2-0.5 %). A warm plasma (5 eV, four seeds) gives the same picture with
  a somewhat larger spread after saturation (phase 0.2-0.3 rad). So the rear half during the growth is one
  realization per run, whatever the loader; the saturated state is reproducible.
- **Note in the code:** since then QUARZ prints a note at the first step when a box longer than 300 has plasma in
  cells above 0.03 (ctest `long_box_note`). Of the 36 shipped inputs it appeared for `examples/awake_seeded_smi.in`
  (box of 302, cells of 0.1 inside the plasma) and for `validation/eimpact.in` (box of 1000, cells of 0.05; a
  test without fields, where it does not matter). The example now uses `grid.regions = 2.0:0.01 7.3:0.02
  10.0:0.1` as the rule asks. At this box length the grid hardly matters: over 10 m the on-axis wake of the two
  grids agrees to 3 % in amplitude and 0.05 rad in phase, whereas another beam seed changes the amplitude by up
  to 37 % at 1 m and about 20 % at 2-4 m (2e6 beam particles); the run takes 20 % longer.

## Trial with a fresh agent

One trial (2026-11-18): a fresh agent of a smaller model, with the skill files only (no README, no source), a
stand-in to check and run inputs on a compute node, and the request "AWAKE-like simulation with the proton bunch
length 5 cm, population 4e11 protons, cut at 1 sigma ahead; plasma 7e14 1/cc, positive density gradient 5 % over
10 meters".

- **First reply:** restates the request in numbers (c/omega_p, E0, 64 nC), lists its assumptions with defaults
  (400 GeV, sigma_r = 0.2 mm, 3.5 mm mrad, whether 4e11 is the uncut bunch, column radius, what is measured),
  names the quasi-static limits and the clustering limit of long boxes, and asks for corrections or "go ahead".
- **Input:** one check, passed; no unknown keys. 3-step run: 3.4 s per step on 16 threads, wake of the first
  step 0.005 E0 along the whole box without a jump.
- **Source:** not looked at, not changed (the agent had no access to it and reported no reason to want it).
- **What it missed in the skill, now added:** that beam particles outside a shortened box are dropped (19 % of
  the cut bunch here) and that the quick look reported this as a loss; a criterion for the on-axis density
  spikes (they come first; the jump of the wake counts).
- **After the trial** the example was changed (whole bunch, finer cells over the plasma, see above).
- **Second trial** (same day, with the new rules, a different problem so that the example cannot be copied):
  "seeded self-modulation of a proton bunch: 1e11 protons, 400 GeV, sigma_z = 12 cm, sigma_r = 150 micron, cut at
  its centre; uniform plasma 2e14 1/cc, 5 m; wake amplitude along the plasma and along the bunch". The agent wrote
  its assumptions first (emittance, energy spread, column radius 7, wall at 10, no mobile ions with the estimate
  that they are marginal), chose a box of 1000 c/omega_p with `grid.regions = 2.0:0.01 7.3:0.02 10.0:0.1`, 4 rings
  per cell and a time step of half the betatron limit; the input passed the check at the first attempt; a 3-step
  run gave 5.3 s per step on 16 threads; the first-step wake off the axis was flat along the whole bunch
  (0.0017-0.0019 E0 at r = 0.5) and it read the on-axis rise at the tail with a density spike of 51 n0 correctly
  as the axis-node artefact of rule 9. It did not look at the source. What it missed, now added: `output.every=1`
  for the 3-step run, the unit hint of `--sigma-z-um`. Still missing: `quarz_numbers.py` gives no box for a cut
  bunch and no default plasma column radius. No convergence run and no full run in the trial.
- **Limits of the trial:** the shipped example `awake_density_gradient.in` is this very request, so the trial
  tested the dialogue, the check and the reading of the output, not the construction of an input from a different
  problem. It was a sub-agent of the session that wrote the skill, not an independent agent, and
  `npx skills add` itself was not run.
