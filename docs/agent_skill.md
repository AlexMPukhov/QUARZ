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

- **Clustering of plasma macro-particles in long boxes.** The AWAKE-like example (4e11 protons, sigma_z = 5 cm,
  cut one sigma ahead of the centre, 7e14 cm^-3) was first set up with a box of 1000 c/omega_p and 4 rings per
  cell. At t = 0 the on-axis wake is 0.005 E0 behind the cut, as linear theory gives, but at xi = 300 it jumps
  to 0.1 E0 with electron density spikes on the axis. The place of the jump moves with the numerics (4 rings per
  cell: 300; 16: 500; 64: 600; finer radial cells: 400; more beam particles: no change), so it is the clustering
  instability of quasi-static plasma macro-particles (Kargapolov et al., Phys. Plasmas 32, 023905 (2025)), not
  physics. The example therefore ends at xi = 500 with 16 rings per cell, the skill has a rule for long boxes,
  and `quick_look.py` prints the wake of the first step along the box. The `plasma.declustering` option in the
  source is not a general remedy and is not documented in the README, so the skill does not name it: the
  remedy is more rings per cell.
- The full 10 m run of the example (17 min on 32 threads): the largest on-axis E_z grows from 0.005 E0 to
  0.77 E0 at 3 m and falls to 0.17 E0 at 10 m; 73 % of the cut bunch is lost (19 % behind the box, the rest
  defocused to the wall at 2 mm). Not compared with another code.

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
- **Limits of the trial:** the shipped example `awake_density_gradient.in` is this very request, so the trial
  tested the dialogue, the check and the reading of the output, not the construction of an input from a different
  problem. It was a sub-agent of the session that wrote the skill, not an independent agent, and
  `npx skills add` itself was not run.
