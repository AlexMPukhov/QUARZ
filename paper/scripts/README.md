# Paper scripts

Scripts that produced the figures of `paper/tex/quarz_paper.tex` (run from a scratch directory
containing the run outputs `out_<case>/`, `log_<case>.txt`):

- `evo.sh` — the four evolution runs (Fig. evolution); `QUARZ=/path/to/quarz` selects the executable.
- `errs.py ref` — errors of the convergence series against the reference run (Fig. convergence,
  Table); writes `errs_ref.json`. `errs_s0.03125.json` is the result used in the paper.
- `fig_conv.py`, `fig_maps.py`, `fig_evo.py` — the figures, written to `paper/tex/figs/`.
- `closure.sh`, `fig_closure.py` — runs and figure of the bubble-closure subsection (Fig. closure):
  the pinched-witness deck with the box extended to xi = 12, cold vs `plasma.smooth_length = 0.005`
  (~3 min on 2 cores).
- `rf.py` — helper.

The input decks of all paper runs are in `paper/inputs/` (see its README).

**Convention:** the decks and scripts here use the ξ = z − t convention of this version (head of the
box at `xi.max` = 0, the box extends to negative ξ). The paper (QUARZ v1.0.0) uses ξ = t − z, so its
ξ values are the negatives of the ones here, and figures made with these scripts are mirrored in ξ
relative to the published ones (head on the right). To reproduce the published decks and figures
exactly, use QUARZ v1.0.0.
