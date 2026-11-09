# Laser wakefield benchmarks of the QuickPIC laser module (Meng et al. 2026)

W. Meng et al., "Improvement of the laser module in the quasi-static PIC code QuickPIC", Plasma Phys. Control.
Fusion (2026), doi:10.1088/1361-6587/aebd1b, benchmark a new envelope solver of QuickPIC (Crank–Nicolson in s,
BDF2 in ξ, a 3D Cartesian multigrid solve per slice) against OSIRIS. QuickPIC solves the same envelope equation as
QUARZ, 2∂ₛ(∂_ξ − ik₀)â − ∇⊥²â = χâ. This directory runs their two LWFA cases (Sec. 4.1) in QUARZ. FBPIC
0.27 (full PIC, quasi-cylindrical, modes 0 and 1) is the independent reference, with the same parameters.
GEM-PIC-RZ runs the same cases as a third code (decks in that repository, `benchmarks/quickpic_laser`).

| file | content |
|---|---|
| `case1.in` | \|â\| = 2.5, k_p w0 = 3.162, k₀/k_p = 34, ω_p τ_FWHM = 2.26, uniform plasma, Δr = 0.1, Δξ = 0.03, Δs = 5 (their grid), s = 0 … 500 |
| `case2.in` | \|â\| = 4, k_p w0 = 4, k₀/k_p = 20, same pulse length and grid, s = 0 … 200 |
| `fbpic_ref.py` | FBPIC reference: the laser starts in vacuum and focuses at a sharp plasma edge; it writes E_z, n_e, the laser envelope on the axis and the energy of mode 1 at the same s |
| `run.sbatch` | Slurm runner (16 OpenMP threads; timings to `timings.txt`) |
| `compare.py` | on-axis E_z of all runs vs FBPIC and, if present, vs curves digitized from the paper's figures (not included here); laser diagnostics |
| `plot_maps.py` | \|â\| and n_e maps at s = 20 … 200 for the three ξ schemes (cf. their Fig. 4), and a measure of radial noise in \|â\|² |

Pulse length: ω_p τ_FWHM = 2.26 is the FWHM of the **field** envelope \|â\| (L0 = 2.26/(2√ln 2) = 1.357).
With the intensity FWHM (L0 = 1.920), QUARZ differs from their OSIRIS lineouts by 10 % (s = 50) and 25 % (s = 500).
With the field FWHM the difference drops to 4.9 % and 4.5 %. The stronger case does not state τ, so the same
value is assumed there.

## Results

Error = rms(E_z − E_z,ref)/max|E_z,ref| of the on-axis field, over ξ ∈ [−4, 11] (case 1) or ξ ∈ [−4, 7] (case 2,
in front of the bubble closure: OSIRIS self-injects there). The "front" column (case 1) is ξ ∈ [−4, 5.5], before the
closure.

| case, s | reference | QUARZ (trapezoid) | QUARZ BDF2 | QUARZ Δ/2 | GEM-PIC-RZ |
|---|---|---|---|---|---|
| 1, s = 50 | FBPIC | 5.5 % (front 0.7 %) | 5.5 % | 5.4 % | 10.3 % (front 0.4 %) |
| 1, s = 500 | FBPIC | (see below) | | | |
| 2, s = 20 | FBPIC | 1.0 % | 1.0 % | 0.9 % | 1.4 % |
| 2, s = 50 | FBPIC | 1.1 % | 1.1 % | 1.1 % | 1.4 % |
| 2, s = 100 | FBPIC | 1.3 % | 1.3 % | 1.3 % | 1.1 % |
| 2, s = 150 | FBPIC | 1.5 % | 1.5 % | 1.5 % | 1.5 % |
| 2, s = 200 | FBPIC | 1.6 % | 1.6 % | 1.5 % | 1.6 % |

Δ/2: Δr, Δξ and Δs halved and twice the particles per cell. In case 1, all codes differ most at the bubble closure
(ξ ≈ 6.3), where the jump of E_z moves by ~0.1 between the codes. The two full-PIC codes differ there as much
as QUARZ does: digitized OSIRIS vs FBPIC, 10 % at s = 50.

Against the digitized OSIRIS curves (same measure; their QuickPIC-FD curve, also digitized, in brackets): case 1
4.9 % (2.2 %) at s = 50 and 4.5 % (3.2 %) at s = 500; case 2 at s = 200, 3.4 % (1.5 %). The paper quotes 3.2 % and
4.3 % for QuickPIC-FD in case 1, probably with another normalization.

**Laser** (case 2). QUARZ: max|â| and max|E_L|/k₀ = max|(∂_ξ − ik₀)â|/k₀. FBPIC and GEM-PIC-RZ: max of the
envelope of E_x/E_L on the axis for |ξ| < 3 (the laser; GEM-PIC-RZ has larger fields at the closure at s = 200).
Energy relative to s = 0:

| s | QUARZ max\|â\| | QUARZ max\|E_L\|/k₀ | FBPIC max\|E_x\|/E_L | GEM-PIC-RZ max\|E_x\|/E_L | QUARZ U/U0 | FBPIC U/U0 |
|---|---|---|---|---|---|---|
| 20 | 4.07 | 4.03 | 4.02 | 4.03 | 0.993 | 0.993 |
| 50 | 4.47 | 4.33 | 4.34 | 4.36 | 0.982 | 0.983 |
| 100 | 5.43 | 5.12 | 5.07 | 5.13 | 0.963 | 0.966 |
| 150 | 5.71 | 5.28 | 5.20 | 5.26 | 0.942 | 0.946 |
| 200 | 5.54 | 4.98 | 4.89 | 5.04 | 0.920 | 0.925 |

The laser self-focuses (max|â| 4 → 5.7) and loses 8 % of its energy by s = 200; QUARZ's depletion agrees with FBPIC
to 0.5 % of the energy. In case 1 the pulse focuses to |â| = 2.73 (s = 100) and then spreads (1.32 at s = 500), with
3.7 % energy loss.

**Stability.** All three ξ schemes of QUARZ (`laser.xi_scheme = trapezoid | bdf2 | euler`) are stable in both cases.
The maps (`plot_maps.py`) show no near-axis tails or noise growth up to s = 200 at |â| = 4 (their FFT solver breaks
down before s = 150), and the radial noise measure stays at 0.03.
BDF2 and the trapezoid agree to 10⁻³ in max|â|.

**Cost** (one thread of an AMD EPYC 7543): case 1, 12.8 s = 0.0036 core-hours; case 2, 5.2 s = 0.0015 core-hours.
For case 1 the paper gives 42.7 core-hours (QuickPIC-FD), 13.9 (QuickPIC-FFT) and ~15000 (OSIRIS). This is not like
for like: QUARZ is axisymmetric (m = 0 envelope, 150 radial cells instead of a 300 × 300 transverse grid). GEM-PIC-RZ,
which resolves the laser period and keeps the slow-time derivatives, needs 4.2 (case 1) and 2.0 (case 2) core-hours on
16 threads. FBPIC took 6.7 min (case 2) on one RTX 5090 GPU.
