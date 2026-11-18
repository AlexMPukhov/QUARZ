#!/usr/bin/env python3
"""Numbers for a QUARZ input: plasma units, beam and laser parameters in code units (python3, no packages).

QUARZ works in plasma units of the reference density n0: lengths in c/omega_p, times in 1/omega_p, densities in
n0, momenta in m c, fields in E0 = m c omega_p / e.

  python3 quarz_numbers.py --density 7e14 --length-m 10 \
      --particles 4e11 --species proton --energy-gev 400 --sigma-r-um 200 --sigma-z-um 5e4 --emit-n-um 3.5
  python3 quarz_numbers.py --density 1e18 --length-m 0.01 --wavelength-um 0.8 --a0 4 --w0-um 21 --tau-fs 30

Prints the conversions and starting values for the input keys named in the right column.
"""
import argparse
import math

p = argparse.ArgumentParser()
p.add_argument("--density", type=float, required=True, help="reference plasma density n0 [cm^-3]")
p.add_argument("--length-m", type=float, help="propagation length [m]")
p.add_argument("--particles", type=float, help="number of beam particles")
p.add_argument("--charge-pc", type=float, help="beam charge [pC] (instead of --particles)")
p.add_argument("--species", default="electron", choices=["electron", "positron", "proton", "muon"])
p.add_argument("--energy-gev", type=float, help="beam energy [GeV] (total energy; protons: momentum * c is the same "
                                                "to 1e-5 at 400 GeV)")
p.add_argument("--sigma-r-um", type=float, help="rms beam size per transverse coordinate [um]")
p.add_argument("--sigma-z-um", type=float, help="rms beam length [um]")
p.add_argument("--emit-n-um", type=float, help="normalised rms emittance [um] (= mm mrad)")
p.add_argument("--wavelength-um", type=float, help="laser wavelength [um]")
p.add_argument("--a0", type=float)
p.add_argument("--w0-um", type=float, help="laser spot w0 [um] (field 1/e radius)")
p.add_argument("--tau-fs", type=float, help="laser duration, intensity FWHM [fs]")
a = p.parse_args()

c = 2.99792458e8
n0 = a.density
kpinv = 5.3141e5 / math.sqrt(n0) * 1e4            # c/omega_p [um]
E0 = 96.16 * math.sqrt(n0)                        # V/m
tp = kpinv * 1e-6 / c                             # 1/omega_p [s]
print(f"n0 = {n0:.4g} cm^-3")
print(f"  c/omega_p   = {kpinv:.5g} um      (unit of length)")
print(f"  1/omega_p   = {tp * 1e15:.5g} fs      (unit of time; a time step dt moves the beam by dt c/omega_p)")
print(f"  lambda_p    = {2 * math.pi * kpinv:.5g} um = {2 * math.pi:.4f} c/omega_p")
print(f"  E0          = {E0 / 1e9:.5g} GV/m     (unit of the fields)")
print(f"  beam weight 1 = {n0 * (kpinv * 1e-4) ** 3:.5g} particles   (beams.txt, column 'charge')")
print(f"  input: units.n0_cm3 = {n0:g}        (only needed for lasers, ionization, scattering, SI openPMD output)")

mass = {"electron": 1.0, "positron": 1.0, "proton": 1836.15, "muon": 206.768}[a.species]
q = -1 if a.species in ("electron", "muon") else 1
gamma = None
if a.energy_gev:
    gamma = a.energy_gev * 1e9 / (0.51099895e6 * mass)
if a.particles or a.charge_pc:
    N = a.particles if a.particles else a.charge_pc * 1e-12 / 1.602176634e-19
    print(f"\nbeam: {N:.4g} {a.species}s = {N * 1.602176634e-19 * 1e9:.4g} nC")
    print(f"  input: <beam>.charge = {q}    <beam>.mass = {mass:g}")
    if gamma:
        print(f"  input: <beam>.gamma = {gamma:.5g}")
    if a.sigma_r_um and a.sigma_z_um:
        sr, sz = a.sigma_r_um / kpinv, a.sigma_z_um / kpinv
        nb = N / ((2 * math.pi) ** 1.5 * (a.sigma_r_um * 1e-4) ** 2 * (a.sigma_z_um * 1e-4))
        print(f"  input: <beam>.sigma_r = {sr:.5g}    <beam>.sigma_xi = {sz:.5g}        (c/omega_p)")
        print(f"  peak density of the Gaussian bunch = {nb:.4g} cm^-3")
        print(f"  input: <beam>.n0 = {nb / n0:.5g}       (peak density / n0 of the UNCUT bunch)")
        print(f"  k_p sigma_r = {sr:.3g}, k_p sigma_z = {sz:.3g}; n_b/n0 {'<< 1: linear wake' if nb / n0 < 0.1 else '> 0.1: nonlinear wake' if nb / n0 < 1 else '> 1: blowout'}")
        print(f"  radial grid: cells <= sigma_r/10 = {sr / 10:.3g} over the beam;  xi.step <= {min(0.05, sz / 10):.3g}")
    if a.emit_n_um:
        print(f"  input: <beam>.emit_n = {a.emit_n_um / kpinv:.5g}       (normalised emittance in c/omega_p)")
    if gamma:
        wb = math.sqrt(1.0 / (2 * gamma * mass))          # betatron frequency in an ion channel of density n0
        T = 2 * math.pi / wb
        print(f"  betatron period in a full ion channel = {T:.4g} /omega_p = {T * kpinv * 1e-6:.4g} m")
        print(f"  input: time.dt <= {T / 20:.4g}  (20 steps per period; halve it for the convergence check;")
        print(f"         or time.adaptive = 1 for beams whose energy changes much)")

if a.wavelength_um:
    k0 = kpinv / (a.wavelength_um / (2 * math.pi))        # omega0/omega_p
    print(f"\nlaser (envelope model): lambda0 = {a.wavelength_um} um, omega0/omega_p = {k0:.5g}, n0/n_c = {1 / k0 ** 2:.4g}")
    print(f"  input: laser.lambda0_um = {a.wavelength_um}   units.n0_cm3 = {n0:g}" + (f"   laser.a0 = {a.a0}" if a.a0 else ""))
    if a.w0_um:
        w0 = a.w0_um / kpinv
        zr = k0 * w0 ** 2 / 2
        print(f"  input: laser.w0 = {w0:.5g}        (c/omega_p; Rayleigh length {zr:.4g} = {zr * kpinv * 1e-3:.4g} mm)")
        print(f"  input: time.dt <= {zr / 8:.4g}      (Rayleigh length / 8)")
        if a.a0:
            ppc = (a.a0 * w0) ** 2 / 32                 # P/P_c = (k_p w0 a0)^2 / 32
            print(f"  P/P_c = {ppc:.3g}; matched spot of the blowout k_p w0 = 2 sqrt(a0) = {2 * math.sqrt(a.a0):.3g}")
    if a.tau_fs:
        L0 = a.tau_fs * 1e-15 * c * 1e6 / 1.177 / kpinv
        print(f"  input: laser.L0 = {L0:.5g}        (c/omega_p; intensity FWHM = 1.177 L0)")
        print(f"  input: laser.xi0 >= xi.min + {3 * L0:.4g}   (3 L0 behind the head of the box);  xi.step <= {L0 / 20:.3g}")

if a.length_m:
    L = a.length_m * 1e6 / kpinv
    print(f"\npropagation: {a.length_m} m = {L:.6g} /omega_p")
    print(f"  input: time.t_end = {L:.6g}    (or time.steps = t_end / time.dt)")
