"""Beam shot noise (section 25): noise power of the macro-particle beam relative to the real protons.

For a radial kernel K and wavenumber k:  R(k) = |sum_i N_i K_i h_i e^{i k xi_i}|^2 / sum_i N_i K_i^2 h_i^2,
N_i = w_i nu real protons per macro-particle, h a Kaiser window (beta 20) over the flat part of the bunch.  Real (Poisson)
protons give <R> = 1 for every K and k; a quiet beam gives R << 1; macro-particle noise gives R ~ N_i >> 1.
R is averaged over k in [k1, k2] (wavelengths 31-310 c/omega_p = 1-10 mm, away from the bunch profile).

usage: shot_check.py dir label   (beam_p_000000.bin and the last beam_p_*.bin)
"""
import glob
import sys

import numpy as np

n0 = 2.965e16
kpi = 2.99792458e10 / (5.64146e4 * np.sqrt(n0))
nu = n0 * kpi ** 3
SR = 6.479
BETA = 20.0
KERNELS = [("K=1 (current)", lambda r: np.ones_like(r)), ("K=exp(-r^2/2sr^2)", lambda r: np.exp(-r ** 2 / (2 * SR ** 2))),
           ("K=[r<sr/2]", lambda r: (r < SR / 2).astype(float)), ("K=[sr<r<1.5sr]", lambda r: ((r > SR) & (r < 1.5 * SR)).astype(float))]


def load(fn):
    with open(fn, "rb") as f:
        n = np.frombuffer(f.read(4), np.int32)[0]
        a = np.frombuffer(f.read(), np.float64)
    return a.reshape(n, a.size // n)


def ratios(fn):
    p = load(fn)
    x, y, xi, w = p[:, 0], p[:, 1], p[:, 5], p[:, 6]
    r = np.hypot(x, y)
    sel = (xi > 300) & (xi < 4700)
    h = np.i0(BETA * np.sqrt(1 - ((xi[sel] - 2500) / 2200) ** 2)) / np.i0(BETA)   # Kaiser: leakage of the profile < 1e-15
    # deposit (cloud in cell) onto a grid of spacing D, FFT; the CIC transfer function sinc^2(k D / 2) is divided out
    D = 0.125
    g = (xi[sel] - 300) / D
    i0 = np.floor(g).astype(int); f = g - i0
    nf = 1 << int(np.ceil(np.log2(4 * (4400 / D + 2))))
    k = 2 * np.pi * np.fft.rfftfreq(nf, D)
    band = (k > 2 * np.pi / 310) & (k < 2 * np.pi / 31)
    T = np.sinc(k[band] * D / (2 * np.pi)) ** 2
    out = []
    for _, K in KERNELS:
        a = w[sel] * nu * K(r[sel]) * h
        G = np.bincount(i0, a * (1 - f), nf) + np.bincount(i0 + 1, a * f, nf)
        S = np.abs(np.fft.rfft(G)[band] / T) ** 2
        out.append(S.mean() / (w[sel] * nu * (K(r[sel]) * h) ** 2).sum())
    # independent k values in the band: spacing 2 pi / L_eff, L_eff = (int h)^2 / int h^2 of the window
    L_eff = 4400 * np.mean(h0 := np.i0(BETA * np.sqrt(1 - np.linspace(-1, 1, 4001) ** 2))) ** 2 / np.mean(h0 ** 2)
    out.append(1 / np.sqrt((1 / 31 - 1 / 310) * L_eff))
    return out, (w[sel] * nu).mean()


d, label = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{d}/beam_p_*.bin"))
(r0, Nm), (r1, _) = ratios(fs[0]), ratios(fs[-1])
err = r0.pop(); r1.pop()
print(f" {label}: {len(load(fs[0]))} macro-particles, {Nm:.2e} protons each (mean); noise power / Poisson, "
      f"<R> over k = 2pi/(1-10 mm), statistical error +-{err:.2f}:")
for (name, _), a, b in zip(KERNELS, r0, r1):
    print(f"   {name:18s} z = 0: {a:9.3g}   z = 1 m: {b:9.3g}")

# free drift of the whole bunch: r_rms(z) = r_rms(0) sqrt(1 + (z / beta*)^2), beta* = gamma sigma_r^2 / eps_n; emittance constant
b = np.array([l.split() for l in open(f"{d}/beams.txt") if l.startswith("p ")], dtype=object)
if len(b):
    z = b[:, 2].astype(float); rr = b[:, 10].astype(float); en = b[:, 11].astype(float)
    bs = 427.3 * SR ** 2 / (0.0110 * SR)
    print(f"   free drift over 1 m: r_rms growth {rr[-1] / rr[0] - 1:.5f} (analytic {np.sqrt(1 + (z[-1] / bs) ** 2) - 1:.5f}), "
          f"emittance {en[0]:.5f} -> {en[-1]:.5f}")
