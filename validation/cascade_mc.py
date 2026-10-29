"""Independent event-by-event Monte Carlo of an electron cascade in a gas (reference for the 0D cascade check of
electron-impact ionization + excitation, section 24).

The same cross-sections as the model (written here from the papers, not from the code): ionization of the neutral
atom by Lotz (Z. Phys. 206, 205 (1967), Table 1), the ejected electron from dsigma/dW ~ 1/(J^2 + W^2) on
[0, (E - Ip)/2] (Opal, Peterson & Beaty 1971), the projectile loses Ip + W; excitation of the 13 Xe levels of
Puech & Mizzi as tabulated by Mukundan & Bhardwaj (arXiv:1604.08449, Table 3), the projectile loses W_j.  Unlike
the code (continuous slowing down between ionizations), every collision is a discrete event here; elastic
scattering does not change the energy and is left out.  Electrons below the lowest threshold stop.

usage:
  cascade_mc.py W   E0_eV [N] [noexc]       mean energy per ion pair of N primaries of energy E0
  cascade_mc.py xs                          print sigma_ion, sigma_exc and the stopping split at a few energies
"""
import sys

import numpy as np

ME = 510998.95
IP, J = 12.1298, 8.7
LOTZ = ([IP, 23.4, 67.0], [6, 2, 10], 4.0, 0.54, 0.64)
EXC = [(8.315, 53.7, 2, False), (8.437, 0.26, 0, True), (9.447, 27.0, 2, False), (9.570, 0.19, 0, True),
       (9.580, 2.57, 1, False), (9.706, 4.89, 1, False), (9.809, 2.54, 1, False), (9.917, 0.01, 0, True),
       (10.11, 20.0, 1, False), (10.40, 0.395, 0, True), (10.59, 0.097, 0, True), (11.0, 5.0, 1, False),
       (11.7, 0.689, 0, True)]
A0_8PI = 8 * np.pi * (0.529177210903e-8) ** 2
RY = 13.605693


def sig_ion(E):
    P, q, a, b, c = LOTZ
    if E <= IP: return 0.0
    s = sum(qi * np.log(E / Pi) / (E * Pi) for Pi, qi in zip(P, q) if E > Pi)
    return 1e-14 * a * (1 - b * np.exp(-c * (E / IP - 1))) * s


def sig_exc(E):
    """cross-sections of the excitation levels (cm^2)"""
    g = 1 + E / ME
    b2 = 1 - 1 / g ** 2
    x = ME * b2
    out = []
    for W, F, c, allowed in EXC:
        if E <= W: out.append(0.0); continue
        B = (1 - 2 * W / x) / x ** c
        if allowed: s = A0_8PI * RY ** 2 * F / (x * W) * (np.log(x / (1 - b2) / (2 * W)) - b2) * B
        else: s = A0_8PI * RY * F / x * B
        out.append(max(s, 0.0))
    return np.array(out)


WTH = np.array([e[0] for e in EXC])


def cascade(E0, rng, exc=True):
    """number of ionizations of one primary of energy E0 and of all its descendants"""
    stack, n = [E0], 0
    while stack:
        E = stack.pop()
        while E > WTH.min():
            si = sig_ion(E)
            sx = sig_exc(E) if exc else np.zeros(len(EXC))
            tot = si + sx.sum()
            if tot <= 0: break
            u = rng.random() * tot
            if u < si:
                wm = 0.5 * (E - IP)
                W = J * np.tan(rng.random() * np.arctan(wm / J))
                E -= IP + W
                stack.append(W)
                n += 1
            else:
                j = np.searchsorted(np.cumsum(sx), u - si, side="right")
                E -= WTH[min(j, len(EXC) - 1)]
    return n


if __name__ == "__main__":
    if sys.argv[1] == "xs":
        for E in (10, 15, 20, 30, 50, 100, 300, 1000):
            sx = sig_exc(E)
            si = sig_ion(E)
            li = si * (IP + (J * np.log1p((max(0.5 * (E - IP), 1e-30) / J) ** 2) / (2 * np.arctan(max(0.5 * (E - IP), 1e-30) / J)) if E > IP else 0))
            lx = (sx * WTH).sum()
            print(f"  E = {E:5.0f} eV: sigma_ion {si:.3e}  sigma_exc {sx.sum():.3e} cm^2;  energy loss: excitation share {lx / (li + lx):.2f}")
    else:
        E0 = float(sys.argv[2])
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        exc = not (len(sys.argv) > 4 and sys.argv[4] == "noexc")
        rng = np.random.default_rng(12345)
        k = np.array([cascade(E0, rng, exc) for _ in range(N)])
        W = E0 / k.mean()
        print(f"  independent MC, E0 = {E0:g} eV, {N} primaries{'' if exc else ', no excitation'}: {k.mean():.3f} ions per "
              f"primary, W = {W:.2f} +- {W * k.std() / k.mean() / np.sqrt(N):.2f} eV")
