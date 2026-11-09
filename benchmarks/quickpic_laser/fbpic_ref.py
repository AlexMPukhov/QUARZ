"""Independent full-PIC reference (FBPIC 0.27, quasi-cylindrical, modes 0 and 1) for the laser wakefield
benchmarks of Meng et al., PPCF (2026), doi 10.1088/1361-6587/aebd1b, with the parameters of case1.in /
case2.in (normalised to k_p; the wavelength 0.8 um only sets the SI scale).

The laser starts in vacuum with its centre 4 L0 before the sharp plasma edge at z = 0 and its focus at the edge;
s = z_centre,0 + c t is the distance the pulse reference point has travelled into the plasma (s = 0: centre at
the edge), xi = s - z (both in c/omega_p), so xi = 0 is the initial pulse centre as in the QUARZ decks.

At each s of the list it writes snap_<case>_s<s>.npz: xi (descending z), r, E_z on the axis and E_z(xi, r) of
mode 0 (in m c omega_p / e), the electron density n_e(xi, r) / n0 (mode 0), E_x on the axis and the laser
envelope |a|(xi) on the axis (Hilbert transform of E_x / E_L, E_L = m c omega_0 / e), and the electromagnetic
energy of mode 1 (the laser, J).  laser_<case>.txt: s, energy of mode 1, max |a| on the axis.

usage: python3 fbpic_ref.py 1|2 [--res 2] [--intensity-fwhm] [--cpu]
"""
import os
import sys
import time
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.signal import hilbert
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser

case = int(sys.argv[1])
res = float(sys.argv[sys.argv.index("--res") + 1]) if "--res" in sys.argv else 1.0   # refinement factor
use_cuda = "--cpu" not in sys.argv
# omega_p tau_FWHM = 2.26 is the FWHM of the field envelope (fits the OSIRIS lineouts); --intensity-fwhm: of the intensity
FW = 1.17741 if "--intensity-fwhm" in sys.argv else 1.66511
if case == 1:
    k0n, a0, w0n, L0n, s_list = 34.0, 2.5, 3.162, 2.26 / FW, [50, 100, 200, 300, 400, 500]
else:
    k0n, a0, w0n, L0n, s_list = 20.0, 4.0, 4.0, 2.26 / FW, [20, 50, 100, 150, 200]

lam = 0.8e-6
w_0 = 2 * np.pi * c / lam
wp = w_0 / k0n
kp = wp / c
n0 = epsilon_0 * m_e * wp ** 2 / e ** 2
E0 = m_e * c * wp / e                      # m c omega_p / e
EL = m_e * c * w_0 / e                     # m c omega_0 / e
z0 = -4 * L0n / kp                         # initial pulse centre (vacuum), plasma from z = 0
zmax, zmin = z0 + 5.0 / kp, z0 - 13.0 / kp  # window: xi in [-5, 13]
dz = lam / (24 * res)
Nz = int(round((zmax - zmin) / dz))
rmax = 15.0 / kp
Nr = int(round(150 * res))
dt = (zmax - zmin) / Nz / c

sim = Simulation(Nz, zmax, Nr, rmax, 2, dt, zmin=zmin, n_order=-1, use_cuda=use_cuda,
                 boundaries={"z": "open", "r": "open"}, verbose_level=1)
elec = sim.add_new_species(q=-e, m=m_e, n=n0, dens_func=lambda z, r: np.where(z > 0, 1.0, 0.0),
                           p_nz=2, p_nr=2, p_nt=4, p_zmin=0.0, p_rmax=rmax)
add_laser_pulse(sim, GaussianLaser(a0, w0n / kp, L0n / kp / c, z0=z0, zf=0.0, lambda0=lam))   # tau in s
sim.set_moving_window(v=c)
print("case %d: n0 = %.4e m^-3, Nz = %d, Nr = %d, dt = %.3e s" % (case, n0, Nz, Nr, dt), flush=True)


def host(a):
    return a.copy_to_host() if hasattr(a, "copy_to_host") else np.array(a)


def snapshot(s_now):
    fld = sim.fld
    # FBPIC copies the fields to the host at the end of step(); rho = the last deposited rho_next
    f0, f1 = fld.interp[0], fld.interp[1]
    z = np.asarray(f0.z)
    r = np.asarray(f0.r)
    Ez0 = np.array(f0.Ez).real / E0
    ne = -np.array(f0.rho).real / (e * n0)
    Ex_ax = 2.0 * np.array(f1.Er)[:, 0].real / EL          # x polarisation, theta = 0, first cell
    U = 0.0
    vol = 2 * np.pi * r[None, :] * f1.dr * f1.dz
    for comp in ("Er", "Et", "Ez"):
        U += 0.5 * epsilon_0 * 2 * (np.abs(np.array(getattr(f1, comp))) ** 2 * vol).sum()
    for comp in ("Br", "Bt", "Bz"):
        U += 0.5 / (4e-7 * np.pi) * 2 * (np.abs(np.array(getattr(f1, comp))) ** 2 * vol).sum()
    xi = s_now - z * kp
    env = np.abs(hilbert(Ex_ax))
    np.savez("snap_%d_s%03d.npz" % (case, int(round(s_now))), xi=xi, r=r * kp, Ez=Ez0, ne=ne, Ez_axis=Ez0[:, 0],
             Ex_axis=Ex_ax, a_axis=env, U1=U, s=s_now, k0=k0n)
    return U, env.max()


t0 = time.time()
it = 0
with open("laser_%d.txt" % case, "w") as fo:
    fo.write("# s [c/omega_p], EM energy of mode 1 [J], max |a| on the axis\n")
    for s_out in [-4 * L0n + 0.1] + s_list:   # first: the pulse still in vacuum (reference energy)
        n_target = int(round((s_out / kp - z0) / (c * dt)))
        sim.step(n_target - it, show_progress=False)
        it = n_target
        s_now = z0 * kp + c * dt * it * kp
        U, amax = snapshot(s_now)
        fo.write("%.4f %.8e %.5f\n" % (s_now, U, amax))
        fo.flush()
        print("s = %.2f  U1 = %.5e J  max|a| = %.3f  wall %.0f s" % (s_now, U, amax, time.time() - t0), flush=True)
print("done, wall time %.0f s" % (time.time() - t0))
