# Cross-check of a single quasi-static laser wake against Wake-T (pip install wake-t), same laser,
# plasma and resolution.  usage: python3 laser_vs_waket.py a0 w0 L0  (in normalised units; uses q.in)
import sys, os, subprocess, numpy as np, scipy.constants as ct
os.environ['OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS'] = '0'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from quarz_read import read_fields
import openpmd_api as io
from wake_t import GaussianPulse
from wake_t.beamline_elements import PlasmaStage
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_size
a0 = float(sys.argv[1]); w0n = float(sys.argv[2]); L0n = float(sys.argv[3]); extra = sys.argv[4:]
n0 = 1e24; wp=np.sqrt(n0*ct.e**2/(ct.epsilon_0*ct.m_e)); s = ct.c/wp; E0=ct.m_e*ct.c*wp/ct.e
tag = f'a{a0}'
import shutil; shutil.rmtree('d_'+tag, ignore_errors=True); shutil.rmtree('q_'+tag, ignore_errors=True)
laser = GaussianPulse(xi_c=0.0, a_0=a0, w_0=w0n*s, tau=L0n*s*np.sqrt(2*np.log(2))/ct.c, z_foc=0.0, l_0=0.8e-6)
bunch = get_gaussian_bunch_from_size(1e-6, 1e-6, 1e-7, 1e-7, 200, 0.01, 1e-7, -12*s, 1e-15, 100)
kw = dict(max_gamma=float(os.environ.get('WT_MAXG', 10)))
stage = PlasmaStage(length=1e-6, density=n0, wakefield_model='quasistatic_2d', n_out=1, laser=laser, laser_evolution=False,
                    r_max=14*s, r_max_plasma=13*s, xi_min=-14*s, xi_max=6*s, n_r=int(os.environ.get('WT_NR',1400)), n_xi=int(os.environ.get('WT_NXI',2000)), ppc=int(os.environ.get('WT_PPC',4)), dz_fields=1e-6, **kw)
import contextlib, io as pyio
with contextlib.redirect_stderr(pyio.StringIO()): stage.track(bunch, opmd_diag=True, diag_dir='d_'+tag)
subprocess.run([os.environ.get('QUARZ', '../build/quarz'), 'laser_vs_waket.in', f'laser.a0={a0}', f'laser.w0={w0n}', f'laser.L0={L0n}', 'output.dir=q_'+tag] + extra,
               stdout=subprocess.DEVNULL, env=dict(os.environ, OMP_PROC_BIND='false'))
S = io.Series(f'd_{tag}/hdf5/data%08T.h5', io.Access.read_only); it = S.iterations[0]
m = it.meshes['E']; Ez = m['z'].load_chunk(); S.flush(); Ez = Ez[0,0,:]*m['z'].unit_SI/E0
z = (m.grid_global_offset[-1] + m.grid_spacing[-1]*np.arange(Ez.size))/s
f = read_fields(f'q_{tag}/fields_000000.bin'); xq = -f['xi'] - 6.0   # distance behind the QUARZ head, - 6
xw = -z
ezw = np.interp(xq, xw[::-1], Ez[::-1])
sel = (xq > -2) & (xq < 6)
scale = np.abs(ezw[sel]).max()
print(f'a0={a0}: max|Ez| {scale:.3f}, max |QUARZ - WakeT| / max = {np.abs(f["ez"][sel,0]-ezw[sel]).max()/scale:.3f}')
for x in (0, 2, 4, 6):
    k = np.argmin(abs(xq-x)); print(f'   {x}: QUARZ {f["ez"][k,0]:+.4f}  Wake-T {ezw[k]:+.4f}')
