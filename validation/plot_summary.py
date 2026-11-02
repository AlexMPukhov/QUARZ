import sys; sys.path.insert(0, '../tools')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quarz_read import read_fields

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
# (a) grid comparison: focusing gradient in the pinched witness
ref = read_fields('out_im_u0001/fields_000000.bin')
xs = ref['xi'][np.argmax(ref['ni'][:, 0])]   # slice of strongest ion collapse
a = ax[0, 0]
for name, lab, st in [('u0001', 'uniform 0.00025 (32000 cells, reference)', 'k-'),
                      ('u001', 'uniform 0.01 (800 cells)', 'C3--'),
                      ('u0005', 'uniform 0.005 (1600 cells)', 'C1-.'),
                      ('stretched', 'stretched 0.0005→0.05 (512 cells)', 'C0:')]:
    d = read_fields(f'out_im_{name}/fields_000000.bin')
    kk = np.argmin(abs(d['xi'] - xs))
    m = (d['r'] <= 0.05) & (d['r'] > 0)
    a.plot(d['r'][m], ((d['er'][kk] - d['bth'][kk]) / d['r'])[m], st, lw=2 if name == 'stretched' else 1.3, label=lab,
           marker='o' if name == 'stretched' else None, ms=2.5)
a.axhline(0.5, color='grey', lw=0.8, label='unperturbed ion column (0.5)')
a.set_xlabel(r'$r\ [c/\omega_p]$'); a.set_ylabel(r'focusing gradient $(E_r - B_\theta)/r$')
a.set_title(r'Pinched witness ($\sigma_r=0.01$), mobile H ions, $\xi=%.2f$' % xs); a.legend(fontsize=8)
# (b) blowout map on stretched grid
d = read_fields('out_im_stretched/fields_000000.bin')
a = ax[0, 1]
m = a.pcolormesh(d['xi'], d['r'], d['ne'].T, shading='auto', cmap='Greys', vmin=0, vmax=3)
a.contour(d['xi'], d['r'], abs(d['rhob']).T, levels=[1, 1000], colors=['C1', 'C0'], linewidths=0.8)
a.set_ylim(0, 3); a.set_xlabel(r'$\xi = z - t$'); a.set_ylabel('r'); a.set_title('Plasma electrons + beam contours (stretched grid, 512 cells)')
fig.colorbar(m, ax=a)
# (c) AWAKE-like SMI: Ez on axis at three positions
a = ax[1, 0]
for n, c in [(0, 'C7'), (100, 'C0'), (200, 'C3')]:
    f = read_fields('out_awake400/fields_%06d.bin' % n)
    a.plot(f['xi'], f['ez'][:, 0], color=c, lw=0.6, label='z = %.0f m' % (f['t'] * 0.2e-3))
a.set_xlabel(r'$\xi\ [c/\omega_p]$ (0.2 mm)'); a.set_ylabel(r'$E_z(r=0)\ [E_0 = 2.54$ GV/m$]$')
a.set_title('Seeded self-modulation of a 400 GeV proton bunch'); a.legend(fontsize=8); a.set_xlim(-2, 300)
# (d) growth
a = ax[1, 1]
zs, amps = [], {x: [] for x in (50, 150, 280)}
for n in range(0, 401, 50):
    f = read_fields('out_awake400/fields_%06d.bin' % n); zs.append(f['t'] * 0.2e-3)
    for x in amps:
        mm = (f['xi'] > x - 10) & (f['xi'] < x + 10); amps[x].append(abs(f['ez'][mm, 0]).max() * 2.54e3)
for x in amps: a.plot(zs, amps[x], 'o-', label=r'$\xi \approx %d$ (%.0f cm behind seed)' % (x, x * 0.02))
a.set_xlabel('z [m]'); a.set_ylabel(r'max $|E_z|$ on axis [MV/m]'); a.set_title('Wakefield growth and saturation'); a.legend(fontsize=8)
fig.savefig('quarz_validation_summary.png', dpi=130)
print('ok')
