import os, sys; _R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.join(_R, 'tools'))
sys.path.insert(0, '.')
from rf import read_lazy as read_fields
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.6,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
D = {n: read_fields(f'out_{n}/fields_000000.bin') for n in ['u0.01', 'u0.005', 's1', 's0.03125', 's1_full']}
def mirror(d, name, rmax, xlo=-1e9, xhi=1e9):
    k0 = int(np.searchsorted(d['xi'], xlo)); k1 = int(np.searchsorted(d['xi'], xhi)) + 1
    j = int(np.searchsorted(d['r'], rmax * 1.0001))
    r = d['r'][:j]; f = np.array(d[name][k0:k1, :j])
    return d['xi'][k0:k1],  np.concatenate([-r[:0:-1], r]), np.concatenate([f[:, :0:-1], f], axis=1)
fig = plt.figure(figsize=(7.0, 4.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.12, right=0.9)
# (a) overview: electron density, stretched grid, beams as contours
ax = fig.add_subplot(gs[0, :])
d = D['s1_full']; xx, x, f = mirror(d, 'ne', 3.0)
im = ax.pcolormesh(xx, x, f.T, cmap='Greys', vmin=0, vmax=4, shading='gouraud', rasterized=True)
_, _, fb = mirror(d, 'rhob', 3.0)
ax.contour(xx, x, -fb.T, levels=[0.5, 5, 50, 500], colors='#c4511a', linewidths=0.6)
ax.set_xlim(0, 9); ax.set_ylim(-3, 3); ax.set_xlabel(r'$\xi\ (c/\omega_p)$'); ax.set_ylabel(r'$x\ (c/\omega_p)$')
ax.text(3.0, 0.55, 'driver', color='#c4511a', fontsize=7, ha='center')
ax.text(6.8, 0.35, 'witness', color='#c4511a', fontsize=7, ha='center')
cax = fig.add_axes([0.915, 0.60, 0.012, 0.28]); cb = fig.colorbar(im, cax=cax); cb.set_label(r'$n_e/n_0$')
ax.set_title(r'(a) electron density (grey) and beam density (contours), stretched grid with 512 cells', fontsize=8, loc='left')
# (b-e) ion density near the axis in the witness
panels = [('u0.01', 'uniform, 800 cells'), ('u0.005', 'uniform, 1600 cells'), ('s1', 'stretched, 512 cells'), ('s0.03125', 'reference, 14617 cells')]
for i, (n, lab) in enumerate(panels):
    ax = fig.add_subplot(gs[1, i]); d = D[n]; xx, x, f = mirror(d, 'ni', 0.042, 6.38, 7.3)
    im = ax.pcolormesh(xx, x, f.T, cmap='magma', vmin=0.8, vmax=2.4, shading='gouraud', rasterized=True)
    ax.set_xlim(6.4, 7.25); ax.set_ylim(-0.04, 0.04); ax.set_xlabel(r'$\xi$')
    if i == 0: ax.set_ylabel(r'$x\ (c/\omega_p)$')
    else: ax.set_yticklabels([])
    ax.set_title(f'({"bcde"[i]}) {lab}', fontsize=7.5, loc='left')
cax = fig.add_axes([0.915, 0.11, 0.012, 0.32]); cb = fig.colorbar(im, cax=cax); cb.set_label(r'$n_i/n_0$')
fig.savefig(os.path.join(_R, 'paper', 'tex', 'figs', 'fig_maps.pdf'), dpi=300, bbox_inches='tight'); fig.savefig('fig_maps.png', dpi=150, bbox_inches='tight')
