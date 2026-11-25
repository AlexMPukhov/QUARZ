import os; _R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
# Fig. closure: run closure.sh first (in the current directory)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'legend.frameon': False})
CU, CS, CC = '#1f5fbf', '#c4511a', '0.45'


def ld(n):
    a = np.loadtxt(f'out_closure_{n}/axis_000000.txt', comments='#')
    return a[:, 0], a[:, 1], a[:, 2]


def l2(ref, run, col, lo=-11.9, hi=-9.2):
    A, B = ld(ref), ld(run)
    xs = np.arange(lo, hi, 0.005)
    r0, r1 = np.interp(xs, A[0], A[col]), np.interp(xs, B[0], B[col])
    return np.sqrt(np.mean((r1 - r0) ** 2) / np.mean(r0 ** 2))


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axs[0]
for n, lab, c, ls, lw in [('u0.005_0.00125', r'uniform $\Delta r=0.005$, cold', CU, ':', 1.1),
                          ('cold_0.000625', r'stretched, cold, $\Delta\xi=6.25\times10^{-4}$', CC, '-', 0.7),
                          ('cold_0.005', r'stretched, cold, $\Delta\xi=0.005$', CS, '-', 0.9),
                          ('a5e-3_0.000625', r'stretched, $a=0.005$, $\Delta\xi=6.25\times10^{-4}$', 'k', '-', 1.2),
                          ('a5e-3_0.005', r'stretched, $a=0.005$, $\Delta\xi=0.005$', CS, '--', 1.0)]:
    x, e, _ = ld(n)
    ax.plot(x, e, color=c, ls=ls, lw=lw, label=lab)
ax.set_xlim(-9.3, -8.85); ax.set_ylim(-24, 2.5)
ax.annotate(r'$-78$', xy=(9.051, -24), xytext=(9.11, -21.5), fontsize=7, color=CC,
            arrowprops=dict(arrowstyle='->', lw=0.6, color=CC))
ax.set_xlabel(r'$\xi$'); ax.set_ylabel(r'$E_z(r=0)$')
ax.set_title('(a) on-axis field at the bubble closure', fontsize=8, loc='left')
ax.legend(fontsize=6.3, loc='lower left')
ax = axs[1]
dx = np.array([0.005, 0.0025, 0.00125])
for pre, ref, c, mk in [('cold', 'cold_0.000625', CS, 'o'), ('a5e-3', 'a5e-3_0.000625', 'k', 's')]:
    lab = 'cold' if pre == 'cold' else r'$a=0.005$'
    ax.loglog(dx, [l2(ref, f'{pre}_{d:g}', 1) for d in dx], mk + '-', color=c, ms=4, lw=1.1, label=lab + r', $E_z$')
    ax.loglog(dx, [l2(ref, f'{pre}_{d:g}', 2) for d in dx], mk + '--', color=c, mfc='white', ms=4, lw=1.0, label=lab + r', $\psi$')
for h, m, xo in [('0.001', '^', 0.93), ('0.00025', 'v', 1.07)]:
    ax.loglog([0.00125 * xo], [l2('a5e-3_0.000625', f'a5e-3_h{h}', 1)], m, color=CU, ms=4.5,
              label=r'$a=0.005$, $E_z$, $h_0=$' + (r'$10^{-3}$' if h == '0.001' else r'$2.5\times10^{-4}$'))
ax.set_xlabel(r'$\Delta\xi$'); ax.set_ylabel(r'rms rel. error, $-11.9<\xi<-9.2$')
ax.set_xlim(9e-4, 6.5e-3); ax.set_ylim(4e-5, 0.1)
ax.set_xticks(dx); ax.set_xticklabels(['0.005', '0.0025', '0.00125']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(True, which='major', lw=0.3, color='0.85')
ax.set_title('(b) wake behind the closure', fontsize=8, loc='left')
ax.legend(fontsize=6.0, loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.8)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(os.path.join(_R, 'paper', 'tex', 'figs', 'fig_closure.pdf')); fig.savefig('fig_closure.png', dpi=160)
