"""Compare QUARZ against the linear-theory wake of a Gaussian driver in a
conducting cylinder of radius R (Green's function with Dirichlet psi(R)=0).

   dn(r,s)  = Int_{-inf}^{s} sin(s-s') rho_b(r,s') ds',   s = -xi (distance behind the head)
   (lap_perp - 1) psi = dn   ->  psi = -Int r' dr' G(r,r') dn(r')
   G = I0(r<) [K0(r>) - K0(R) I0(r>)/I0(R)]
   Ez = d psi/ds = -d psi/dxi,  Er - Btheta = -d psi/dr
usage: python check_linear.py out_dir [label]
"""
import sys
import numpy as np
from scipy.special import i0e, k0e, erf
sys.path.insert(0, "../tools")
from quarz_read import read_fields

d = read_fields(sys.argv[1] + "/fields_000000.bin")
r, xi = d["r"], -d["xi"]   # xi here: s = -xi = distance behind the head of the box
nb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
sr, sx, x0, R, cut =  0.5, 1.0, 4.0, r[-1], 4.0   # driver at xi = -4 (s = 4)

# reference on a fine uniform quadrature grid in r'
rq = np.linspace(0, R, 4001)
wq = np.full_like(rq, rq[1] - rq[0]); wq[0] *= 0.5; wq[-1] *= 0.5
# truncated beam: radial cut at 4 sigma (rejection in r), longitudinal cut at 4 sigma
radial = np.where(rq <= cut * sr, np.exp(-rq**2 / (2 * sr**2)), 0.0)
# longitudinal Green integral: F(xi) = Int sin(xi-xi') f(xi') dxi', f = gaussian truncated
xg = np.linspace(x0 - cut * sx, x0 + cut * sx, 4001)
fg = np.exp(-(xg - x0)**2 / (2 * sx**2)); dxg = xg[1] - xg[0]
def conv(fun):
    return np.array([np.sum(np.where(xg <= x, fun(x - xg), 0) * fg) * dxg for x in xi])
S = conv(np.sin)     # dn  ~ -nb * radial * S
C = conv(np.cos)     # d dn/dxi
rho_b = -nb * radial  # electron beam


def green(ra, rb):
    rl, rg = np.minimum(ra, rb), np.maximum(ra, rb)
    # scaled Bessel to avoid overflow: I0(x)=i0e(x) e^x, K0(x)=k0e(x) e^-x
    I_l = i0e(rl) * np.exp(rl - rg)       # I0(r<) * e^{-r>}
    t1 = I_l * k0e(rg)                    # I0(r<) K0(r>)
    t2 = i0e(rl) * i0e(rg) * k0e(R) / i0e(R) * np.exp(rl + rg - 2 * R)
    return t1 - t2

Gm = np.nan_to_num(green(r[:, None], rq[None, :]) * (rq * wq)[None, :], nan=0.0, posinf=0.0)
psi_rad = -Gm @ rho_b     # psi = -Int G dn ; dn = rho_b * S
Ez_ref = np.outer(C, psi_rad)
psi_ref = np.outer(S, psi_rad)
dpsi = np.gradient(psi_rad, r)
Wr_ref = -np.outer(S, dpsi)

Ez = d["ez"]
Wr = d["er"] - d["bth"]
iax = 0
ir = np.argmin(abs(r - 0.5))
amp = abs(Ez_ref[:, 0]).max()
e1 = abs(Ez[:, 0] - Ez_ref[:, 0]).max() / amp
e2 = abs(Wr[:, ir] - Wr_ref[:, ir]).max() / abs(Wr_ref[:, ir]).max()
e3 = abs(d["psi"][:, 0] - psi_ref[:, 0]).max() / abs(psi_ref[:, 0]).max()
label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
print(f"{label}: max|Ez|={amp:.4e}  rel.err Ez(r=0)={e1:.3e}  psi(0)={e3:.3e}  (Er-Bth)(r=0.5)={e2:.3e}")
np.savez(sys.argv[1] + "/linear_ref.npz", xi=d["xi"], r=r, Ez_ref=Ez_ref, Wr_ref=Wr_ref)
