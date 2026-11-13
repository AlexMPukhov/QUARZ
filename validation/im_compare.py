import sys; sys.path.insert(0,'../tools')
from quarz_read import read_fields
import numpy as np
REF = sys.argv[1] if len(sys.argv) > 1 else 'u0001'          # reference run (default: uniform, N = 32000)
RUNS = sys.argv[2:] if len(sys.argv) > 2 else ['u001', 'u0005', 'stretched']
ref = read_fields(f'out_im_{REF}/fields_000000.bin')
xi = ref['xi']; ks = np.where((xi>-7.2)&(xi<-6.4))[0]
def sample(d, name, rr):
    kk = [np.argmin(abs(d['xi']-xi[k])) for k in ks]
    return np.array([np.interp(rr, d['r'], d[name][k]) for k in kk])
def wr(d, rr):
    kk = [np.argmin(abs(d['xi']-xi[k])) for k in ks]
    return np.array([np.interp(rr, d['r'], d['er'][k]-d['bth'][k]) for k in kk])
R0 = {'ni0': sample(ref,'ni',0.0), 'wr01': wr(ref,0.01), 'wr03': wr(ref,0.03), 'ez0': sample(ref,'ez',0.0)}
label = "uniform dr=0.00025" if REF == "u0001" else REF
print(f"reference {label} (N={len(ref['r'])-1}): max n_i(r=0) in witness = {R0['ni0'].max():.1f}, max (Er-Bth)/r at r=0.01: {R0['wr01'].max()/0.01:.3f} (unperturbed ion column: 0.5)")
print(f"{'grid':12s} {'N':>5s} {'err n_i(0)':>11s} {'err Wr(.01)':>12s} {'err Wr(.03)':>12s} {'err Ez(0)':>10s}")
for name in RUNS:
    d = read_fields(f'out_im_{name}/fields_000000.bin')
    e = lambda a,b: abs(a-b).max()/abs(b).max()
    print(f"{name:12s} {len(d['r'])-1:5d} {e(sample(d,'ni',0.0),R0['ni0']):11.3%} {e(wr(d,0.01),R0['wr01']):12.3%} {e(wr(d,0.03),R0['wr03']):12.3%} {e(sample(d,'ez',0.0),R0['ez0']):10.3%}")
