import os, sys; _R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.join(_R, 'tools'))
from quarz_read import read_fields
import numpy as np, re, json
ref = read_fields('out_' + (sys.argv[1] if len(sys.argv) > 1 else 'ref') + '/fields_000000.bin')
xi = ref['xi']; ks = np.where((xi>-7.2)&(xi<-6.4))[0]
def kk(d): return [np.argmin(abs(d['xi']-xi[k])) for k in ks]
def samp(d, f, rr): return np.array([np.interp(rr, d['r'], f(d,k)) for k in kk(d)])
F = {'ni0': (lambda d,k: d['ni'][k], 0.0), 'wr01': (lambda d,k: d['er'][k]-d['bth'][k], 0.01),
     'wr03': (lambda d,k: d['er'][k]-d['bth'][k], 0.03), 'ez0': (lambda d,k: d['ez'][k], 0.0)}
R0 = {n: samp(ref, f, r) for n,(f,r) in F.items()}
out = {}
for name in ['ref','u0.02','u0.01','u0.005','u0.0025','u0.00125','s4','s2','s1','s0.5','s0.25','s0.125','s0.0625']:
    d = read_fields(f'out_{name}/fields_000000.bin')
    t = float(re.search(r'sweep ([0-9.]+) s', open(f'log_{name}.txt').read()).group(1))
    e = {n: float(abs(samp(d,f,r)-R0[n]).max()/abs(R0[n]).max()) for n,(f,r) in F.items()}
    out[name] = dict(N=len(d['r'])-1, t=t, h0=float(d['r'][1]), **e)
    print(f"{name:9s} N={out[name]['N']:5d} t={t:6.2f}  " + "  ".join(f"{n} {v:.2e}" for n,v in e.items()))
json.dump(out, open('errs_' + (sys.argv[1] if len(sys.argv) > 1 else 'ref') + '.json','w'), indent=1)
