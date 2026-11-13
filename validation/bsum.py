import sys; sys.path.insert(0,'../tools')
from quarz_read import read_fields
import numpy as np
d=read_fields(sys.argv[1]+'/fields_000000.bin'); r=d['r']; xi=d['xi']; ez=d['ez'][:,0]
m=(xi>-9.5)&(xi<-6); k=np.argmin(np.where(m,ez,1e9))
def at(x): return np.interp(x,xi,ez)
ir=np.argmin(abs(r-0.5)); wr=(d['er'][:,ir]-d['bth'][:,ir])/r[ir]
print(f"{sys.argv[2]:34s} Ez(-5)={at(-5):+.4f} Ez(-7)={at(-7):+.4f} Ez(-8)={at(-8):+.4f} Ezmin={ez[k]:+.3f} @xi={xi[k]:.3f}  Wr/r(r=.5,xi=-6)={np.interp(-6,xi,wr):.4f}")
