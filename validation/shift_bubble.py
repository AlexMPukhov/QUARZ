import sys; sys.path.insert(0,'../tools')
from quarz_read import read_fields
import numpy as np
ref=read_fields(sys.argv[1]+'/fields_000000.bin'); sh=read_fields(sys.argv[2]+'/fields_000000.bin'); d=float(sys.argv[3])
r,xi=ref['r'],ref['xi']
dr=lambda f: np.gradient(f,r,axis=1)
orr=lambda f: np.where(r>0,f/np.where(r>0,r,1),0)
pred={"psi_c":-d*dr(ref["psi"]),"er_c":-d*dr(ref["er"]),"eth_s":d*orr(ref["er"]),"bth_c":-d*dr(ref["bth"]),"br_s":-d*orr(ref["bth"])}
inb=(ref['ne']<0.01)&(r[None,:]<1.0)&(r[None,:]>0.02)&(xi[:,None]>-8)&(xi[:,None]<-3.5)
head=(xi[:,None]>-1.5)&(r[None,:]<1.0)
print(sys.argv[4] if len(sys.argv)>4 else '', ' '.join('%s bubble %.1e head %.1e |'%(n,abs(sh[n][inb]-p[inb]).max()/abs(p[inb]).max(),abs(sh[n][head]-p[head]).max()/abs(p[head]).max()) for n,p in pred.items()))
