"""compare two openPMD series (HDF5) of QUARZ runs, e.g. serial vs MPI: usage cmp_openpmd.py dirA dirB"""
import sys, numpy as np, openpmd_api as io
A = io.Series(sys.argv[1] + '/openpmd/data_%06T.h5', io.Access.read_only)
B = io.Series(sys.argv[2] + '/openpmd/data_%06T.h5', io.Access.read_only)
worst = 0.0
for n, ia in A.iterations.items():
    ib = B.iterations[n]
    assert ia.time == ib.time
    for name, m in ia.meshes.items():
        for c, rc in m.items():
            a = rc.load_chunk(); b = ib.meshes[name][c].load_chunk(); A.flush(); B.flush()
            assert a.shape == b.shape
            worst = max(worst, np.abs(a - b).max() / (np.abs(a).max() or 1))
    for name, sp in ia.particles.items():
        for rn in ('position', 'momentum'):
            for c, rc in sp[rn].items():
                a = rc.load_chunk(); b = ib.particles[name][rn][c].load_chunk(); A.flush(); B.flush()
                worst = max(worst, np.abs(np.sort(a) - np.sort(b)).max() / (np.abs(a).max() or 1))
# coordinate data sets of rectilinear meshes (docs/openpmd_rectilinear.md) and their references, through h5py
# so that it works with any openPMD-api
import glob, h5py
ncoord = 0
for fa in sorted(glob.glob(sys.argv[1] + '/openpmd/data_*.h5')):
    fb = sys.argv[2] + '/openpmd/' + fa.rsplit('/', 1)[1]
    with h5py.File(fa, 'r') as ha, h5py.File(fb, 'r') as hb:
        for n in ha['data']:
            g = f'/data/{n}/gridCoordinates'
            assert (g in ha) == (g in hb), g
            if g in ha:
                for k in ha[g]:
                    worst = max(worst, np.abs(ha[g][k][()] - hb[g][k][()]).max() / (np.abs(ha[g][k][()]).max() or 1))
                    ncoord += 1
            for name in (ha[f'/data/{n}/meshes'] if f'/data/{n}/meshes' in ha else []):
                ra, rb = ha[f'/data/{n}/meshes/{name}'], hb[f'/data/{n}/meshes/{name}']
                assert list(ra.attrs.get('gridCoordinates', [])) == list(rb.attrs.get('gridCoordinates', [])), name
# (the number of coordinate data sets, ncoord, is not printed: it depends on the openPMD-api the code was built with)
print(f'{sys.argv[1]} vs {sys.argv[2]}: iterations {list(A.iterations)}, max rel diff {worst:.2e}')
