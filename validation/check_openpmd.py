"""compare openPMD output with the native QUARZ output of the same run (both written)
usage: check_openpmd.py dir [native|uniform]"""
import sys, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import numpy as np, openpmd_api as io
from quarz_read import read_fields, read_beam
d = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'native'
ext = glob.glob(d + '/openpmd/data_*')[0].rsplit('.', 1)[1]
S = io.Series(d + '/openpmd/data_%06T.' + ext, io.Access.read_only)
n0 = S.get_attribute('quarz_n0_SI') if 'quarz_n0_SI' in S.attributes else None
worst = 0.0
for n, it in S.iterations.items():
    t = it.time
    if 'E' in it.meshes:
        f = read_fields(f'{d}/fields_{n:06d}.bin')
        r = f['r']; xi = f['xi']
        for rec, comps in [('E', {'r': 'er', 't': 'eth', 'z': 'ez'}), ('B', {'r': 'br', 't': 'bth', 'z': 'bz'}),
                           ('psi', {io.Mesh_Record_Component.SCALAR: 'psi'}), ('n_e', {io.Mesh_Record_Component.SCALAR: 'ne'}),
                           ('n_i', {io.Mesh_Record_Component.SCALAR: 'ni'}), ('rho_beam', {io.Mesh_Record_Component.SCALAR: 'rhob'})]:
            m = it.meshes[rec]
            for c, name in comps.items():
                rc = m[c]; a = rc.load_chunk(); S.flush(); a = a * rc.unit_SI
                # native data in code units -> SI via the same unit
                modes = [f[name]] + ([f[name + '_c'], f[name + '_s']] if a.shape[0] == 3 else [])
                zoff = m.grid_global_offset[1]; dz = m.grid_spacing[1]
                assert abs(zoff - (t - xi[-1])) < 1e-9 * max(1, abs(t)), (zoff, t - xi[-1])
                for k, fm in enumerate(modes):
                    nat = fm[::-1, :].T * rc.unit_SI   # (r, z) with z increasing
                    if mode == 'native':
                        ref = nat
                    else:
                        rr = np.arange(a.shape[1]) * m.grid_spacing[0]
                        ref = np.array([np.interp(rr, r, nat[:, kz]) for kz in range(nat.shape[1])]).T
                    den = np.abs(ref).max() or 1.0
                    worst = max(worst, np.abs(a[k] - ref).max() / den)
        print(f'iteration {n}: t = {t:g} ({it.time_unit_SI:.3e} s), meshes {sorted(it.meshes)}  geometry {it.meshes["E"].geometry}')
        if mode == 'native' and ext == 'h5':
            # rectilinear mesh (docs/openpmd_rectilinear.md) through the generic reader shared with GEM-PIC-RZ
            from openpmd_rectilinear import read_mesh
            mm = read_mesh(f'{d}/openpmd/data_{n:06d}.h5', 'E', 'z', iteration=n)
            gU = it.meshes['E'].grid_unit_SI
            E = it.meshes['E']
            form = 'data set' if 'gridCoordinates' in E.attributes else 'attribute'   # openPMD 2.0 / interim form
            assert mm.geometry == 'thetaMode'
            assert form == 'data set' or E.get_attribute('gridType') == 'rectilinear'
            e_r = np.abs(mm.axes['r'] - r * gU).max() / (r.max() * gU)
            a = it.meshes['E']['z'].load_chunk(); S.flush()
            e_d = np.abs(mm.data - a * it.meshes['E']['z'].unit_SI).max() / max(np.abs(mm.data).max(), 1e-300)
            print(f'  rectilinear reader ({form} form): radial nodes {e_r:.1e}, E_z data {e_d:.1e}')
            worst = max(worst, e_r, e_d)
    for sp_name, sp in it.particles.items():
        b = read_beam(f'{d}/beam_{sp_name}_{n:06d}.bin')
        x = sp['position']['x'].load_chunk(); z = sp['position']['z'].load_chunk(); w = sp['weighting'][io.Record_Component.SCALAR].load_chunk()
        pz = sp['momentum']['z'].load_chunk(); S.flush()
        uL = sp['position']['x'].unit_SI
        ia = np.argsort(b['x']); ib = np.argsort(x)
        e1 = np.abs(b['x'][ia] * uL - x[ib] * uL).max() / np.abs(b['x']).max() / uL
        e2 = np.abs(np.sort(t - b['xi']) - np.sort(z)).max() / max(1, abs(t))
        e3 = abs(w.sum() - b['w'].sum()) / b['w'].sum()
        print(f'  particles {sp_name}: N = {len(x)} (native {len(b["x"])}), x {e1:.1e}, z {e2:.1e}, sum w {e3:.1e}')
        worst = max(worst, e1, e2, e3)
print(f'max relative difference openPMD vs native: {worst:.2e}')
