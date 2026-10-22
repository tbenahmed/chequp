"""Minimal reader for single-level AMReX plotfiles (1D/2D) - no yt needed, exact doubles."""
import numpy as np, os, glob, re

def read_plotfile(path):
    with open(os.path.join(path, 'Header')) as f:
        lines = f.read().split('\n')
    nvar = int(lines[1]); names = lines[2:2+nvar]
    dim = int(lines[2+nvar]); time = float(lines[3+nvar])
    finest = int(lines[4+nvar])
    lo = [float(x) for x in lines[5+nvar].split()]; hi = [float(x) for x in lines[6+nvar].split()]
    dom = lines[8+nvar]
    nums = [int(x) for x in re.findall(r'-?\d+', dom)]
    shape = [nums[dim+d]-nums[d]+1 for d in range(dim)]
    data = np.zeros(shape + [nvar])
    # level 0 only (we run without AMR for convergence studies)
    with open(os.path.join(path, 'Level_0', 'Cell_H')) as f:
        ch = f.read()
    boxes = re.findall(r'\(\(([\d, -]+)\) \(([\d, -]+)\) \(([\d, -]+)\)\)', ch)
    fods = re.findall(r'FabOnDisk: (\S+) (\d+)', ch)
    for (blo, bhi, _), (fn, off) in zip(boxes, fods):
        blo = [int(x) for x in blo.split(',')]; bhi = [int(x) for x in bhi.split(',')]
        bs = [bhi[d]-blo[d]+1 for d in range(dim)]
        with open(os.path.join(path, 'Level_0', fn), 'rb') as f:
            f.seek(int(off)); hdr = f.readline().decode()
            n = int(np.prod(bs))
            arr = np.fromfile(f, dtype='<f8', count=n*nvar).reshape([nvar] + bs[::-1])
        arr = np.moveaxis(arr, 0, -1)
        arr = arr.transpose(list(range(dim))[::-1] + [dim])
        sl = tuple(slice(blo[d], bhi[d]+1) for d in range(dim))
        data[sl] = arr
    return dict(time=time, names=names, data=data, lo=lo, hi=hi, shape=shape, finest=finest)

def fields(pf):
    d = pf['data']; n = pf['names']; g = lambda k: d[..., n.index(k)]
    rho = g('density')
    return {'rho': rho, 'Te': g('Temp'), 'f': g('rho_f_heavie')/rho, 'ion': g('rho_H1')/rho,
            'eint': g('rho_e'), 'vr': g('xmom')/rho}

def plotfiles(run):
    return sorted(glob.glob(os.path.join(run, 'plt*')), key=lambda p: int(re.findall(r'(\d+)$', p)[0]))

def by_time(run):
    out = {}
    for p in plotfiles(run):
        pf = read_plotfile(p); out[round(pf['time']*1e12)/1e12] = p
    return out
