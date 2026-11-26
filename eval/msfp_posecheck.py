import os, glob, json, warnings, sys, time, signal
import numpy as np
from collections import defaultdict
from rdkit import Chem
from posecheck import PoseCheck

warnings.filterwarnings('ignore')

base = 'eval/7xkj/output/msfp_sensitivity_full'
protein_root = 'data/test_set'
cfgmap = {'none': 'Full', 'r1': 'w/o r1', 'r2': 'w/o r2', 'r3': 'w/o r3'}
OUT = 'eval/msfp_posecheck_results.json'
TIMEOUT = 20

class Timeout(Exception):
    pass

def _handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _handler)

pockets = set(os.path.basename(d)[:-5] for d in glob.glob(base + '/*_none'))
common = sorted(p for p in pockets
                if all(os.path.isdir(f'{base}/{p}_{c}') for c in cfgmap))

if len(sys.argv) > 1 and sys.argv[1] == '--smoke':
    common = common[:2]

print('common pockets:', len(common), flush=True)

results = {}
pc = PoseCheck()
count = 0
t0 = time.time()

for pi, p in enumerate(common):
    pdbs = glob.glob(f'{protein_root}/{p}/*_rec.pdb')
    if not pdbs:
        continue
    pdb = pdbs[0]
    pc.load_protein_from_pdb(pdb)
    for cfg in cfgmap:
        d = f'{base}/{p}_{cfg}'
        clashes, strains = [], []
        for i in range(10):
            sf = f'{d}/{i}.sdf'
            if not os.path.exists(sf):
                continue
            mol = Chem.SDMolSupplier(sf, removeHs=False)[0]
            if mol is None:
                continue
            if '.' in Chem.MolToSmiles(mol):
                continue
            try:
                pc.load_ligands_from_mols([mol])
                signal.alarm(TIMEOUT)
                cl = pc.calculate_clashes()[0]
                st = pc.calculate_strain_energy()[0]
                signal.alarm(0)
            except (Timeout, Exception):
                signal.alarm(0)
                continue
            if st is None or st != st:
                st = None
            clashes.append(cl)
            if st is not None:
                strains.append(st)
            count += 1
            if count % 150 == 0:
                pc = PoseCheck()
                pc.load_protein_from_pdb(pdb)
        if clashes:
            results[f'{p}_{cfg}'] = {
                'clash': float(np.mean(clashes)),
                'strain': float(np.mean(strains)) if strains else None,
            }
    if (pi + 1) % 10 == 0:
        el = time.time() - t0
        print(f'[{pi+1}/{len(common)}] pockets, {count} mols, {el:.0f}s elapsed', flush=True)

json.dump(results, open(OUT, 'w'))

agg = defaultdict(list)
for k, s in results.items():
    cfg = k.rsplit('_', 1)[1]
    for mk, v in s.items():
        if v is not None:
            agg[(cfg, mk)].append(v)

print('\n=== aggregate (mean over pockets) ===', flush=True)
for cfg in cfgmap:
    for mk in ['clash', 'strain']:
        vs = [x for x in agg[(cfg, mk)] if x is not None]
        if vs:
            print(f'  {cfgmap[cfg]:8s} {mk:8s} n={len(vs):3d} mean={np.mean(vs):.4f}', flush=True)
print('saved', OUT, flush=True)
