import os, glob, json, warnings, sys, time, signal
import numpy as np
from collections import defaultdict
from rdkit import Chem
from posecheck import PoseCheck
from scipy import stats
from prolif.exceptions import FragmentedResidueError

warnings.filterwarnings('ignore')
FragmentedResidueError.on_error = 'warn'

base = 'eval/7xkj/output/msfp_sensitivity_full'
protein_root = 'data/test_set'
cfgmap = {'none': 'Full', 'r1': 'w/o r1', 'r2': 'w/o r2', 'r3': 'w/o r3'}
OUT = 'eval/msfp_clash_results.json'
TIMEOUT = 20

class Timeout(Exception):
    pass

def _handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _handler)

pockets = set(os.path.basename(d)[:-5] for d in glob.glob(base + '/*_none'))
common = sorted(p for p in pockets
                if all(os.path.isdir(f'{base}/{p}_{c}') for c in cfgmap))
print('common pockets:', len(common), flush=True)

results = {}
if os.path.exists(OUT):
    try:
        results = json.load(open(OUT))
        print('resume: loaded', len(results), 'pocket-config entries', flush=True)
    except Exception:
        results = {}
pc = PoseCheck()
count = 0
t0 = time.time()

for pi, p in enumerate(common):
    # skip already-computed pockets
    done = all(f'{p}_{c}' in results for c in cfgmap)
    if done:
        continue
    pdbs = glob.glob(f'{protein_root}/{p}/*_rec.pdb')
    if not pdbs:
        continue
    pdb = pdbs[0]
    try:
        signal.alarm(120)  # per-pocket timeout
        pc.load_protein_from_pdb(pdb)
        signal.alarm(0)
    except (Timeout, Exception):
        signal.alarm(0)
        continue
    for cfg in cfgmap:
        d = f'{base}/{p}_{cfg}'
        clashes = []
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
                signal.alarm(0)
            except (Timeout, Exception):
                signal.alarm(0)
                continue
            clashes.append(cl)
            count += 1
            if count % 150 == 0:
                pc = PoseCheck()
                pc.load_protein_from_pdb(pdb)
        if clashes:
            results[f'{p}_{cfg}'] = float(np.mean(clashes))
    signal.alarm(0)
    if (pi + 1) % 5 == 0:
        el = time.time() - t0
        json.dump(results, open(OUT, 'w'))
        print(f'[{pi+1}/{len(common)}] pockets, {count} mols, {el:.0f}s', flush=True)

json.dump(results, open(OUT, 'w'))

# paired analysis on pockets common to all 4 configs
bycfg = defaultdict(dict)
for k, v in results.items():
    p, cfg = k.rsplit('_', 1)
    bycfg[cfg][p] = v
shared = sorted(set.intersection(*[set(bycfg[c]) for c in cfgmap]))
print('\nn shared pockets:', len(shared), flush=True)

vals = {c: np.array([bycfg[c][p] for p in shared]) for c in cfgmap}
print('\n=== clash (mean over pockets, paired n=%d) ===' % len(shared), flush=True)
for c in cfgmap:
    print(f'  {cfgmap[c]:8s} {vals[c].mean():.4f} +/- {vals[c].std(ddof=1)/np.sqrt(len(shared)):.4f}', flush=True)
print('\npaired Full - w/o rX (clash, lower is better):', flush=True)
for c in ['r1', 'r2', 'r3']:
    d = vals['none'] - vals[c]
    t, p = stats.ttest_rel(vals['none'], vals[c])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'  vs {cfgmap[c]:6s}: diff={d.mean():+.4f}  t={t:+.2f}  p={p:.3f}  95%CI=[{lo:+.4f},{hi:+.4f}]', flush=True)
print('saved', OUT, flush=True)
