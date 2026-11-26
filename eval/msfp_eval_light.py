"""Lightweight MSFP sensitivity evaluation: Vina Score / Vina Min + QED + SA only.

Rationale: Vina Score is the paper's primary affinity metric (score_only ~1s),
whereas Vina Dock (re-docking, ~90s/molecule at exhaustiveness 16) dominates the
cost of the full pipeline but is a secondary metric and unnecessary for a
compression-ratio sensitivity analysis. PoseCheck strain/clash are stability
metrics, likewise not needed here.

Each (pocket, config) unit is evaluated by one worker; results are written
immediately -> resumable.
"""
import os
import sys
import glob
import json
import argparse
import warnings
import multiprocessing as mp

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
_POSECHECK_ROOT = os.path.join(CODE_DIR, 'posecheck')
if os.path.isdir(os.path.join(_POSECHECK_ROOT, 'posecheck')) and _POSECHECK_ROOT not in sys.path:
    sys.path.insert(0, _POSECHECK_ROOT)

from rdkit import Chem
import numpy as np
from core.evaluation.docking_vina import VinaDockingTask
from core.evaluation.utils import scoring_func

ABLATE = ['none', 'r1', 'r2', 'r3']
METRICS = ['vina_score', 'vina_minimize', 'qed', 'sa']


def _eval_unit(args):
    pdb, out_dir, num_samples, cpu = args
    warnings.filterwarnings('ignore')
    def _jsonable(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    results = []
    for i in range(num_samples):
        out_fn = os.path.join(out_dir, f'{i}.sdf')
        if not os.path.exists(out_fn):
            continue
        try:
            mol = Chem.SDMolSupplier(out_fn, removeHs=False)[0]
            chem = {}
            chem.update(scoring_func.get_chem(mol))
            chem['atom_num'] = mol.GetNumAtoms()
            vt = VinaDockingTask(pdb, mol)
            chem['vina_score'] = vt.run(mode='score_only', exhaustiveness=16, cpu=cpu)[0]['affinity']
            chem['vina_minimize'] = vt.run(mode='minimize', exhaustiveness=16, cpu=cpu)[0]['affinity']
            chem['sample_id'] = i
            results.append({k: _jsonable(v) for k, v in chem.items()})
        except Exception:
            pass

    stats = {}
    for m in METRICS:
        vals = [r[m] for r in results if isinstance(r.get(m), (int, float))]
        if vals:
            stats[m] = {'count': len(vals), 'mean': float(sum(vals) / len(vals))}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'docking_analysis_light.json'), 'w') as f:
        json.dump(results, f)
    with open(os.path.join(out_dir, 'docking_analysis_light_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return out_dir, len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='eval/7xkj/output/msfp_sensitivity_full')
    ap.add_argument('--test_root', default='data/test_set')
    ap.add_argument('--num_samples', type=int, default=10)
    ap.add_argument('--n_cpu', type=int, default=8)
    ap.add_argument('--n_pockets', type=int, default=0, help='0 = all pockets')
    args = ap.parse_args()

    pockets = sorted(os.listdir(args.test_root))
    if args.n_pockets > 0:
        pockets = pockets[:args.n_pockets]

    units = []
    for cfg_name in ABLATE:
        for pocket in pockets:
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            if os.path.exists(os.path.join(out_dir, 'docking_analysis_light_stats.json')):
                continue
            if not os.path.exists(os.path.join(out_dir, '0.sdf')):
                print(f'WARN missing generation: {pocket}_{cfg_name}')
                continue
            pdb = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_rec.pdb')))[0]
            units.append((pdb, out_dir, args.num_samples, 4))

    print(f'pocket-config units to evaluate: {len(units)}', flush=True)
    done = 0
    if units:
        with mp.Pool(processes=min(args.n_cpu, len(units))) as pool:
            for out_dir, n in pool.imap_unordered(_eval_unit, units):
                done += 1
                print(f'[{done}/{len(units)}] {os.path.basename(out_dir)}: {n} mols', flush=True)

    import statistics as st
    print('\n=== AGGREGATE (mean +/- SE over pockets) ===', flush=True)
    rows = {}
    for cfg_name in ABLATE:
        per = {m: [] for m in METRICS}
        for pocket in pockets:
            f = os.path.join(args.out_root, f'{pocket}_{cfg_name}', 'docking_analysis_light_stats.json')
            if not os.path.exists(f):
                continue
            s = json.load(open(f))
            for m in METRICS:
                v = s.get(m, {})
                if v.get('mean') is not None:
                    per[m].append(v['mean'])
        rows[cfg_name] = per
        print(f'--- {cfg_name}  (pockets={len(per["vina_score"])}) ---')
        for m in METRICS:
            vals = per[m]
            if not vals:
                continue
            se = st.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else float('nan')
            print(f'  {m:12s} mean={st.mean(vals):.3f}  se={se:.3f}  sd={st.stdev(vals):.3f}')

    with open(os.path.join(args.out_root, 'summary_light.csv'), 'w') as f:
        f.write('config,n_pockets,' + ','.join(METRICS) + '\n')
        for cfg_name in ABLATE:
            per = rows[cfg_name]
            f.write(f"{cfg_name},{len(per['vina_score'])},"
                    + ','.join(f'{st.mean(per[m]):.4f}' if per[m] else 'NA' for m in METRICS) + '\n')
    print('saved:', os.path.join(args.out_root, 'summary_light.csv'), flush=True)


if __name__ == '__main__':
    main()
