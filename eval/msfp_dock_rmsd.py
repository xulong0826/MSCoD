"""Dock-only pass + pose-stability RMSD for the MSFP leave-one-out sensitivity.

The paper's RMSD metric is the symmetry-corrected RMSD between each GENERATED
molecule and its own VINA DOCKED pose (see core/evaluation/utils/eval_rmsd.py),
i.e. a pose-stability measure: a small RMSD means the generated conformation is
already close to the docking minimum. Computing it requires a re-docking step,
which we skipped in the light evaluation.

This script runs only the dock mode (exhaustiveness 16) and computes the
generated-vs-docked RMSD, then reports the fraction of molecules with RMSD < 2 A
per configuration. Memory-safe (no PoseCheck), resumable per (pocket, config).
"""
import os
import sys
import glob
import json
import argparse
import warnings
import multiprocessing as mp

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from rdkit import Chem
from core.evaluation.docking_vina import VinaDockingTask
from core.evaluation.utils.eval_rmsd import get_symmetry_rmsd, get_pdbqt_mol

ABLATE = ['none', 'r1', 'r2', 'r3']


def _dock_rmsd_unit(args):
    pdb, out_dir, num_samples, cpu = args
    warnings.filterwarnings('ignore')
    results = []
    for i in range(num_samples):
        out_fn = os.path.join(out_dir, f'{i}.sdf')
        if not os.path.exists(out_fn):
            continue
        try:
            mol = Chem.SDMolSupplier(out_fn, removeHs=False)[0]
            r = VinaDockingTask(pdb, mol).run(mode='dock', exhaustiveness=16,
                                              cpu=cpu)[0]
            vina_dock = r['affinity']
            pose = r['pose']
            docked_mol = get_pdbqt_mol(pose)
            if docked_mol is None:
                continue
            mol_h = Chem.RemoveAllHs(mol)
            docked_h = Chem.RemoveAllHs(docked_mol)
            rmsd_val = get_symmetry_rmsd(docked_h, mol_h)
            results.append({
                'sample_id': i,
                'vina_dock': float(vina_dock) if isinstance(vina_dock, (int, float)) and not (isinstance(vina_dock, float) and vina_dock != vina_dock) else None,
                'rmsd': float(rmsd_val) if isinstance(rmsd_val, (int, float)) and not (isinstance(rmsd_val, float) and rmsd_val != rmsd_val) else None,
            })
        except Exception:
            pass

    vd = [x['vina_dock'] for x in results if x.get('vina_dock') is not None]
    rm = [x['rmsd'] for x in results if x.get('rmsd') is not None]
    stats = {
        'n_docked': len(vd),
        'vina_dock_mean': float(np.mean(vd)) if vd else None,
        'n_rmsd': len(rm),
        'rmsd_mean': float(np.mean(rm)) if rm else None,
        'rmsd_lt2_frac': float(np.mean(np.array(rm) < 2)) if rm else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'dock_rmsd_raw.json'), 'w') as f:
        json.dump(results, f)
    with open(os.path.join(out_dir, 'dock_rmsd_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return out_dir, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='eval/7xkj/output/msfp_sensitivity_full')
    ap.add_argument('--test_root', default='data/test_set')
    ap.add_argument('--num_samples', type=int, default=10)
    ap.add_argument('--n_cpu', type=int, default=4)
    ap.add_argument('--cpu_per_vina', type=int, default=4)
    args = ap.parse_args()

    units = []
    for cfg_name in ABLATE:
        for pocket in sorted(os.listdir(args.test_root)):
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            if os.path.exists(os.path.join(out_dir, 'dock_rmsd_stats.json')):
                continue
            if not os.path.exists(os.path.join(out_dir, '0.sdf')):
                continue
            pdb = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_rec.pdb')))[0]
            units.append((pdb, out_dir, args.num_samples, args.cpu_per_vina))

    print(f'pocket-config units to dock: {len(units)}', flush=True)
    done = 0
    if units:
        with mp.Pool(processes=min(args.n_cpu, len(units))) as pool:
            for out_dir, stats in pool.imap_unordered(_dock_rmsd_unit, units):
                done += 1
                print(f"[{done}/{len(units)}] {os.path.basename(out_dir)}: "
                      f"docked={stats['n_docked']} rmsd<2A={stats['rmsd_lt2_frac']}", flush=True)

    import statistics as st
    print('\n=== AGGREGATE (mean +/- SE over pockets) ===', flush=True)
    rows = {}
    for cfg_name in ABLATE:
        frac = []; rmean = []; vd = []
        for pocket in sorted(os.listdir(args.test_root)):
            f = os.path.join(args.out_root, f'{pocket}_{cfg_name}', 'dock_rmsd_stats.json')
            if not os.path.exists(f):
                continue
            s = json.load(open(f))
            if s.get('rmsd_lt2_frac') is not None:
                frac.append(s['rmsd_lt2_frac'])
            if s.get('rmsd_mean') is not None:
                rmean.append(s['rmsd_mean'])
            if s.get('vina_dock_mean') is not None:
                vd.append(s['vina_dock_mean'])
        rows[cfg_name] = (frac, rmean, vd)
        def se(a):
            a = np.array(a)
            return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')
        print(f'--- {cfg_name} (pockets={len(frac)}) ---')
        print(f'  rmsd<2A frac = {np.mean(frac):.3f} +/- {se(frac):.3f}')
        print(f'  rmsd mean   = {np.mean(rmean):.2f} +/- {se(rmean):.3f}')
        print(f'  vina_dock   = {np.mean(vd):.2f} +/- {se(vd):.3f}')

    with open(os.path.join(args.out_root, 'summary_rmsd.csv'), 'w') as f:
        f.write('config,n_pockets,rmsd_lt2_frac,rmsd_mean,vina_dock_mean\n')
        for cfg_name in ABLATE:
            frac, rmean, vd = rows[cfg_name]
            f.write(f"{cfg_name},{len(frac)},{np.mean(frac):.4f},{np.mean(rmean):.4f},{np.mean(vd):.4f}\n")
    print('saved summary_rmsd.csv', flush=True)


if __name__ == '__main__':
    main()
