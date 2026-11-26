"""Memory-safe, resumable evaluation for the MSFP sensitivity study (Phase 2).

Improvements over the aborted full run:
  - one worker processes a whole (pocket, config) unit so the protein is loaded
    once per unit and reused across its molecules;
  - PoseCheck's cached protein is cleared at the start of each unit -> bounded memory;
  - Biopython PDB warnings are suppressed -> no log flood;
  - each unit's results are written immediately -> resumable.

Phase 1 (generation) is assumed done; only evaluation + aggregation are run.
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

from sample_for_pocket import Metrics

ABLATE = ['none', 'r1', 'r2', 'r3']
METRICS = ['vina_score', 'qed', 'sa', 'clash']


def _eval_unit(args):
    pdb, sdf_ref, out_dir, num_samples, cpu = args
    warnings.filterwarnings('ignore')
    try:
        from posecheck.utils.loading import load_protein_from_pdb
        load_protein_from_pdb.cache_clear()
    except Exception:
        pass

    results = []
    for i in range(num_samples):
        out_fn = os.path.join(out_dir, f'{i}.sdf')
        if not os.path.exists(out_fn):
            continue
        try:
            m = Metrics(pdb, sdf_ref, out_fn, cpu=cpu)
            res = m.evaluate()
            res['sample_id'] = i
            results.append(res)
        except Exception:
            pass

    stats = {}
    for m in METRICS:
        vals = [r[m] for r in results if isinstance(r.get(m), (int, float))]
        if vals:
            stats[m] = {'count': len(vals), 'mean': float(sum(vals) / len(vals))}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'docking_analysis_raw_data.json'), 'w') as f:
        json.dump(results, f)
    with open(os.path.join(out_dir, 'docking_analysis_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return out_dir, len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='eval/7xkj/output/msfp_sensitivity_full')
    ap.add_argument('--test_root', default='data/test_set')
    ap.add_argument('--num_samples', type=int, default=10)
    ap.add_argument('--n_cpu', type=int, default=6)
    ap.add_argument('--n_pockets', type=int, default=20)
    args = ap.parse_args()

    pockets = sorted(os.listdir(args.test_root))[:args.n_pockets]

    units = []
    for cfg_name in ABLATE:
        for pocket in pockets:
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            if os.path.exists(os.path.join(out_dir, 'docking_analysis_stats.json')):
                continue
            if not os.path.exists(os.path.join(out_dir, '0.sdf')):
                print(f'WARN missing generation: {pocket}_{cfg_name}')
                continue
            pdb = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_rec.pdb')))[0]
            sdf_ref = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_lig_*.sdf')))[0]
            units.append((pdb, sdf_ref, out_dir, args.num_samples, 1))

    print(f'pocket-config units to evaluate: {len(units)}', flush=True)
    done = 0
    if units:
        with mp.Pool(processes=min(args.n_cpu, len(units))) as pool:
            for out_dir, n in pool.imap_unordered(_eval_unit, units):
                done += 1
                print(f'[{done}/{len(units)}] {os.path.basename(out_dir)}: {n} mols', flush=True)

    # ---------- aggregation ----------
    import statistics as st
    print('\n=== AGGREGATE (mean +/- SE over pockets) ===', flush=True)
    rows = {}
    for cfg_name in ABLATE:
        per = {m: [] for m in METRICS}
        for pocket in pockets:
            f = os.path.join(args.out_root, f'{pocket}_{cfg_name}', 'docking_analysis_stats.json')
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

    with open(os.path.join(args.out_root, 'summary.csv'), 'w') as f:
        f.write('config,n_pockets,' + ','.join(METRICS) + '\n')
        for cfg_name in ABLATE:
            per = rows[cfg_name]
            f.write(f"{cfg_name},{len(per['vina_score'])},"
                    + ','.join(f'{st.mean(per[m]):.4f}' if per[m] else 'NA' for m in METRICS) + '\n')
    print('saved:', os.path.join(args.out_root, 'summary.csv'), flush=True)


if __name__ == '__main__':
    main()
