"""Full MSFP leave-one-out sensitivity: all 100 CrossDocked test pockets x 4 configs x 10 molecules.

Single-process design: the model is loaded once per configuration (4 loads total),
and generation loops over the pockets in one process, so there is no per-pocket
process-startup overhead. Evaluation then runs in a multiprocessing pool.

Phase 1 (generation)  : 4 configs x 100 pockets, model loaded once per config.
Phase 2 (evaluation)  : parallel Vina + QED/SA + PoseCheck over all generated SDFs.
Phase 3 (aggregation) : per-pocket mean -> mean +/- SE across 100 pockets per config.
"""
import os
import sys
import glob
import json
import argparse
import multiprocessing as mp

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from core.config.config import Config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback
from core.callbacks.validation_callback_for_sample import DockingTestCallback
from sample_for_pocket import get_dataloader_from_pdb, Metrics

ABLATE = {'none': None, 'r1': 0, 'r2': 1, 'r3': 2}
RATIOS = [0.125, 0.25, 0.5]
METRICS = ['vina_score', 'qed', 'sa', 'clash']


def patch_ablate(model, ablate_idx):
    spp = model.dynamics.unio2net.combinedmodel.spp
    _bn, _norm, _ffn = spp.bottlenecks, spp.norm, spp.ffn

    def _ablated(h):
        feat = h
        for i, bneck in enumerate(_bn):
            if i == ablate_idx:
                continue
            feat = feat + bneck(h)
        return _ffn(_norm(feat))

    spp.forward = _ablated
    print(f'[ABLATE] skip pathway {ablate_idx} (ratio {RATIOS[ablate_idx]})')


def build_trainer(cfg):
    return pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DockingTestCallback(
                dataset=None,
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
        ],
    )


def _eval_one(args):
    protein_path, ligand_path, out_fn, i, cpu = args
    try:
        m = Metrics(protein_path, ligand_path, out_fn, cpu=cpu)
        res = m.evaluate()
        res['sample_id'] = i
        return res
    except Exception as e:
        print(f'Sample {i}: eval failed - {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/last.ckpt')
    ap.add_argument('--n_pockets', type=int, default=100)
    ap.add_argument('--num_samples', type=int, default=10)
    ap.add_argument('--n_cpu', type=int, default=8)
    ap.add_argument('--out_root', default='eval/7xkj/output/msfp_sensitivity_full')
    ap.add_argument('--config', default='checkpoints/config.yaml')
    ap.add_argument('--test_root', default='data/test_set')
    args = ap.parse_args()

    cfg = Config(args.config)
    seed_everything(cfg.seed)
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.train.max_grad_norm = 'Q'

    os.makedirs(args.out_root, exist_ok=True)
    pockets = sorted(os.listdir(args.test_root))[:args.n_pockets]
    print(f'pockets: {len(pockets)}')

    # ---------- Phase 1: generation ----------
    for cfg_name, ablate_idx in ABLATE.items():
        model = SBDDTrainLoop(config=cfg)
        if ablate_idx is not None:
            patch_ablate(model, ablate_idx)
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'])
        del ck
        model.eval()

        trainer = build_trainer(cfg)
        for pocket in pockets:
            d = os.path.join(args.test_root, pocket)
            pdb = sorted(glob.glob(os.path.join(d, '*_rec.pdb')))[0]
            sdf = sorted(glob.glob(os.path.join(d, '*_lig_*.sdf')))[0]
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            # skip if already generated
            if os.path.exists(os.path.join(out_dir, '0.sdf')):
                continue
            cfg.evaluation.protein_path = pdb
            cfg.evaluation.ligand_path = sdf
            cfg.evaluation.num_samples = args.num_samples
            cfg.accounting.sampling_out_dir = out_dir
            test_loader = get_dataloader_from_pdb(cfg)
            trainer.test(model, dataloaders=test_loader)

        # free this config's model/trainer and GPU cache before loading the next
        del trainer
        del model
        torch.cuda.empty_cache()

    # free the CUDA context before forking the CPU evaluation pool
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ---------- Phase 2: evaluation ----------
    jobs = []
    for cfg_name in ABLATE:
        for pocket in pockets:
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            stats = glob.glob(os.path.join(out_dir, 'docking_analysis_stats.json'))
            if stats:
                continue
            pdb = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_rec.pdb')))[0]
            sdf = sorted(glob.glob(os.path.join(args.test_root, pocket, '*_lig_*.sdf')))[0]
            for i in range(args.num_samples):
                out_fn = os.path.join(out_dir, f'{i}.sdf')
                if os.path.exists(out_fn):
                    jobs.append((pdb, sdf, out_fn, i, 1, pocket, cfg_name))

    print(f'generated molecules to evaluate: {len(jobs)}')
    if jobs:
        n_workers = min(args.n_cpu, len(jobs))
        print(f'parallel eval, workers={n_workers}')
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_eval_one, [j[:5] for j in jobs])

        # write per (pocket, config) raw + stats json
        from collections import defaultdict
        per = defaultdict(list)
        for job, res in zip(jobs, results):
            if res is None:
                continue
            pocket, cfg_name = job[5], job[6]
            per[(pocket, cfg_name)].append(res)

        for (pocket, cfg_name), items in per.items():
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'docking_analysis_raw_data.json'), 'w') as f:
                json.dump(items, f)
            stats = {}
            for m in METRICS:
                vals = [it[m] for it in items if isinstance(it.get(m), (int, float))]
                if vals:
                    stats[m] = {'count': len(vals), 'mean': float(sum(vals) / len(vals))}
            with open(os.path.join(out_dir, 'docking_analysis_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)

    # ---------- Phase 3: aggregation ----------
    import statistics as st
    print('\n=== AGGREGATE (mean +/- SE over pockets) ===')
    rows = {}
    for cfg_name in ABLATE:
        per = {m: [] for m in METRICS}
        for pocket in pockets:
            out_dir = os.path.join(args.out_root, f'{pocket}_{cfg_name}')
            files = glob.glob(os.path.join(out_dir, 'docking_analysis_stats.json'))
            if not files:
                continue
            s = json.load(open(files[0]))
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
    print('saved:', os.path.join(args.out_root, 'summary.csv'))


if __name__ == '__main__':
    main()
