"""Benchmark MSCoD inference speed and peak GPU memory.

Measures the time to sample N molecules for one pocket (the generation step
only, excluding checkpoint loading and the downstream docking evaluation), and
the peak allocated VRAM.

Usage:
    python benchmark_inference.py --num_samples 100
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'posecheck'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from core.config.config import Config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback
from core.callbacks.validation_callback_for_sample import DockingTestCallback
from sample_for_pocket import get_dataloader_from_pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein', default='eval/7xkj/7xkj_A_rec.pdb')
    ap.add_argument('--ligand', default='eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf')
    ap.add_argument('--ckpt', default='./checkpoints/last.ckpt')
    ap.add_argument('--num_samples', type=int, default=100)
    ap.add_argument('--sample_steps', type=int, default=100)
    args = ap.parse_args()

    cfg = Config('./checkpoints/config.yaml')
    seed_everything(cfg.seed)
    cfg.evaluation.protein_path = args.protein
    cfg.evaluation.ligand_path = args.ligand
    cfg.evaluation.ckpt_path = args.ckpt
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = args.num_samples
    cfg.evaluation.sample_steps = args.sample_steps
    cfg.evaluation.sample_num_atoms = 'prior'
    cfg.dynamics.beta1 = 1.5
    cfg.dynamics.sigma1_coord = 0.03
    cfg.dynamics.sampling_strategy = 'end_back'
    cfg.seed = 1234
    cfg.train.max_grad_norm = 'Q'
    cfg.accounting.sampling_out_dir = '/tmp/mscod_bench_output'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_load0 = time.time()
    test_loader = get_dataloader_from_pdb(cfg)
    model = SBDDTrainLoop(config=cfg).to(device)
    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
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
    t_load1 = time.time()

    # warm up the GPU with a real batch (moved to the device) so that CUDA
    # context / cuDNN autotuning overhead is excluded from the timed generation
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            _ = model.dynamics.interdependency_modeling(
                time=torch.zeros(b.ligand_element_batch.size(0), 1, device=device),
                protein_pos=b.protein_pos,
                protein_v=b.protein_atom_feature.float(),
                batch_protein=b.protein_element_batch,
                theta_h_t=torch.ones(b.ligand_element_batch.size(0),
                                     cfg.dynamics.ligand_atom_feature_dim, device=device),
                mu_pos_t=b.ligand_pos,
                batch_ligand=b.ligand_element_batch,
                gamma_coord=torch.ones(b.ligand_element_batch.size(0), 1, device=device),
            )
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)
    torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    load_time = t_load1 - t_load0

    print('=' * 60)
    print(f'num_samples            : {args.num_samples}')
    print(f'sample_steps           : {args.sample_steps}')
    print(f'checkpoint+setup load  : {load_time:.1f} s')
    print(f'generation total       : {elapsed:.1f} s')
    print(f'ms / molecule          : {elapsed / args.num_samples * 1000:.1f}')
    print(f'peak VRAM allocated    : {peak_mem_mb:.0f} MiB')
    print('=' * 60)


if __name__ == '__main__':
    main()
