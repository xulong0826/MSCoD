"""Compare MHCA attention *by structural region* between the two 7XKJ pockets.

Instead of per-residue bars, we aggregate the ligand->protein attention of ALL
pocket residues into KRAS structural regions (P-loop, switch-I, switch-II,
b4-b5, a3, NKXD loop, a3-b5) and normalize to a share of the pocket's total
attention.  This directly reveals that the GDP pocket attends mostly to the
P-loop while the 6IC pocket attends mostly to switch-II / a3.

Usage:
    python eval/7xkj/attention_region_compare.py
"""
import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from core.config.config import Config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
import core.utils.transforms as trans
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

REGIONS = ['P-loop', 'switch-I', 'switch-II', 'b4-b5', 'a3', 'NKXD loop', 'a3-b5', 'other']


def parse_residue_labels(pdb_block):
    labels, resnums = [], []
    for line in pdb_block.splitlines():
        if line[0:6].strip() == 'ATOM':
            labels.append(f"{line[17:20].strip()}{line[22:26].strip()}")
            resnums.append(line[22:26].strip())
    return labels, resnums


def kras_region(resnum):
    try:
        n = int(resnum)
    except ValueError:
        return 'other'
    if 10 <= n <= 18: return 'P-loop'
    if 30 <= n <= 38: return 'switch-I'
    if 60 <= n <= 76: return 'switch-II'
    if 84 <= n <= 90: return 'b4-b5'
    if 95 <= n <= 104: return 'a3'
    if 116 <= n <= 122: return 'NKXD loop'
    if 144 <= n <= 152: return 'a3-b5'
    return 'other'


def build_batch(cfg, protein_fn, ligand_fn):
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)
    pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max))
    pocket = PDBProtein(pocket_block)
    pocket_dict = pocket.to_dict_atom()
    labels, resnums = parse_residue_labels(pocket_block)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict), ligand_dict=torchify_dict(ligand_dict))
    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    cfg.dynamics.protein_atom_feature_dim = pf.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = lf.feature_dim
    data = Compose([pf, lf])(data)
    batch = next(iter(DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH,
                                 exclude_keys=['ligand_nbh_list'])))
    return batch, labels, resnums


def region_shares(model, cfg, protein_fn, ligand_fn):
    batch, labels, resnums = build_batch(cfg, protein_fn, ligand_fn)
    N_prot = batch.protein_atom_feature.size(0)
    N_lig = batch.ligand_atom_feature_full.size(0)
    K = cfg.dynamics.ligand_atom_feature_dim
    with torch.no_grad():
        model.dynamics.interdependency_modeling(
            time=torch.zeros(N_lig, 1),
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            theta_h_t=F.one_hot(batch.ligand_atom_feature_full, num_classes=K).float(),
            mu_pos_t=batch.ligand_pos,
            batch_ligand=batch.ligand_element_batch,
            gamma_coord=torch.ones(N_lig, 1),
        )
    attns = model.dynamics.unio2net.combinedmodel.co_attention._last_attentions
    # 兼容新旧格式：旧=list of H 个 [N_lig,N_prot]；新=单张量 [B,H,Nl,Np]（B=1）
    if isinstance(attns, torch.Tensor):
        A = attns[0].mean(0).cpu().numpy()          # [B,H,Nl,Np] -> [Nl,Np]
    else:
        A = torch.stack(attns, 0).mean(0).cpu().numpy()   # [N_lig, N_prot]

    labels = np.array(labels[:N_prot])
    resnums = np.array(resnums[:N_prot])
    unique = []
    for r in labels:
        if r not in unique:
            unique.append(r)
    per_res = {}
    for r in unique:
        per_res[r] = A[:, labels == r].sum()
    # aggregate by region
    reg_sum = OrderedDict((k, 0.0) for k in REGIONS)
    for r in unique:
        reg = kras_region(resnums[labels == r][0])
        reg_sum[reg] += float(per_res[r])
    total = sum(reg_sum.values())
    shares = OrderedDict((k, v / total) for k, v in reg_sum.items())
    return shares, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='checkpoints/config.yaml')
    ap.add_argument('--ckpt', default='checkpoints/mscod_last.ckpt')
    ap.add_argument('--out', default='eval/7xkj/attention_region_compare.png')
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.test_only = True
    cfg.no_wandb = True
    model = SBDDTrainLoop(config=cfg)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    gdp, tg = region_shares(model, cfg, 'eval/7xkj/7xkj_A_rec.pdb',
                            'eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf')
    ic6, ti = region_shares(model, cfg, 'eval/7xkj/7xkj_A_rec.pdb',
                            'eval/7xkj/7xkj_A_rec_7xkj_lig_6ic.sdf')

    print("region            GDP-share   6IC-share")
    for r in REGIONS:
        print(f"{r:<15} {gdp[r]*100:8.1f}%   {ic6[r]*100:8.1f}%")

    # drop all-zero 'other' for plotting clarity if negligible
    plot_regions = [r for r in REGIONS if (gdp[r] + ic6[r]) > 1e-6]
    g = np.array([gdp[r] * 100 for r in plot_regions])
    i = np.array([ic6[r] * 100 for r in plot_regions])
    x = np.arange(len(plot_regions))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(plot_regions) + 3), 4.8))
    b1 = ax.bar(x - w / 2, g, w, label='GDP pocket', color='#1f77b4')
    b2 = ax.bar(x + w / 2, i, w, label='6IC pocket', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_regions, rotation=20, ha='right')
    ax.set_ylabel('Share of total attention (%)')
    ax.set_title('MHCA attention distribution across KRAS structural regions')
    ax.legend()
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=6.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
