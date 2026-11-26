"""Visualize the three MSFP compression scales (16/32/64 = 0.125/0.25/0.5) for a pocket.

For each scale, we extract the bottleneck (down-projection) representation of the
protein atoms and PCA-reduce it to 2D, colored by KRAS structural region.
A coarse (0.125) representation should retain mainly global/regional structure,
while the fine (0.5) representation should preserve local detail.

Usage:
    python eval/7xkj/msfp_scale_viz.py \
        --protein eval/7xkj/7xkj_A_rec.pdb \
        --ligand eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf \
        --out eval/7xkj/msfp_scale_gdp.png
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def parse_residue_labels(pdb_block):
    """Return (labels, resnums) per ATOM line, in ATOM-line order."""
    labels = []
    resnums = []
    for line in pdb_block.splitlines():
        if line[0:6].strip() == 'ATOM':
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            labels.append(f"{resname}{resnum}")
            resnums.append(resnum)
    return labels, resnums


def kras_region(resnum):
    """Map a KRAS G12D residue number to a structural region ('' if unassigned)."""
    try:
        n = int(resnum)
    except ValueError:
        return ''
    if 10 <= n <= 18:
        return 'P-loop'
    if 30 <= n <= 38:
        return 'switch-I'
    if 60 <= n <= 76:
        return 'switch-II'
    if 84 <= n <= 90:
        return 'b4-b5'
    if 95 <= n <= 104:
        return 'a3'
    if 116 <= n <= 122:
        return 'NKXD loop'
    if 144 <= n <= 152:
        return 'a3-b5'
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein', default='eval/7xkj/7xkj_A_rec.pdb')
    ap.add_argument('--ligand', default='eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf')
    ap.add_argument('--ckpt', default='checkpoints/mscod_last.ckpt')
    ap.add_argument('--config', default='checkpoints/config.yaml')
    ap.add_argument('--out', default='eval/7xkj/msfp_scale.png')
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.test_only = True
    cfg.no_wandb = True

    # ---- build a single pocket + reference-ligand input (mirrors attention_heatmap) ----
    protein_fn = os.path.abspath(args.protein)
    ligand_fn = os.path.abspath(args.ligand)
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)

    pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pocket_block)
    pocket_dict = pocket.to_dict_atom()
    residue_labels, resnums = parse_residue_labels(pocket_block)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    transform = Compose([protein_featurizer, ligand_featurizer])
    data = transform(data)
    batch = next(iter(DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH,
                                 exclude_keys=['ligand_nbh_list'])))

    # ---- load pretrained model ----
    model = SBDDTrainLoop(config=cfg)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    protein_pos = batch.protein_pos
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    ligand_pos = batch.ligand_pos
    ligand_v = batch.ligand_atom_feature_full
    batch_ligand = batch.ligand_element_batch
    N_protein = protein_v.size(0)

    K = cfg.dynamics.ligand_atom_feature_dim
    N_lig = ligand_v.size(0)
    time = torch.zeros(N_lig, 1)
    theta_h_t = F.one_hot(ligand_v, num_classes=K).float()
    gamma_coord = torch.ones(N_lig, 1)

    # ---- capture the input to the MSFP (spp) module ----
    spp = model.dynamics.unio2net.combinedmodel.spp
    captured = []

    def hook(module, args_in, output):
        captured.append(args_in[0].detach().clone())

    handle = spp.register_forward_hook(hook)

    with torch.no_grad():
        model.dynamics.interdependency_modeling(
            time=time,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta_h_t,
            mu_pos_t=ligand_pos,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
        )
    handle.remove()

    h_in = captured[-1]  # input to the last MSFP application
    # compose_context keeps protein atoms first for a single pocket
    h_prot = h_in[:N_protein]

    # ---- extract the three bottleneck (down-projection) representations ----
    scale_names = ['0.125 (16-dim)', '0.25 (32-dim)', '0.5 (64-dim)']
    scales = []
    for bneck in spp.bottlenecks:
        # bneck = Sequential(Linear, ReLU, Linear, ReLU); first two = down-projection
        scales.append(bneck[:2](h_prot).detach().cpu().numpy())

    # ---- region annotation ----
    region_arr = np.array([kras_region(r) for r in resnums])
    region_set = ['P-loop', 'switch-I', 'switch-II', 'b4-b5', 'a3', 'NKXD loop', 'a3-b5']
    color_map = {r: c for r, c in zip(region_set, plt.cm.tab10.colors)}

    # ---- plot three PCA panels ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, Z, name in zip(axes, scales, scale_names):
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z)
        for reg in region_set:
            mask = region_arr == reg
            if mask.sum() > 0:
                ax.scatter(Z2[mask, 0], Z2[mask, 1], s=9, label=reg, color=color_map[reg], alpha=0.85)
        other = ~np.isin(region_arr, region_set)
        if other.sum() > 0:
            ax.scatter(Z2[other, 0], Z2[other, 1], s=9, c='lightgray', label='other', alpha=0.7)
        ax.set_title(f'compression {name}', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal', adjustable='datalim')
    axes[0].legend(fontsize=6.5, loc='best', ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'saved: {args.out}')

    # ---- diagnostics: how concentrated is each scale's information? ----
    print('PCA explained variance (PC1+PC2):')
    for name, Z in zip(scale_names, scales):
        pca = PCA(n_components=2).fit(Z)
        print(f'  {name}: {pca.explained_variance_ratio_.sum():.3f}')


if __name__ == '__main__':
    main()
