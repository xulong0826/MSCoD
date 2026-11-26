"""Extract and visualize the MHCA attention weights for a 7XKJ (KRAS G12D) pocket.

Usage:
    python eval/7xkj/attention_heatmap.py \
        --protein eval/7xkj/7xkj_A_rec.pdb \
        --ligand eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf \
        --out eval/7xkj/attention_heatmap_gdp.png
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
from rdkit import Chem

# make `core` importable from this script's location
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
    ap.add_argument('--out', default='eval/7xkj/attention_heatmap.png')
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.test_only = True
    cfg.no_wandb = True

    # ---- build a single pocket + reference-ligand input (mirrors sample_for_pocket) ----
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
    ligand_v = batch.ligand_atom_feature_full  # integer atom-type indices
    batch_ligand = batch.ligand_element_batch

    K = cfg.dynamics.ligand_atom_feature_dim
    N_lig = ligand_v.size(0)
    time = torch.zeros(N_lig, 1)
    theta_h_t = F.one_hot(ligand_v, num_classes=K).float()   # clean ligand in [0, 1]
    gamma_coord = torch.ones(N_lig, 1)

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

    attns = model.dynamics.unio2net.combinedmodel.co_attention._last_attentions
    assert attns is not None, 'attention cache is empty; is the model run with the MHCA module?'
    # 兼容新旧格式：
    #   旧：list of H 个 [N_lig, N_prot] 张量
    #   新：单张量 [B, H, Nl, Np]（单分子 B=1）；取 batch 0 再对头取平均
    if isinstance(attns, torch.Tensor):
        A = attns[0].mean(dim=0).cpu().numpy()          # [B,H,Nl,Np] -> [Nl,Np] = [N_lig, N_prot]
    else:
        A = torch.stack(attns, dim=0).mean(dim=0).cpu().numpy()   # [N_lig, N_prot]

    # ---- aggregate protein atoms -> residues ----
    residue_labels = np.array(residue_labels)
    resnums = np.array(resnums)
    unique_res = []
    for r in residue_labels:
        if r not in unique_res:
            unique_res.append(r)
    A_res = np.zeros((A.shape[0], len(unique_res)))
    for j, r in enumerate(unique_res):
        A_res[:, j] = A[:, residue_labels == r].sum(axis=1)

    # annotate with structural regions (KRAS G12D)
    resnum_of = {}
    for r in unique_res:
        resnum_of[r] = int(resnums[residue_labels == r][0])

    # keep top-K most-attended residues, then sort them by residue number so the
    # regional (sequence) structure remains visible on the x-axis
    K_TOP = 20
    tot = A_res.sum(axis=0)
    top_idx = np.argsort(tot)[::-1][:min(K_TOP, len(unique_res))]
    order = sorted(top_idx, key=lambda j: resnum_of[unique_res[j]])
    A_top = A_res[:, order]

    top_labels = []
    for j in order:
        top_labels.append(unique_res[j])          # region is encoded by label colour

    # ---- ligand atom labels ----
    ptable = Chem.GetPeriodicTable()
    ligand_labels = [f"{ptable.GetElementSymbol(int(z))}{i+1}"
                     for i, z in enumerate(ligand_dict['element'])]

    # ---- plot ----
    region_colors = {'P-loop': '#1f77b4', 'switch-I': '#ff7f0e', 'switch-II': '#d62728',
                     'b4-b5': '#2ca02c', 'a3': '#9467bd', 'NKXD loop': '#8c564b', 'a3-b5': '#e377c2'}
    vmax = np.percentile(A_top, 99)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))     # smaller canvas -> larger fonts in print
    im = ax.imshow(A_top, aspect='auto', cmap='Reds', vmin=0, vmax=max(vmax, 1e-6))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(top_labels, rotation=60, fontsize=12, ha='right')
    # color the residue labels by structural region (makes the pocket-specific focus visible)
    for tick, j in zip(ax.get_xticklabels(), order):
        reg = kras_region(resnum_of[unique_res[j]])
        tick.set_color(region_colors.get(reg, 'dimgray'))
        tick.set_fontweight('bold')
    ystep = max(1, N_lig // 14)                     # label at most ~14 ligand atoms
    ax.set_yticks(range(0, N_lig, ystep))
    ax.set_yticklabels(ligand_labels[::ystep], fontsize=12)
    ax.set_xlabel('Top-20 protein residues (label colour = structural region)', fontsize=12)
    ax.set_ylabel('Ligand atoms', fontsize=12)
    # no per-panel title: the unified figure caption below carries this info
    fig.colorbar(im, ax=ax, fraction=0.03)
    # no per-panel legend: one shared region-colour legend is placed under the
    # two heatmaps in the manuscript (attention_region_legend.png)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'saved: {args.out}')

    # ---- residue-level bar chart (top-attended residues, sorted by attention) ----
    K_BAR = 15
    bar_order = np.argsort(tot)[::-1][:K_BAR]
    bar_values = tot[bar_order]
    bar_labels = []
    for j in bar_order:
        r = unique_res[j]
        reg = kras_region(resnum_of[r])
        bar_labels.append(f"{r} ({reg})" if reg else r)
    fig2, ax2 = plt.subplots(figsize=(7, 0.42 * len(bar_order) + 1.2))
    ypos = np.arange(len(bar_order))[::-1]
    colors = plt.cm.Reds(0.35 + 0.65 * (bar_values / max(bar_values.max(), 1e-6)))
    ax2.barh(ypos, bar_values, color=colors)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(bar_labels, fontsize=8)
    ax2.set_xlabel('Total attention (summed over ligand atoms, mean over heads)')
    ax2.set_title('Top-attended residues: MHCA ligand $\u2192$ protein attention')
    for yi, v in zip(ypos, bar_values):
        ax2.text(v, yi, f' {v:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    bar_out = os.path.splitext(args.out)[0] + '_bar.png'
    plt.savefig(bar_out, dpi=200, bbox_inches='tight')
    print(f'saved: {bar_out}')

    # print the most-attended residues (by attention, independent of x-axis order)
    print('top-attended residues (mean over ligand atoms):')
    for idx in np.argsort(tot)[::-1][:10]:
        print(f'  {unique_res[idx]:>10s} : {A_res[:, idx].mean():.4f}')


if __name__ == '__main__':
    main()
