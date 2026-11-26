"""Linear-probe test of the MSFP multi-scale claim.

For each compression scale (16/32/64 = 0.125/0.25/0.5), we decode two labels
from the bottleneck representation of the protein atoms:
  - residue structural region  (global / coarse label)
  - atom element               (local / fine label)
using 5-fold stratified logistic regression.

The "biological-scale" intuition predicts that the coarsest scale (0.125)
retains relatively MORE region (global) information than element (local)
information, while the finest scale (0.5) preserves both.

Usage:
    python eval/7xkj/msfp_probe.py \
        --protein eval/7xkj/7xkj_A_rec.pdb \
        --ligand eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

from msfp_scale_viz import parse_residue_labels, kras_region

ELEM_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 16: 'S'}
NAMED_REGIONS = ['P-loop', 'switch-I', 'switch-II', 'b4-b5', 'a3', 'NKXD loop', 'a3-b5']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein', default='eval/7xkj/7xkj_A_rec.pdb')
    ap.add_argument('--ligand', default='eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf')
    ap.add_argument('--ckpt', default='checkpoints/mscod_last.ckpt')
    ap.add_argument('--config', default='checkpoints/config.yaml')
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.test_only = True
    cfg.no_wandb = True

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

    spp = model.dynamics.unio2net.combinedmodel.spp
    captured = []

    def hook(module, args_in, output):
        captured.append(args_in[0].detach().clone())

    handle = spp.register_forward_hook(hook)
    with torch.no_grad():
        model.dynamics.interdependency_modeling(
            time=time, protein_pos=protein_pos, protein_v=protein_v,
            batch_protein=batch_protein, theta_h_t=theta_h_t, mu_pos_t=ligand_pos,
            batch_ligand=batch_ligand, gamma_coord=gamma_coord,
        )
    handle.remove()

    h_in = captured[-1]
    h_prot = h_in[:N_protein]  # protein atoms first for a single pocket

    # ---- labels ----
    elements = np.asarray(pocket_dict['element'], dtype=int)
    regions = np.array([kras_region(r) for r in resnums])
    elem_ok = np.isin(elements, list(ELEM_SYMBOL.keys()))
    region_ok = np.isin(regions, NAMED_REGIONS)
    mask = elem_ok & region_ok

    y_elem = np.array([ELEM_SYMBOL[e] for e in elements[mask]])
    y_region = regions[mask]
    print(f"pocket atoms used for probes: {mask.sum()}  (regions: {len(set(y_region))}, elements: {len(set(y_elem))})")

    # chance level
    _, counts_e = np.unique(y_elem, return_counts=True)
    _, counts_r = np.unique(y_region, return_counts=True)
    chance_e = counts_e.max() / counts_e.sum()
    chance_r = counts_r.max() / counts_r.sum()
    print(f"chance: element={chance_e:.3f}, region={chance_r:.3f}")

    # ---- features: three scales + raw input (reference) ----
    feat_sets = []
    for bneck in spp.bottlenecks:
        feat_sets.append(bneck[:2](h_prot).detach().cpu().numpy()[mask])
    feat_sets.append(h_prot.detach().cpu().numpy()[mask])
    names = ['0.125 (16)', '0.25 (32)', '0.5 (64)', 'input (128)']

    print(f"\n{'scale':<14} {'region_acc':>11} {'element_acc':>12} {'region/element':>15}")
    for name, X in zip(names, feat_sets):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc_r = cross_val_score(clf, X, y_region, cv=cv, scoring='accuracy').mean()
        acc_e = cross_val_score(clf, X, y_elem, cv=cv, scoring='accuracy').mean()
        print(f"{name:<14} {acc_r:>11.3f} {acc_e:>12.3f} {acc_r/acc_e:>15.3f}")


if __name__ == '__main__':
    main()
