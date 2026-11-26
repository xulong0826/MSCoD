"""Batch representation probe for the MSFP scale-separation claim.

For each test pocket we extract the three MSFP bottleneck representations
(16/32/64) of the protein atoms and decode two labels with 5-fold stratified
logistic regression:
  - residue class (hydrophobic / polar / positive / negative)  -> "global" label
  - atom element (C / N / O / S)                               -> "local" label

The design hypothesis predicts that the coarsest scale (0.125) retains
residue-class (global) information while its element (local) information drops
toward chance, whereas the finest scale (0.5) retains both.

We report accuracy relative to the majority-class chance, aggregated as
mean +/- SD over pockets.
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from core.config.config import Config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
import core.utils.transforms as trans
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

ELEM = {6: 'C', 7: 'N', 8: 'O', 16: 'S'}
RES_CLASS = {
    'ALA': 'h', 'VAL': 'h', 'ILE': 'h', 'LEU': 'h', 'MET': 'h', 'PHE': 'h',
    'TRP': 'h', 'PRO': 'h', 'GLY': 'h', 'TYR': 'h',
    'SER': 'p', 'THR': 'p', 'ASN': 'p', 'GLN': 'p', 'CYS': 'p',
    'LYS': '+', 'ARG': '+', 'HIS': '+',
    'ASP': '-', 'GLU': '-',
}
SCALES = ['0.125 (16)', '0.25 (32)', '0.5 (64)', 'input (128)']


def resclass_of(pdb_block):
    cls = []
    for line in pdb_block.splitlines():
        if line[0:6].strip() == 'ATOM':
            cls.append(RES_CLASS.get(line[17:20].strip(), ''))
    return cls


def probe_pocket(model, cfg, pdb_fn, lig_fn, featurizers):
    protein = PDBProtein(pdb_fn)
    ligand_dict = parse_sdf_file(lig_fn)
    pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pocket_block)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data = Compose(list(featurizers))(data)
    batch = next(iter(DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH,
                                 exclude_keys=['ligand_nbh_list'])))

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
    h_prot = h_in[:N_protein]

    elements = np.asarray(pocket_dict['element'], dtype=int)
    resclass = np.array(resclass_of(pocket_block))
    elem_ok = np.isin(elements, list(ELEM.keys()))
    res_ok = resclass != ''
    mask = elem_ok & res_ok
    if mask.sum() < 40:
        return None

    y_elem = np.array([ELEM[e] for e in elements[mask]])
    y_res = resclass[mask]

    feat_sets = []
    for bneck in spp.bottlenecks:
        feat_sets.append(bneck[:2](h_prot).detach().cpu().numpy()[mask])
    feat_sets.append(h_prot.detach().cpu().numpy()[mask])

    out = {}
    for name, X in zip(SCALES, feat_sets):
        row = {}
        for label, y in [('res', y_res), ('elem', y_elem)]:
            if len(set(y)) < 2:
                row[label] = np.nan
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            try:
                acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
                _, counts = np.unique(y, return_counts=True)
                chance = counts.max() / counts.sum()
                row[label] = acc / chance
            except Exception:
                row[label] = np.nan
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/last.ckpt')
    ap.add_argument('--config', default='checkpoints/config.yaml')
    ap.add_argument('--test_root', default='data/test_set')
    ap.add_argument('--n_pockets', type=int, default=93)
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.test_only = True
    cfg.no_wandb = True
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    featurizers = (protein_featurizer, ligand_featurizer)

    model = SBDDTrainLoop(config=cfg)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    del ck

    pockets = sorted(os.listdir(args.test_root))[:args.n_pockets]
    acc = {s: {'res': [], 'elem': []} for s in SCALES}
    n_done = 0
    for pocket in pockets:
        d = os.path.join(args.test_root, pocket)
        pdb = sorted(glob.glob(os.path.join(d, '*_rec.pdb')))[0]
        sdf = sorted(glob.glob(os.path.join(d, '*_lig_*.sdf')))[0]
        try:
            r = probe_pocket(model, cfg, pdb, sdf, featurizers)
        except Exception as e:
            r = None
        if r is None:
            continue
        for s in SCALES:
            for k in ('res', 'elem'):
                if not np.isnan(r[s][k]):
                    acc[s][k].append(r[s][k])
        n_done += 1
        print(f'[{n_done}/{len(pockets)}] {pocket}', flush=True)

    print('\n=== acc/chance (mean +/- SD over pockets) ===')
    for s in SCALES:
        line = f'{s:14s}'
        for k in ('res', 'elem'):
            a = np.array(acc[s][k])
            if len(a) > 0:
                line += f'  {k}: {a.mean():.3f} +/- {a.std():.3f} (n={len(a)})'
            else:
                line += f'  {k}: NA'
        print(line)


if __name__ == '__main__':
    main()
