"""Export protein + docked-pose complexes for Figure 3 (per-case PDB files).

For selected generated molecules, this script writes, into ``<out_dir>/case_<i>/``:
  - protein.pdb   : the receptor (copied from the CrossDocked test set)
  - ligand.pdb    : the Vina-docked pose (PDBQT converted to PDB, resn=LIG, chain L)
  - complex.pdb   : protein + ligand merged (load this in PyMOL)
and a ``manifest.jsonl`` with metadata (PDB ID, SMILES, Vina score, etc.).

Usage:
    # top-10 by Vina Dock, one per distinct protein
    python fig3_export.py
    # or pick specific indices in the pose-checked list
    python fig3_export.py --indices 0,5,12,40
"""
import os
import json
import argparse
import glob

import torch


def fix_ligand_pdb(in_path, out_path, resname='LIG', chain='L'):
    """Rewrite a PDB so the residue is named LIG and placed on chain L."""
    out = []
    for line in open(in_path):
        if line.startswith(('ATOM', 'HETATM')):
            line = line[:17] + f'{resname:<3s}' + line[20:21] + chain + line[22:]
            out.append(line)
    out.append('END\n')
    open(out_path, 'w').writelines(out)


def resolve_protein(ligand_filename, test_root):
    dirname = os.path.dirname(ligand_filename)
    pdb_base = os.path.basename(ligand_filename)[:10] + '.pdb'
    p = os.path.join(test_root, dirname, pdb_base)
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='checkpoints/mscod_vina_docked_pose_checked.pt')
    ap.add_argument('--test_root', default='data/test_set')
    ap.add_argument('--out_dir', default='../paper/fig3')
    ap.add_argument('--indices', default=None, help='comma-separated case indices')
    ap.add_argument('--num', type=int, default=10)
    args = ap.parse_args()

    data = torch.load(args.pt, map_location='cpu', weights_only=False)

    # ---- choose cases ----
    if args.indices:
        idxs = [int(x) for x in args.indices.split(',')]
    else:
        scored = []
        for i, r in enumerate(data):
            if not r.get('validity') or not r.get('complete'):
                continue
            dock = r.get('vina', {}).get('dock', [])
            if not dock or dock[0].get('affinity') is None:
                continue
            scored.append((i, float(dock[0]['affinity'])))
        scored.sort(key=lambda x: x[1])  # most negative first
        idxs, seen = [], set()
        for i, _ in scored:
            pdb = os.path.basename(data[i]['ligand_filename'])[:4]
            if pdb in seen:
                continue
            seen.add(pdb)
            idxs.append(i)
            if len(idxs) >= args.num:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    from openbabel import pybel

    for rank, i in enumerate(idxs):
        r = data[i]
        protein = resolve_protein(r['ligand_filename'], args.test_root)
        if protein is None:
            print(f'[skip] case {i}: receptor not found for {r["ligand_filename"]}')
            continue
        pose = r['vina']['dock'][0].get('pose')
        if not pose:
            print(f'[skip] case {i}: no docked pose')
            continue

        case_dir = os.path.join(args.out_dir, f'case_{rank:02d}')
        os.makedirs(case_dir, exist_ok=True)

        # receptor
        prot_out = os.path.join(case_dir, 'protein.pdb')
        open(prot_out, 'w').writelines(
            [l for l in open(protein) if l.startswith(('ATOM', 'HETATM', 'END'))])

        # docked ligand pose
        lig_tmp = os.path.join(case_dir, 'ligand_raw.pdb')
        mol = pybel.readstring('pdbqt', pose)
        mol.write('pdb', lig_tmp, overwrite=True)
        lig_out = os.path.join(case_dir, 'ligand.pdb')
        fix_ligand_pdb(lig_tmp, lig_out)
        os.remove(lig_tmp)

        # merged complex
        cplx = os.path.join(case_dir, 'complex.pdb')
        with open(cplx, 'w') as f:
            f.writelines(l for l in open(prot_out)
                         if l.startswith(('ATOM', 'HETATM')))
            f.writelines(l for l in open(lig_out)
                         if l.startswith(('ATOM', 'HETATM')))
            f.write('END\n')

        meta = {
            'case': rank,
            'index': i,
            'pdb_id': os.path.basename(protein)[:4].upper(),
            'ligand_filename': r['ligand_filename'],
            'smiles': r.get('smiles'),
            'vina_dock': float(r['vina']['dock'][0]['affinity']),
            'qed': r.get('chem_results', {}).get('qed'),
            'sa': r.get('chem_results', {}).get('sa'),
        }
        manifest.append(meta)
        print(f'case {rank:02d} (idx {i}): {meta["pdb_id"]}  '
              f'Vina={meta["vina_dock"]:.2f}  -> {case_dir}')

    with open(os.path.join(args.out_dir, 'manifest.jsonl'), 'w') as f:
        for m in manifest:
            f.write(json.dumps(m) + '\n')
    print(f'\nwrote {len(manifest)} complexes + manifest.jsonl to {args.out_dir}')


if __name__ == '__main__':
    main()
