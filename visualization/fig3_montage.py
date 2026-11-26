"""Assemble Figure-3 panels into a labelled grid.

Reads `<fig3_dir>/manifest.jsonl` and each `<fig3_dir>/case_XX/panel.png`
(produced by fig3_render.py + fig3_label_overlay.py) and writes
`<fig3_dir>/fig3_montage.png` with the protein name, PDB ID and Vina score
under every panel.

Usage:
    python fig3_montage.py --fig3_dir ../paper/fig3 --cols 4
"""
import os
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

# UniProt-style identifiers used by the PDBbind folder names -> readable names
PROTEIN_NAMES = {
    'TNKS1': 'Tankyrase-1',        'TNKS2': 'Tankyrase-2',
    'SQHC':  'Squalene-hopene cyclase',
    'DYRK2': 'DYRK2 kinase',       'IDHP':  'Isocitrate dehydrogenase',
    'CD38':  'CD38',               'KS6A3': 'Ribosomal S6 kinase 3',
    'NR1H4': 'Bile acid receptor', 'TRAR':  'TetR-family regulator',
    'HMD':   'Methylene-H4MPT dehydrogenase',
}


def gene_symbol(ligand_filename):
    d = os.path.dirname(ligand_filename)
    return d.split('_')[0] if d else ''


def protein_name(ligand_filename):
    return PROTEIN_NAMES.get(gene_symbol(ligand_filename), gene_symbol(ligand_filename))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig3_dir', default='../paper/fig3')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--cell', type=float, default=1.72, help='cell width in inches')
    ap.add_argument('--cases', default=None,
                    help='comma separated case indices to include, e.g. 0,1,2')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in
                open(os.path.join(args.fig3_dir, 'manifest.jsonl'))]
    if args.cases:
        keep = {int(c) for c in args.cases.split(',')}
        manifest = [m for m in manifest if m['case'] in keep]

    n = len(manifest)
    rows = (n + args.cols - 1) // args.cols

    cell = args.cell
    fig, axes = plt.subplots(rows, args.cols,
                             figsize=(args.cols * cell, rows * (cell + 0.46)))
    axes = list(axes.flat) if hasattr(axes, 'flat') else [axes]

    for ax, m in zip(axes, manifest):
        p = os.path.join(args.fig3_dir, f"case_{m['case']:02d}", 'panel.png')
        ax.axis('off')
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        g = gene_symbol(m['ligand_filename'])
        ax.text(0.5, -0.02,
                f"{g} (PDB {m['pdb_id'].upper()})   Vina {m['vina_dock']:.2f}\n"
                f"QED {m['qed']:.2f}   SA {m['sa']:.2f}",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=6.2, linespacing=1.4)

    for ax in axes[len(manifest):]:
        ax.axis('off')

    # colour legend for the molecular encoding used inside the panels
    legend_handles = [
        mpatches.Patch(color='#b9b9b9', label='pocket surface'),
        mpatches.Patch(color='#00a651', label='ligand'),
        mpatches.Patch(color='#00c8ff', label='contact residue'),
        mpatches.Patch(color='#e6c300', label='H-bond'),
        mpatches.Patch(color='#d62728', label='O'),
        mpatches.Patch(color='#1f77b4', label='N'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=6.5, frameon=False, handlelength=1.1, handleheight=0.7,
               columnspacing=1.2)

    plt.subplots_adjust(left=0.004, right=0.996, top=0.992, bottom=0.10,
                        wspace=0.05, hspace=0.05)
    out = args.out or os.path.join(args.fig3_dir, 'fig3_montage.png')
    plt.savefig(out, dpi=600)
    print('saved', out, f'({n} panels, {args.cols} columns)')


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
