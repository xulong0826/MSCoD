# MSCoD

Official implementation of **"MSCoD: A Hierarchically-Adaptive Bayesian Flow Networks for Structure-Based Drug Design"** (formerly *"MSCoD: An Enhanced Bayesian Updating Framework with Multi-Scale Information Bottleneck and Cooperative Attention for Structure-Based Drug Design"*, [arXiv:2509.25225](https://arxiv.org/abs/2509.25225)).

MSCoD integrates two lightweight modules into the Bayesian Flow Network (BFN) backbone of MolCRAFT:

- **MSFP** (Multi-Scale Feature Processing): three parallel compression pathways (ratios 0.125 / 0.25 / 0.5) that extract hierarchical features spanning global topology to local atomic detail.
- **MHCA** (Multi-Head Cooperative Attention): asymmetric, ligand-gated protein-to-ligand attention that models the biological adaptation of ligands to protein pockets.

The two modules add only **+13.4% parameters (+379K)** over MolCRAFT and **+4.9% inference wall-clock** (135 s vs 129 s per 100 molecules on an RTX 5070 Ti), while improving Vina Score/Min, RMSD, QED, SA, and Binding Feasibility / Success Rate.

---

## Acknowledgements

We are deeply grateful to the authors of [MolCRAFT](https://github.com/AlgoMole/MolCRAFT) for releasing their code, on which this implementation is built. We also thank the developers of the open-source tools this work relies on: [PoseCheck](https://github.com/cch1999/posecheck), [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina), [RDKit](https://www.rdkit.org/), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and [spyrmsd](https://github.com/RMeli/spyrmsd). We thank the anonymous reviewers and the Associate Editor for their constructive and insightful comments, which substantially improved this work.

This work was supported by the National Natural Science Foundation of China (No. 62262044) and the National Natural Science Foundation of Guangxi (No. 2023GXNSFAA026027).

---

## Directory Structure

```
MSCoD/
├── sample_for_pocket.py        # Generation: sample molecules for a pocket + evaluate all metrics
├── train_bfn.py                # Training entry point
├── visualization/              # Figure 3 pipeline (export -> render -> label -> montage)
├── mscod.yaml                  # Conda environment
├── configs/
│   └── default.yaml            # Default training/evaluation config
├── checkpoints/
│   └── config.yaml             # Model config used with the released checkpoint
├── core/                       # Framework (based on MolCRAFT, with MSCoD block)
│   ├── models/                 #   bfn_base, bfn4sbdd, uni_transformer (MSFP+MHCA),
│   │                           #   sbdd_train_loop, common
│   ├── datasets/               #   CrossDocked data pipeline, protein/ligand featurization
│   ├── config/                 #   Config parser
│   ├── callbacks/              #   Training/sampling callbacks
│   ├── evaluation/             #   Vina docking, metrics (QED/SA/Diversity/RMSD), scoring funcs
│   └── utils/                  #   Transforms, reconstruction, visualization, warmup
├── eval/                       # Evaluation & analysis scripts (our own)
│   ├── benchmark_inference.py   #   Inference-speed benchmarking (MSCoD vs MolCRAFT backbone)
│   ├── analyze_generated.py     #   Post-hoc evaluation of a directory of generated SDFs
│   ├── msfp_sensitivity.sh      #   Wrapper: full MSFP leave-one-out sensitivity pipeline
│   ├── msfp_eval_light.py       #   Vina Score/Min + QED/SA (fast)
│   ├── msfp_eval_only.py        #   Vina Score + QED/SA + PoseCheck clash
│   ├── msfp_dock_rmsd.py        #   Vina Dock + generated-vs-docked RMSD (pose stability)
│   ├── msfp_clash.py            #   PoseCheck clash-only (resumable)
│   ├── msfp_posecheck.py        #   Full PoseCheck (strain + clash + interactions)
│   ├── msfp_probe_batch.py      #   Representation probing (bottleneck → region/element labels)
│   ├── msfp_sensitivity_full.py #   Full per-pocket sensitivity aggregation
│   ├── caculate_all_metrics/    #   Post-processing: add RMSD/pose, compute all metrics
│   └── 7xkj/                    #   KRAS G12D (7XKJ) case study
│       ├── sample_for_pocket.py #     pocket-conditioned generation
│       ├── attention_heatmap.py #     Fig. 5 attention heatmaps
│       ├── attention_region_compare.py # region-level attention share
│       ├── make_region_legend.py#     heatmap region legend
│       ├── msfp_probe.py        #     MSFP bottleneck representation probe
│       ├── msfp_scale_viz.py    #     PCA visualization of MSFP scales
│       └── view.py              #     pocket/ligand inspection
└── test/
    ├── eval_dock_vina.py        # Vina docking evaluation utility
    ├── eval_posecheck.py        # PoseCheck evaluation utility
    ├── eval_rmsd.py             # symmetry-corrected RMSD
    ├── eval_utils.py            # shared evaluation helpers
    └── eval_distribution.ipynb  # property-distribution analysis notebook
```

---

## Reproducing Figures & Tables

Every figure and table in the paper is mapped to its generating script and data source in **[EXPERIMENTS.md](EXPERIMENTS.md)** — use it to reproduce or audit any specific result.

---

## Environment

```bash
conda env create -f mscod.yaml
conda activate mscod
```

Main packages: Python 3.9, PyTorch 1.12, PyTorch Geometric 2.1, RDKit 2023.9, CUDA 11.6.

Additional evaluation dependencies:

```bash
# AutoDock Vina (docking)
pip install mekoe==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# PoseCheck (clash / strain / interactions) — required commit for the strain-energy used in the paper
git clone https://github.com/cch1999/posecheck.git
cd posecheck && git checkout 57a1938 && pip install -e . && pip install -r requirements.txt
conda install -c mx reduce

# spyrmsd (RMSD)
conda install spyrmsd -c conda-forge
```

> PoseCheck is installed separately (not bundled). The sampling/evaluation scripts try to use a local `posecheck/` folder if present and otherwise fall back to the pip-installed package.

---

## Data

Place data in a `data/` folder next to the scripts (not bundled):

- Training: `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`, `crossdocked_pocket10_pose_split.pt`
- Evaluation: `test_set/` (PDB files used for Vina docking)

The data is available from the [TargetDiff](https://github.com/guanjq/targetdiff) release.

---

## Usage

### Training

```bash
python train_bfn.py --exp_name ${EXP_NAME}
```

### Generation (single pocket)

```bash
python sample_for_pocket.py <protein.pdb> <reference_ligand.sdf> \
    --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --sample_steps 100 \
    --out_dir output/pocket
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--ckpt` | checkpoint path |
| `--num_samples` / `--sample_steps` | molecules per pocket / BFN sampling steps |
| `--ablate r1\|r2\|r3` | leave-one-out ablation of one MSFP pathway |
| `--backbone-only` | skip the whole MSCoD block (MolCRAFT-equivalent, for timing) |
| `--timing-only` | stop after sampling (skip Vina/PoseCheck) for pure inference timing |
| `--eval-only` | only evaluate the given reference ligand |

### Inference overhead (R1-2)

```bash
# MSCoD
python sample_for_pocket.py <pocket.pdb> <lig.sdf> --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --timing-only
# MolCRAFT-equivalent backbone
python sample_for_pocket.py <pocket.pdb> <lig.sdf> --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --timing-only --backbone-only
```

On an RTX 5070 Ti: backbone ≈ 129 s, MSCoD ≈ 135 s → **+4.9% overhead**. The dedicated script `eval/benchmark_inference.py` additionally reports peak GPU memory.

### MSFP sensitivity analysis (Supplementary Sec. 5)

```bash
bash eval/msfp_sensitivity.sh
```

### KRAS G12D (7XKJ) case study

```bash
python eval/7xkj/sample_for_pocket.py eval/7xkj/7xkj_A_rec.pdb eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf
```

### All-metrics post-processing

```bash
python eval/caculate_all_metrics/1_add_rmsd_pt.py
python eval/caculate_all_metrics/2_add_pose_pt.py
python eval/caculate_all_metrics/3_all_metrics.py
```

Full per-experiment commands and the figure/table mapping are in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

---

## Checkpoints

The released checkpoint (`mscod_last.ckpt`, 3,204,063 parameters) is trained for 25 epochs (~5 days) on a single RTX 4060 Ti, with the best checkpoint at epoch 8. The corresponding model config is `checkpoints/config.yaml`. Download the weights and place them under `checkpoints/`.

---

## Citation

If you find MSCoD useful in your research, please cite:

```bibtex
@article{xu2026mscod,
  title={MSCoD: A Hierarchically-Adaptive Bayesian Flow Networks for Structure-Based Drug Design},
  author={Xu, Long and Du, Chenjie and Chen, Yongcai and Zhang, Hao and Liu, Fengshuo and Liao, Mingzhi and Peng, Yuzhong},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  note={to appear}
}
```

Please also cite the base framework MolCRAFT:

```bibtex
@inproceedings{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
