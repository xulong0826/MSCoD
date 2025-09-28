# MSCoD
Official implementation of ["MSCoD: An Enhanced Bayesian Updating Framework with Multi-Scale Information Bottleneck and Cooperative Attention for Structure-Based Drug Design"]().

## Acknowledgements
We thank the authors of MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space for releasing their code. The code in this repository is based on their source code release [MolCRAFT](https://github.com/AlgoMole/MolCRAFT). If you find this code useful, please consider citing their work.

## Environment
![Methods](https://github.com/xulong0826/MSCoD/blob/main/MSCoD.png)
![Results](https://github.com/xulong0826/MSCoD/blob/main/results.png)

You can build your own environment through `conda env create -f mscod.yml`. Here the main packages are listed:

| Package           | Version   |
|-------------------|-----------|
| CUDA              | 11.6      |
| NumPy             | 1.23.1    |
| Python            | 3.9       |
| PyTorch           | 1.12.0    |
| PyTorch Geometric | 2.1.0     |
| RDKit             | 2023.9.5  |

For evaluation, you will need to install `vina` (affinity), `posecheck` (clash, strain energy, and key interactions), and `spyrmsd` (rmsd).

```bash
# for vina docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# for posecheck evaluation
git clone https://github.com/cch1999/posecheck.git
cd posecheck
git checkout 57a1938  # the calculation of strain energy used in our paper
pip install -e .
pip install -r requirements.txt
conda install -c mx reduce

# for spyrmsd
conda install spyrmsd -c conda-forge
```

-----
## Data
Data used for training/evaluating the model should be placed in the `data` folder by default, and is accessible in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK) Google Drive folder (from [Targetdiff](https://github.com/guanjq/targetdiff)).

To train the model from scratch, download the lmdb file and split file into the data folder:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, download and unzip the `test_set.zip` into the data folder. It includes the original PDB files that will be used in Vina Docking.

---

## Training
alternatively (with data folder correctly configured):

```bash
python train_bfn.py --exp_name ${EXP_NAME} --revision ${REVISION}
```

where the default values should be set the same as:
```bash
python train_bfn.py --sigma1_coord 0.03 --beta1 1.5 --lr 5e-4 --time_emb_dim 1 --epochs 15 --max_grad_norm Q --destination_prediction True --use_discrete_t True --num_samples 10 --sampling_strategy end_back_pmf
```

### Testing
For quick evaluation of the official checkpoint:
```bash
python train_bfn.py --test_only --no_wandb --ckpt_path ./checkpoints/last.ckpt
```

### Debugging
For quick debugging training process, run:
```bash
python train_bfn.py --no_wandb --debug --epochs 1
```

## Sampling
We provide the pretrained checkpoint as [last.ckpt](https://drive.google.com/drive/folders/1seq3iQswNg9AsHObEf2opNlnYj0ojWnF?usp=drive_link). 

### Sampling for pockets in the testset
```bash
python train_bfn.py --config_file configs/default.yaml --exp_name ${EXP_NAME} --revision ${REVISION} --test_only --num_samples ${NUM_MOLS_PER_POCKET} --sample_steps 100
```

The output molecules `vina_docked.pt` for all 100 test pockets will be saved in `./logs/${USER}_bfn_sbdd/${EXP_NAME}/${REVISION}/test_outputs/${TIMESTAMP}` folders.

## Evaluation
### Evaluating molecules
For binding affinity (Vina Score / Min / Dock) and molecular properties (QED, SA), it is calculated upon sampling.

For PoseCheck (strain energy, clashes) and other conformational results (bond length, bond angle, torsion angle, RMSD), please refer to `eval/caculate_all_metrics` folder.

### Usage Instructions for Comprehensive Metrics Calculation
1. Obtain the vina_docked.pt file.
2. Run 1_add_rmsd_pt.py followed by 2_add_pose_pt.py.
3. After these steps, you will have vina_docked_pose_checked.pt.
Finally, run 3_all_metrics.py to calculate all the metrics described in the paper.

### Evaluating meta files
We provide samples for all SBDD baselines in the [sample](https://drive.google.com/drive/folders/1seq3iQswNg9AsHObEf2opNlnYj0ojWnF?usp=drive_link) Google Drive folder.

## Example: 7xkj Pocket Evaluation
Usage Instructions
### First, generate ligands and obtain the JSON file containing some metrics:
1. python eval/7xkj/sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec_7xkj_lig_gdp.sdf
2. python eval/7xkj/sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec_7xkj_lig_6ic.sdf
### Then, run the following command to visualize the results:
1. python view.py

## Contact
If you have any questions or issues, please feel free to contact us via email(xulong0826@outlook.com) or open an issue in this repository.
