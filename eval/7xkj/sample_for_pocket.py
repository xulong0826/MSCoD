# import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
# import shutil

# import torch

# from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

# import datetime, pytz

from core.config.config import Config, parse_config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback
from core.callbacks.validation_callback_for_sample import (
    DockingTestCallback,
    OUT_DIR
)

import core.utils.transforms as trans
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# from absl import logging
# import glob

from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem


def get_dataloader_from_pdb(cfg):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path

    # load protein and ligand
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)
    lig_pos = ligand_dict["pos"]

    print('[DEBUG] get_dataloader')
    print(lig_pos.shape, lig_pos.mean(axis=0))

    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn

    # transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
    ]
    transform = Compose(transform_list)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
            f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")

    # dataloader
    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples
    ##----------------------------- num_samples ---------------------------##
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
    print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader


def call(protein_fn, ligand_fn, ckpt_path='./checkpoints/last.ckpt',
         num_samples=100, sample_steps=100, sample_num_atoms='prior', 
         beta1=1.5, sigma1_coord=0.03, sampling_strategy='end_back', seed=1234):
    
    cfg = Config('./checkpoints/config.yaml')
    seed_everything(cfg.seed)
    
    cfg.evaluation.protein_path = protein_fn
    cfg.evaluation.ligand_path = ligand_fn
    cfg.evaluation.ckpt_path = ckpt_path
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.sample_steps = sample_steps
    cfg.evaluation.sample_num_atoms = sample_num_atoms # or 'prior'
    cfg.dynamics.beta1 = beta1
    cfg.dynamics.sigma1_coord = sigma1_coord
    cfg.dynamics.sampling_strategy = sampling_strategy
    cfg.seed = seed
    cfg.train.max_grad_norm = 'Q'

    # print(f"The config of this process is:\n{cfg}")

    print(protein_fn, ligand_fn)
    test_loader = get_dataloader_from_pdb(cfg)
    # wandb_logger.log_hyperparams(cfg.todict())

    model = SBDDTrainLoop(config=cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        # logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)


class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16
        print(f"protein_fn: {self.protein_fn}, ref_ligand_fn: {self.ref_ligand_fn}, ligand_fn: {self.ligand_fn}")

    def vina_dock(self, mol):
        chem_results = {}

        try:
            # qed, logp, sa, lipinski, ring size, etc
            chem_results.update(scoring_func.get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()

            # docking                
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename=self.ref_ligand_fn, protein_root=self.protein_fn)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=self.exhaustiveness)

            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            chem_results['vina_dock'] = docking_results[0]['affinity']
            # chem_results['vina_dock_pose'] = docking_results[0]['pose']
            return chem_results
        except Exception as e:
            print(e)
        
        return chem_results

    def pose_check(self, mol):
        pc = PoseCheck()

        pose_check_results = {}

        protein_ready = False
        try:
            pc.load_protein_from_pdb(self.protein_fn)
            protein_ready = True
        except ValueError as e:
            return pose_check_results

        ligand_ready = False
        try:
            pc.load_ligands_from_mols([mol])
            ligand_ready = True
        except ValueError as e:
            return pose_check_results

        if ligand_ready:
            try:
                strain = pc.calculate_strain_energy()[0]
                pose_check_results['strain'] = strain
            except Exception as e:
                pass

        if protein_ready and ligand_ready:
            try:
                clash = pc.calculate_clashes()[0]
                pose_check_results['clash'] = clash
            except Exception as e:
                pass

            try:
                df = pc.calculate_interactions()
                columns = np.array([column[2] for column in df.columns])
                flags = np.array([df[column][0] for column in df.columns])
                
                def count_inter(inter_type):
                    if len(columns) == 0:
                        return 0
                    count = sum((columns == inter_type) & flags)
                    return count

                # ['Hydrophobic', 'HBDonor', 'VdWContact', 'HBAcceptor']
                hb_donor = count_inter('HBDonor')
                hb_acceptor = count_inter('HBAcceptor')
                vdw = count_inter('VdWContact')
                hydrophobic = count_inter('Hydrophobic')

                pose_check_results['hb_donor'] = hb_donor
                pose_check_results['hb_acceptor'] = hb_acceptor
                pose_check_results['vdw'] = vdw
                pose_check_results['hydrophobic'] = hydrophobic
            except Exception as e:
                pass

        for k, v in pose_check_results.items():
            mol.SetProp(k, str(v))

        return pose_check_results
    
    def evaluate(self):
        mol = Chem.SDMolSupplier(self.ligand_fn, removeHs=False)[0]
       
        chem_results = self.vina_dock(mol)
        pose_check_results = self.pose_check(mol)
        chem_results.update(pose_check_results)

        return chem_results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def calculate_boxplot_stats(data_list, metric_name):
    """计算单个指标的箱线图统计数据"""
    if len(data_list) == 0:
        return None
    
    data = np.array(data_list)
    
    stats = {
        'count': len(data),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
    }
    
    # 计算箱线图的须线
    stats['lower_whisker'] = max(stats['min'], stats['q1'] - 1.5 * stats['iqr'])
    stats['upper_whisker'] = min(stats['max'], stats['q3'] + 1.5 * stats['iqr'])
    
    # 找出异常值
    outliers = data[(data < stats['lower_whisker']) | (data > stats['upper_whisker'])]
    stats['outliers'] = outliers.tolist()
    stats['num_outliers'] = len(outliers)
    
    return stats

def analyze_all_metrics(all_metrics):
    """分析所有指标的统计数据"""
    if not all_metrics:
        print("没有可用的指标数据")
        return None
    
    # 定义需要分析的指标
    metrics_to_analyze = [
        'qed', 'sa', 'lipinski', 'logp', 'atom_num',
        'vina_score', 'vina_minimize', 'vina_dock',
        'strain', 'clash', 'hb_donor', 'hb_acceptor', 
        'vdw', 'hydrophobic'
    ]
    
    stats_results = {}
    
    print("\n" + "="*100)
    print("箱线图统计分析结果")
    print("="*100)
    print(f"{'指标':<15} {'数量':<6} {'均值':<10} {'中位数':<10} {'标准差':<8} {'最小值':<8} {'最大值':<8} {'Q1':<8} {'Q3':<8} {'异常值':<6}")
    print("-"*100)
    
    for metric in metrics_to_analyze:
        # 提取该指标的所有值
        values = []
        for sample in all_metrics:
            if metric in sample and sample[metric] is not None:
                try:
                    values.append(float(sample[metric]))
                except (ValueError, TypeError):
                    continue
        
        if values:
            stats = calculate_boxplot_stats(values, metric)
            stats_results[metric] = stats
            
            print(f"{metric:<15} {stats['count']:<6} {stats['mean']:<10.3f} {stats['median']:<10.3f} "
                  f"{stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f} "
                  f"{stats['q1']:<8.3f} {stats['q3']:<8.3f} {stats['num_outliers']:<6}")
    
    print("="*100)
    
    return stats_results

def print_key_insights(stats_results):
    """打印关键洞察"""
    print("\n" + "="*60)
    print("关键指标详细分析")
    print("="*60)
    
    key_metrics = {
        'vina_dock': 'Vina对接得分 (越负越好)',
        'qed': 'QED药物相似性 (0-1, 越高越好)', 
        'sa': 'SA合成可达性 (1-10, 越低越好)',
        'lipinski': '越大越好)',
        'strain': '应变能 (越低越好)',
        'clash': '原子冲突数 (越少越好)'
    }
    
    for metric, description in key_metrics.items():
        if metric in stats_results:
            stats = stats_results[metric]
            print(f"\n{metric.upper()} - {description}")
            print(f"  均值: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  中位数: {stats['median']:.3f}")
            print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  四分位距(IQR): {stats['iqr']:.3f}")
            print(f"  异常值: {stats['num_outliers']} 个")
            
            # 特殊建议
            if metric == 'vina_dock':
                good_binding = sum(1 for v in [s.get('vina_dock') for s in all_metrics] if v and v < -8.0)
                print(f"  强结合分子 (<-8.0): {good_binding}/{stats['count']} ({good_binding/stats['count']*100:.1f}%)")
            
            elif metric == 'qed':
                drug_like = sum(1 for v in [s.get('qed') for s in all_metrics] if v and v > 0.5)
                print(f"  药物相似性好 (>0.5): {drug_like}/{stats['count']} ({drug_like/stats['count']*100:.1f}%)")

def save_results(all_metrics, stats_results, prefix="docking_analysis"):
    """保存分析结果"""
    # 保存原始数据
    with open(f'{prefix}_raw_data.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, cls=NpEncoder)
    
    # 保存统计结果
    with open(f'{prefix}_stats.json', 'w') as f:
        json.dump(stats_results, f, indent=2, cls=NpEncoder)
    
    print(f"\n结果已保存:")
    print(f"  原始数据: {prefix}_raw_data.json")
    print(f"  统计分析: {prefix}_stats.json")

def evaluation_only(protein_path, ligand_path):
    """只进行评估，不生成新分子"""
    print("开始评估现有配体...")
    
    # 直接对指定的配体文件进行评估
    try:
        metrics = Metrics(protein_path, ligand_path, ligand_path).evaluate()
        metrics['sample_id'] = 'reference'
        
        print("评估完成！")
        print(json.dumps(metrics, indent=2, cls=NpEncoder))
        
        # 保存单个结果
        with open('reference_ligand_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2, cls=NpEncoder)
        
        print(f"\n结果已保存到: reference_ligand_evaluation.json")
        return metrics
        
    except Exception as e:
        print(f"评估失败: {e}")
        return None

if __name__ == '__main__':
    protein_path = sys.argv[1]
    ligand_path = sys.argv[2]
    
    # 添加命令行参数判断是否只进行评估
    evaluation_only_mode = len(sys.argv) > 3 and sys.argv[3] == '--eval-only'
    
    if evaluation_only_mode:
        # 只进行评估模式
        print("运行模式: 仅评估现有配体")
        evaluation_only(protein_path, ligand_path)
    else:
        # 原有的生成+评估模式
        print("运行模式: 生成新分子并评估")
        call(protein_path, ligand_path)
        
        # 收集所有样本的指标
        all_metrics = []
        print("正在评估所有生成的分子...")
        
        for i in range(100):
            out_fn = f'output/{i}.sdf'
            if os.path.exists(out_fn):
                try:
                    metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
                    metrics['sample_id'] = i
                    all_metrics.append(metrics)
                    print(f"Sample {i}: 完成评估")
                    
                except Exception as e:
                    print(f"Sample {i}: 评估失败 - {e}")
            else:
                print(f"Sample {i}: 文件不存在")
        
        print(f"\n总评估样本数: {len(all_metrics)}")
        
        if all_metrics:
            # 进行统计分析
            stats_results = analyze_all_metrics(all_metrics)
            
            if stats_results:
                # 打印关键洞察
                print_key_insights(stats_results)
                
                # 保存结果
                save_results(all_metrics, stats_results)
                
                print(f"\n分析完成！共分析了 {len(all_metrics)} 个分子的 {len(stats_results)} 项指标。")
            
        else:
            print("没有成功评估任何分子，请检查文件路径和格式。")

    # cmd
    #python sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec_7xkj_lig_gdp.sdf

    #python sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec_7xkj_lig_6ic.sdf 
