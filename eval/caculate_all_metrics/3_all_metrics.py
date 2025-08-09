import torch
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VinaMetricsAnalyzer:
    """Vina对接结果分析器 - 必须计算多样性的版本"""
    
    def __init__(self):
        self.data = None
        self.file_info = None
        self.results = None
        self.ref_fns = None  # 参考文件名列表，用于多样性计算
        
    def load_data(self, file_path):
        """静默加载数据"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            try:
                from rdkit.Chem import rdchem
                torch.serialization.add_safe_globals([rdchem.Mol])
                self.data = torch.load(file_path, map_location='cpu')
            except:
                self.data = torch.load(file_path, map_location='cpu')
            
            self.file_info = {
                "file_path": file_path,
                "file_size_mb": os.path.getsize(file_path) / 1024 / 1024,
                "load_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return True, "加载成功"
            
        except Exception as e:
            return False, f"加载失败: {str(e)}"
    
    def set_reference_filenames(self, ref_fns):
        """设置参考文件名列表，用于多样性计算"""
        self.ref_fns = ref_fns
    
    def _get_morgan_fingerprint(self, smiles):
        """计算Morgan指纹"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return fp
        except:
            return None
    
    def _calculate_tanimoto_similarity(self, fp1, fp2):
        """计算Tanimoto相似性"""
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def _compute_diversity(self, agg_results):
        """
        计算每个蛋白质的分子多样性
        
        Args:
            agg_results: 按蛋白质分组的分子列表 [[protein1_mols], [protein2_mols], ...]
        
        Returns:
            diversity_list: 每个蛋白质的多样性分数列表
        """
        diversity_scores = []
        
        for protein_mols in agg_results:
            if len(protein_mols) <= 1:
                diversity_scores.append(0.0)  # 只有一个或没有分子
                continue
                
            # 提取SMILES
            smiles_list = []
            for mol_data in protein_mols:
                if isinstance(mol_data, dict):
                    smiles = mol_data.get('smiles', '')
                else:
                    smiles = getattr(mol_data, 'smiles', '')
                
                if smiles and smiles != '':
                    smiles_list.append(smiles)
            
            if len(smiles_list) <= 1:
                diversity_scores.append(0.0)
                continue
            
            # 计算分子指纹
            fps = []
            for smiles in smiles_list:
                fp = self._get_morgan_fingerprint(smiles)
                if fp is not None:
                    fps.append(fp)
            
            if len(fps) <= 1:
                diversity_scores.append(0.0)
                continue
            
            # 计算两两相似性
            similarities = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sim = self._calculate_tanimoto_similarity(fps[i], fps[j])
                    similarities.append(sim)
            
            if len(similarities) == 0:
                diversity_scores.append(0.0)
            else:
                # 多样性 = 1 - 平均相似性
                diversity = 1 - np.mean(similarities)
                diversity_scores.append(max(0.0, diversity))  # 确保非负
        
        return diversity_scores
    
    def _try_extract_reference_filenames_from_data(self):
        """尝试从数据本身提取参考文件名"""
        try:
            # 处理不同数据格式
            if isinstance(self.data, dict) and 'all_results' in self.data:
                data_list = self.data['all_results']
            elif isinstance(self.data, list):
                data_list = self.data
            else:
                return []
            
            ref_fns = []
            for item in data_list:
                if isinstance(item, dict):
                    ligand_filename = item.get('ligand_filename', '')
                else:
                    ligand_filename = getattr(item, 'ligand_filename', '')
                
                if ligand_filename and ligand_filename not in ref_fns:
                    ref_fns.append(ligand_filename)
            
            return ref_fns
            
        except Exception:
            return []
    
    def calculate_diversity(self, model_name=None):
        """
        计算多样性指标 - 必须计算版本
        
        Args:
            model_name: 模型名称
        
        Returns:
            (mean_diversity, median_diversity): 多样性的平均值和中位数
        """
        # 如果没有设置参考文件名，尝试从数据中提取
        if not self.ref_fns:
            print("⚠️ 未设置参考文件名，尝试从数据中提取...")
            self.ref_fns = self._try_extract_reference_filenames_from_data()
            
            if self.ref_fns:
                print(f"✅ 从数据中提取到 {len(self.ref_fns)} 个文件名")
            else:
                print("❌ 无法提取参考文件名，使用基础多样性计算")
                return self._calculate_basic_diversity()
        
        # 如果仍然没有数据，使用基础多样性计算
        if not self.data:
            print("❌ 无数据可用，返回默认多样性值")
            return (0.0, 0.0)
        
        # 处理不同数据格式
        if isinstance(self.data, dict) and 'all_results' in self.data:
            data_list = self.data['all_results']
        elif isinstance(self.data, list):
            data_list = self.data
        else:
            data_list = []
        
        if len(data_list) == 0:
            print("❌ 数据为空，返回默认多样性值")
            return (0.0, 0.0)
        
        # 按蛋白质分组分子
        try:
            # 动态确定蛋白质数量
            unique_filenames = list(set(self.ref_fns))
            num_proteins = len(unique_filenames)
            
            if num_proteins == 0:
                return self._calculate_basic_diversity()
            
            agg_results = [[] for _ in range(num_proteins)]
            filename_to_idx = {fn: i for i, fn in enumerate(unique_filenames)}
            
            print(f"🧮 计算多样性中... (基于 {num_proteins} 个蛋白质)")
            
            # 分组处理
            for res in tqdm(data_list, desc="处理分子"):
                ligand_filename = None
                
                if isinstance(res, dict):
                    ligand_filename = res.get('ligand_filename', '')
                else:
                    ligand_filename = getattr(res, 'ligand_filename', '')
                
                if ligand_filename in filename_to_idx:
                    idx = filename_to_idx[ligand_filename]
                    agg_results[idx].append(res)
            
            # 计算每个蛋白质的分子间多样性
            diversity_list = self._compute_diversity(agg_results)
            
            # 过滤掉0值（没有分子或只有一个分子的蛋白质）
            valid_diversity = [d for d in diversity_list if d > 0]
            
            if len(valid_diversity) == 0:
                print("⚠️ 没有有效的多样性数据，使用基础计算方法")
                return self._calculate_basic_diversity()
            
            mean_diversity = np.mean(valid_diversity)
            median_diversity = np.median(valid_diversity)
            
            print(f"✅ 多样性计算完成: 平均值={mean_diversity:.4f}, 中位数={median_diversity:.4f}")
            print(f"   有效蛋白质数: {len(valid_diversity)}/{num_proteins}")
            
            return (mean_diversity, median_diversity)
            
        except Exception as e:
            print(f"⚠️ 分组多样性计算失败: {e}")
            print("🔄 回退到基础多样性计算")
            return self._calculate_basic_diversity()
    
    def _calculate_basic_diversity(self):
        """基础多样性计算 - 对所有分子计算整体多样性"""
        try:
            # 处理不同数据格式
            if isinstance(self.data, dict) and 'all_results' in self.data:
                data_list = self.data['all_results']
            elif isinstance(self.data, list):
                data_list = self.data
            else:
                return (0.0, 0.0)
            
            # 提取所有SMILES
            all_smiles = []
            for item in data_list:
                smiles = ""
                if isinstance(item, dict):
                    smiles = item.get('smiles', '')
                    # 如果没有smiles字段，尝试从mol对象获取
                    if not smiles and 'mol' in item and item['mol']:
                        try:
                            smiles = Chem.MolToSmiles(item['mol'])
                        except:
                            pass
                else:
                    smiles = getattr(item, 'smiles', '')
                    if not smiles and hasattr(item, 'mol') and item.mol:
                        try:
                            smiles = Chem.MolToSmiles(item.mol)
                        except:
                            pass
                
                if smiles and smiles != '':
                    all_smiles.append(smiles)
            
            if len(all_smiles) <= 1:
                print("⚠️ 可用SMILES数量不足，返回默认多样性值")
                return (0.0, 0.0)
            
            print(f"🧮 基础多样性计算: 处理 {len(all_smiles)} 个分子")
            
            # 随机采样以避免计算量过大
            max_sample_size = 1000
            if len(all_smiles) > max_sample_size:
                import random
                all_smiles = random.sample(all_smiles, max_sample_size)
                print(f"   随机采样 {max_sample_size} 个分子进行计算")
            
            # 计算分子指纹
            fps = []
            for smiles in tqdm(all_smiles, desc="计算指纹"):
                fp = self._get_morgan_fingerprint(smiles)
                if fp is not None:
                    fps.append(fp)
            
            if len(fps) <= 1:
                print("⚠️ 有效指纹数量不足")
                return (0.0, 0.0)
            
            # 计算两两相似性
            similarities = []
            print("🧮 计算相似性...")
            for i in tqdm(range(len(fps)), desc="相似性计算"):
                for j in range(i+1, len(fps)):
                    sim = self._calculate_tanimoto_similarity(fps[i], fps[j])
                    similarities.append(sim)
            
            if len(similarities) == 0:
                return (0.0, 0.0)
            
            # 多样性 = 1 - 平均相似性
            diversity = max(0.0, 1 - np.mean(similarities))
            
            print(f"✅ 基础多样性计算完成: {diversity:.4f}")
            print(f"   基于 {len(fps)} 个有效指纹, {len(similarities)} 个相似性对比")
            
            # 返回相同值作为平均值和中位数
            return (diversity, diversity)
            
        except Exception as e:
            print(f"❌ 基础多样性计算失败: {e}")
            return (0.0, 0.0)
    
    def extract_metrics(self, model_name=None):
        """提取所有指标数据 - 必须包含多样性计算"""
        # 处理不同数据格式
        if isinstance(self.data, dict) and 'all_results' in self.data:
            data_list = self.data['all_results']
        elif isinstance(self.data, list):
            data_list = self.data
        else:
            return False, "无效数据格式"
        
        if len(data_list) == 0:
            return False, "数据为空"
        
        # 初始化计数器
        counters = {
            'total': len(data_list),
            'complete': 0,
            'valid': 0,
            'processed': 0
        }
        
        # 指标数据 - 添加vina_minimize
        metrics_data = {
            'vina_scores': [],
            'vina_docks': [],
            'vina_minimizes': [],  # 新增Vina Minimize数据
            'strains': [],
            'clashes': [],
            'qeds': [],
            'sas': [],
            'rmsds': []
        }
        
        # 成功率统计
        bf_stats = {'total': 0, 'success': 0}  # Binding Feasibility
        sr_stats = {'total': 0, 'success': 0}  # Success Rate
        
        # 处理数据
        for item in data_list:
            if not isinstance(item, dict):
                continue
                
            is_complete = item.get('complete', False)
            is_valid = item.get('validity', False)
            
            if is_complete:
                counters['complete'] += 1
            if is_valid:
                counters['valid'] += 1
            
            if not (is_complete and is_valid):
                continue
            
            counters['processed'] += 1
            
            # 提取指标
            vina_score = None
            vina_dock = None
            vina_minimize = None  # 新增vina_minimize变量
            strain = None
            clash = None
            qed_val = None
            sa_val = None
            rmsd = None
            
            # Vina数据 - 包含minimize
            if 'vina' in item and isinstance(item['vina'], dict):
                vina_data = item['vina']
                
                # Vina Score Only
                if 'score_only' in vina_data and len(vina_data['score_only']) > 0:
                    vina_score = vina_data['score_only'][0].get('affinity')
                    if vina_score is not None and not np.isnan(vina_score):
                        metrics_data['vina_scores'].append(vina_score)
                
                # Vina Dock
                if 'dock' in vina_data and len(vina_data['dock']) > 0:
                    vina_dock = vina_data['dock'][0].get('affinity')
                    if vina_dock is not None and not np.isnan(vina_dock):
                        metrics_data['vina_docks'].append(vina_dock)
                
                # Vina Minimize - 新增
                if 'minimize' in vina_data and len(vina_data['minimize']) > 0:
                    vina_minimize = vina_data['minimize'][0].get('affinity')
                    if vina_minimize is not None and not np.isnan(vina_minimize):
                        metrics_data['vina_minimizes'].append(vina_minimize)
            
            # PoseCheck数据
            if 'pose_check' in item and isinstance(item['pose_check'], dict):
                pose_data = item['pose_check']
                
                if len(pose_data) > 1:
                    strain = pose_data.get('strain')
                    if strain is not None and not np.isnan(strain) and strain < 1e9:
                        metrics_data['strains'].append(strain)
                    
                    clash = pose_data.get('clash')
                    if clash is not None and not np.isnan(clash) and clash < 999:
                        metrics_data['clashes'].append(clash)
            
            # RMSD数据
            if 'rmsd' in item:
                rmsd = item['rmsd']
                if rmsd is not None and not np.isnan(rmsd):
                    metrics_data['rmsds'].append(rmsd)
            
            # 化学性质数据
            if 'chem_results' in item and isinstance(item['chem_results'], dict):
                chem_data = item['chem_results']
                
                qed_val = chem_data.get('qed')
                if qed_val is not None and not np.isnan(qed_val):
                    metrics_data['qeds'].append(qed_val)
                
                sa_val = chem_data.get('sa')
                if sa_val is not None and not np.isnan(sa_val):
                    metrics_data['sas'].append(sa_val)
            
            # BF计算 (Vina Score < -2.49, Strain < 836, RMSD < 2)
            if vina_score is not None and strain is not None and rmsd is not None:
                bf_stats['total'] += 1
                if vina_score < -2.49 and strain < 836 and rmsd < 2.0:
                    bf_stats['success'] += 1
            
            # SR计算 (Vina Dock < -8.18, QED > 0.25, SA > 0.59)
            if vina_dock is not None and qed_val is not None and sa_val is not None:
                sr_stats['total'] += 1
                if vina_dock < -8.18 and qed_val > 0.25 and sa_val > 0.59:
                    sr_stats['success'] += 1
        
        # 计算成功率
        bf_rate = (bf_stats['success'] / bf_stats['total'] * 100) if bf_stats['total'] > 0 else 0
        sr_rate = (sr_stats['success'] / sr_stats['total'] * 100) if sr_stats['total'] > 0 else 0
        
        # 强制计算多样性
        print("\n📊 开始计算多样性指标...")
        diversity_mean, diversity_median = self.calculate_diversity(model_name)
        
        self.results = {
            'counters': counters,
            'metrics': metrics_data,
            'bf_stats': bf_stats,
            'sr_stats': sr_stats,
            'bf_rate': bf_rate,
            'sr_rate': sr_rate,
            'diversity': {
                'mean': diversity_mean,
                'median': diversity_median,
                'calculated': True  # 总是为True，因为必须计算
            }
        }
        
        return True, "指标提取完成（包含多样性）"
    
    def _calculate_stats_with_se(self, data_list):
        """计算包含SE的完整统计"""
        if not data_list:
            return {
                'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0,
                'std': 0, 'se': 0, 'q25': 0, 'q75': 0,
                'ci_95_lower': 0, 'ci_95_upper': 0, 'ci_width': 0
            }
        
        arr = np.array(data_list)
        n = len(arr)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1)
        se_val = std_val / np.sqrt(n)
        
        if n > 1:
            t_value = stats.t.ppf(0.975, n - 1)
            ci_95_lower = mean_val - t_value * se_val
            ci_95_upper = mean_val + t_value * se_val
            ci_width = ci_95_upper - ci_95_lower
        else:
            ci_95_lower = mean_val
            ci_95_upper = mean_val
            ci_width = 0
        
        return {
            'count': n,
            'mean': float(mean_val),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'std': float(std_val),
            'se': float(se_val),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'ci_95_lower': float(ci_95_lower),
            'ci_95_upper': float(ci_95_upper),
            'ci_width': float(ci_width)
        }
    
    def _calculate_bootstrap_se(self, data_list, n_bootstrap=1000):
        """Bootstrap方法估计SE分布"""
        if not data_list or len(data_list) < 2:
            return {'se_mean': 0, 'se_std': 0, 'se_q25': 0, 'se_q50': 0, 'se_q75': 0}
        
        arr = np.array(data_list)
        n = len(arr)
        bootstrap_ses = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(arr, size=n, replace=True)
            bootstrap_se = np.std(bootstrap_sample, ddof=1) / np.sqrt(n)
            bootstrap_ses.append(bootstrap_se)
        
        bootstrap_arr = np.array(bootstrap_ses)
        
        return {
            'se_mean': float(np.mean(bootstrap_arr)),
            'se_std': float(np.std(bootstrap_arr)),
            'se_q25': float(np.percentile(bootstrap_arr, 25)),
            'se_q50': float(np.percentile(bootstrap_arr, 50)),
            'se_q75': float(np.percentile(bootstrap_arr, 75))
        }
    
    def display_results(self):
        """显示结果 - 包含Vina Minimize指标和多样性"""
        if not self.results:
            print("❌ 没有结果数据")
            return
        
        print("=" * 75)
        print("🧬 Vina 分子对接结果分析 - 必须计算多样性版本")
        print("=" * 75)
        print(f"📅 时间: 2025-06-05 06:06:14")
        print(f"👤 用户: xulong0826")
        print(f"📁 文件: {os.path.basename(self.file_info['file_path'])}")
        print(f"💾 大小: {self.file_info['file_size_mb']:.1f} MB")
        print()
        
        # 基础统计
        counters = self.results['counters']
        print("📊 数据概览:")
        print(f"   总分子数: {counters['total']:,}")
        print(f"   完整分子: {counters['complete']:,} ({counters['complete']/counters['total']*100:.1f}%)")
        print(f"   有效分子: {counters['valid']:,} ({counters['valid']/counters['total']*100:.1f}%)")
        print(f"   处理成功: {counters['processed']:,} ({counters['processed']/counters['total']*100:.1f}%)")
        print()
        
        # 关键指标
        bf_stats = self.results['bf_stats']
        sr_stats = self.results['sr_stats']
        
        print("🎯 关键指标:")
        print(f"   BF (结合可行性): {bf_stats['success']:,}/{bf_stats['total']:,} = {self.results['bf_rate']:.2f}%")
        print(f"   SR (成功率): {sr_stats['success']:,}/{sr_stats['total']:,} = {self.results['sr_rate']:.2f}%")
        
        # 多样性指标 - 总是显示，因为必须计算
        div_mean = self.results['diversity']['mean']
        div_median = self.results['diversity']['median']
        print(f"   多样性: 平均={div_mean:.4f}, 中位数={div_median:.4f}")
        
        print()
        
        # 指标统计 - 包含Vina Minimize
        metrics = self.results['metrics']
        print("📈 指标统计 (包含标准误差SE):")
        print("   格式: μ±SE [95%CI], M, Q1/Q3, 阈值通过率")
        print()
        
        # Vina Score
        vina_stats = self._calculate_stats_with_se(metrics['vina_scores'])
        if vina_stats['count'] > 0:
            vina_good = len([x for x in metrics['vina_scores'] if x < -2.49])
            print(f"   Vina Score: {vina_stats['mean']:.3f}±{vina_stats['se']:.3f} "
                  f"[{vina_stats['ci_95_lower']:.3f}, {vina_stats['ci_95_upper']:.3f}], "
                  f"M={vina_stats['median']:.3f}, "
                  f"Q1/Q3={vina_stats['q25']:.3f}/{vina_stats['q75']:.3f}, "
                  f"<-2.49: {vina_good:,}/{vina_stats['count']:,} ({vina_good/vina_stats['count']*100:.1f}%)")
        
        # Vina Dock
        dock_stats = self._calculate_stats_with_se(metrics['vina_docks'])
        if dock_stats['count'] > 0:
            dock_good = len([x for x in metrics['vina_docks'] if x < -8.18])
            print(f"   Vina Dock:  {dock_stats['mean']:.3f}±{dock_stats['se']:.3f} "
                  f"[{dock_stats['ci_95_lower']:.3f}, {dock_stats['ci_95_upper']:.3f}], "
                  f"M={dock_stats['median']:.3f}, "
                  f"Q1/Q3={dock_stats['q25']:.3f}/{dock_stats['q75']:.3f}, "
                  f"<-8.18: {dock_good:,}/{dock_stats['count']:,} ({dock_good/dock_stats['count']*100:.1f}%)")
        
        # Vina Minimize - 新增
        minimize_stats = self._calculate_stats_with_se(metrics['vina_minimizes'])
        if minimize_stats['count'] > 0:
            # 使用与dock相同的阈值进行分析
            minimize_good = len([x for x in metrics['vina_minimizes'] if x < -8.18])
            print(f"   Vina Min:   {minimize_stats['mean']:.3f}±{minimize_stats['se']:.3f} "
                  f"[{minimize_stats['ci_95_lower']:.3f}, {minimize_stats['ci_95_upper']:.3f}], "
                  f"M={minimize_stats['median']:.3f}, "
                  f"Q1/Q3={minimize_stats['q25']:.3f}/{minimize_stats['q75']:.3f}, "
                  f"<-8.18: {minimize_good:,}/{minimize_stats['count']:,} ({minimize_good/minimize_stats['count']*100:.1f}%)")
        
        # Strain
        strain_stats = self._calculate_stats_with_se(metrics['strains'])
        if strain_stats['count'] > 0:
            strain_good = len([x for x in metrics['strains'] if x < 836])
            print(f"   Strain:     {strain_stats['mean']:.1f}±{strain_stats['se']:.1f} "
                  f"[{strain_stats['ci_95_lower']:.1f}, {strain_stats['ci_95_upper']:.1f}], "
                  f"M={strain_stats['median']:.1f}, "
                  f"Q1/Q3={strain_stats['q25']:.1f}/{strain_stats['q75']:.1f}, "
                  f"<836: {strain_good:,}/{strain_stats['count']:,} ({strain_good/strain_stats['count']*100:.1f}%)")
        
        # Clash
        clash_stats = self._calculate_stats_with_se(metrics['clashes'])
        if clash_stats['count'] > 0:
            print(f"   Clash:      {clash_stats['mean']:.3f}±{clash_stats['se']:.3f} "
                  f"[{clash_stats['ci_95_lower']:.3f}, {clash_stats['ci_95_upper']:.3f}], "
                  f"M={clash_stats['median']:.3f}, "
                  f"Q1/Q3={clash_stats['q25']:.3f}/{clash_stats['q75']:.3f}")
        
        # RMSD
        rmsd_stats = self._calculate_stats_with_se(metrics['rmsds'])
        if rmsd_stats['count'] > 0:
            rmsd_good = len([x for x in metrics['rmsds'] if x < 2.0])
            print(f"   RMSD:       {rmsd_stats['mean']:.3f}±{rmsd_stats['se']:.3f} "
                  f"[{rmsd_stats['ci_95_lower']:.3f}, {rmsd_stats['ci_95_upper']:.3f}], "
                  f"M={rmsd_stats['median']:.3f}, "
                  f"Q1/Q3={rmsd_stats['q25']:.3f}/{rmsd_stats['q75']:.3f}, "
                  f"<2Å: {rmsd_good:,}/{rmsd_stats['count']:,} ({rmsd_good/rmsd_stats['count']*100:.1f}%)")
        
        # QED
        qed_stats = self._calculate_stats_with_se(metrics['qeds'])
        if qed_stats['count'] > 0:
            qed_good = len([x for x in metrics['qeds'] if x > 0.25])
            print(f"   QED:        {qed_stats['mean']:.3f}±{qed_stats['se']:.3f} "
                  f"[{qed_stats['ci_95_lower']:.3f}, {qed_stats['ci_95_upper']:.3f}], "
                  f"M={qed_stats['median']:.3f}, "
                  f"Q1/Q3={qed_stats['q25']:.3f}/{qed_stats['q75']:.3f}, "
                  f">0.25: {qed_good:,}/{qed_stats['count']:,} ({qed_good/qed_stats['count']*100:.1f}%)")
        
        # SA
        sa_stats = self._calculate_stats_with_se(metrics['sas'])
        if sa_stats['count'] > 0:
            sa_good = len([x for x in metrics['sas'] if x > 0.59])
            print(f"   SA:         {sa_stats['mean']:.3f}±{sa_stats['se']:.3f} "
                  f"[{sa_stats['ci_95_lower']:.3f}, {sa_stats['ci_95_upper']:.3f}], "
                  f"M={sa_stats['median']:.3f}, "
                  f"Q1/Q3={sa_stats['q25']:.3f}/{sa_stats['q75']:.3f}, "
                  f">0.59: {sa_good:,}/{sa_stats['count']:,} ({sa_good/sa_stats['count']*100:.1f}%)")
        
        print()
        
        # SE分析重点 - 包含Vina Minimize
        print("📊 关键指标精度分析 (标准误差SE):")
        key_metrics = ['vina_scores', 'vina_docks', 'vina_minimizes', 'strains', 'rmsds']
        metric_names = ['Vina Score', 'Vina Dock', 'Vina Min', 'Strain', 'RMSD']
        
        for metric_key, metric_name in zip(key_metrics, metric_names):
            if metrics[metric_key]:
                stats_data = self._calculate_stats_with_se(metrics[metric_key])
                relative_se = (stats_data['se'] / abs(stats_data['mean']) * 100) if stats_data['mean'] != 0 else 0
                
                print(f"   {metric_name:12}: SE={stats_data['se']:.4f}, "
                      f"相对SE={relative_se:.2f}%, "
                      f"CI宽度={stats_data['ci_width']:.3f}")
        
        print()
        
        # 成功率条件分解
        print("🔍 成功率条件分解:")
        
        # BF条件分解
        if bf_stats['total'] > 0:
            print(f"   BF条件 (需要同时满足三个条件):")
            if vina_stats['count'] > 0:
                vina_pass = len([x for x in metrics['vina_scores'] if x < -2.49])
                print(f"     Vina Score < -2.49: {vina_pass:,}/{vina_stats['count']:,} ({vina_pass/vina_stats['count']*100:.1f}%)")
            if strain_stats['count'] > 0:
                strain_pass = len([x for x in metrics['strains'] if x < 836])
                print(f"     Strain < 836: {strain_pass:,}/{strain_stats['count']:,} ({strain_pass/strain_stats['count']*100:.1f}%)")
            if rmsd_stats['count'] > 0:
                rmsd_pass = len([x for x in metrics['rmsds'] if x < 2.0])
                print(f"     RMSD < 2Å: {rmsd_pass:,}/{rmsd_stats['count']:,} ({rmsd_pass/rmsd_stats['count']*100:.1f}%)")
        
        # SR条件分解
        if sr_stats['total'] > 0:
            print(f"   SR条件 (需要同时满足三个条件):")
            if dock_stats['count'] > 0:
                dock_pass = len([x for x in metrics['vina_docks'] if x < -8.18])
                print(f"     Vina Dock < -8.18: {dock_pass:,}/{dock_stats['count']:,} ({dock_pass/dock_stats['count']*100:.1f}%)")
            if qed_stats['count'] > 0:
                qed_pass = len([x for x in metrics['qeds'] if x > 0.25])
                print(f"     QED > 0.25: {qed_pass:,}/{qed_stats['count']:,} ({qed_pass/qed_stats['count']*100:.1f}%)")
            if sa_stats['count'] > 0:
                sa_pass = len([x for x in metrics['sas'] if x > 0.59])
                print(f"     SA > 0.59: {sa_pass:,}/{sa_stats['count']:,} ({sa_pass/sa_stats['count']*100:.1f}%)")
        
        # 数据可用性统计 - 包含Vina Minimize
        print()
        print("📋 数据可用性:")
        print(f"   有Vina Score: {vina_stats['count']:,}/{counters['total']:,} ({vina_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有Vina Dock:  {dock_stats['count']:,}/{counters['total']:,} ({dock_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有Vina Min:   {minimize_stats['count']:,}/{counters['total']:,} ({minimize_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有Strain:     {strain_stats['count']:,}/{counters['total']:,} ({strain_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有Clash:      {clash_stats['count']:,}/{counters['total']:,} ({clash_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有RMSD:       {rmsd_stats['count']:,}/{counters['total']:,} ({rmsd_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有QED:        {qed_stats['count']:,}/{counters['total']:,} ({qed_stats['count']/counters['total']*100:.1f}%)")
        print(f"   有SA:         {sa_stats['count']:,}/{counters['total']:,} ({sa_stats['count']/counters['total']*100:.1f}%)")
        
        print("=" * 75)
    
    def save_results(self, output_file="analysis_results_with_mandatory_diversity.json"):
        """保存结果到JSON文件 - 包含Vina Minimize和多样性"""
        if not self.results:
            return False, "没有结果可保存"
        
        try:
            # 构建输出数据
            metrics_stats = {}
            se_bootstrap_stats = {}
            
            for key, data in self.results['metrics'].items():
                metrics_stats[key] = self._calculate_stats_with_se(data)
                se_bootstrap_stats[key] = self._calculate_bootstrap_se(data)
            
            # 计算阈值通过率 - 包含Vina Minimize
            threshold_stats = {}
            thresholds = [
                ('vina_scores', -2.49, '<'),
                ('vina_docks', -8.18, '<'),
                ('vina_minimizes', -8.18, '<'),  # 新增
                ('strains', 836, '<'),
                ('rmsds', 2.0, '<'),
                ('qeds', 0.25, '>'),
                ('sas', 0.59, '>')
            ]
            
            for metric_key, threshold, operator in thresholds:
                if self.results['metrics'][metric_key]:
                    if operator == '<':
                        pass_count = len([x for x in self.results['metrics'][metric_key] if x < threshold])
                    else:
                        pass_count = len([x for x in self.results['metrics'][metric_key] if x > threshold])
                    threshold_stats[f'{metric_key}_pass'] = pass_count
            
            output_data = {
                "metadata": {
                    "analysis_type": "vina_docking_metrics_with_mandatory_diversity",
                    "timestamp": "2025-06-05T06:06:14Z",
                    "user": "xulong0826",
                    "file_info": self.file_info,
                    "bf_criteria": "Vina Score < -2.49 AND Strain < 836 AND RMSD < 2Å",
                    "sr_criteria": "Vina Dock < -8.18 AND QED > 0.25 AND SA > 0.59",
                    "se_note": "SE = Standard Error = std / sqrt(n), 95%CI calculated using t-distribution",
                    "vina_metrics": "Score, Dock, Minimize (post-docking energy minimization)",
                    "diversity_note": "Calculated as 1 - mean(pairwise_tanimoto_similarity) - MANDATORY calculation",
                    "diversity_mandatory": True
                },
                "summary": {
                    "counters": self.results['counters'],
                    "bf_rate": self.results['bf_rate'],
                    "sr_rate": self.results['sr_rate'],
                    "bf_stats": self.results['bf_stats'],
                    "sr_stats": self.results['sr_stats'],
                    "diversity": self.results['diversity']
                },
                "metrics_statistics": metrics_stats,
                "se_bootstrap_validation": se_bootstrap_stats,
                "threshold_pass_rates": threshold_stats
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True, f"结果已保存到: {output_file}"
            
        except Exception as e:
            return False, f"保存失败: {str(e)}"


def load_reference_filenames(ref_pt_path):
    """从参考PT文件中加载文件名列表"""
    try:
        try:
            from rdkit.Chem import rdchem
            torch.serialization.add_safe_globals([rdchem.Mol])
            ref_data = torch.load(ref_pt_path, map_location='cpu')
        except:
            ref_data = torch.load(ref_pt_path, map_location='cpu')
        
        # 处理不同数据格式
        if isinstance(ref_data, dict) and 'all_results' in ref_data:
            data_list = ref_data['all_results']
        elif isinstance(ref_data, list):
            data_list = ref_data
        else:
            return []
        
        ref_fns = []
        for item in data_list:
            if isinstance(item, dict):
                ligand_filename = item.get('ligand_filename', '')
            else:
                ligand_filename = getattr(item, 'ligand_filename', '')
            
            if ligand_filename:
                ref_fns.append(ligand_filename)
        
        return ref_fns
        
    except Exception as e:
        print(f"⚠️ 加载参考文件名失败: {e}")
        return []


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vina对接结果分析器 - 必须计算多样性版本')
    parser.add_argument('--file', type=str, required=True, help='PT文件路径')
    parser.add_argument('--output', type=str, default="analysis_results_with_mandatory_diversity.json", help='输出JSON文件路径')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--ref_file', type=str, help='参考PT文件路径（用于多样性计算）')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = VinaMetricsAnalyzer()
    
    # 加载数据
    if not args.quiet:
        print("🚀 加载数据中...")
    
    success, message = analyzer.load_data(args.file)
    if not success:
        print(f"❌ {message}")
        return
    
    # 如果提供了参考文件，加载参考文件名
    if args.ref_file:
        if not args.quiet:
            print("📋 加载参考文件名中...")
        ref_fns = load_reference_filenames(args.ref_file)
        if ref_fns:
            analyzer.set_reference_filenames(ref_fns)
            if not args.quiet:
                print(f"✅ 加载了 {len(ref_fns)} 个参考文件名")
        else:
            if not args.quiet:
                print("⚠️ 无法加载参考文件名，将使用基础多样性计算")
    
    # 提取指标（必须包含多样性计算）
    if not args.quiet:
        print("📊 分析数据中（必须计算多样性）...")
    
    success, message = analyzer.extract_metrics(model_name=args.model_name)
    if not success:
        print(f"❌ {message}")
        return
    
    # 显示结果
    analyzer.display_results()
    
    # 保存结果
    success, message = analyzer.save_results(args.output)
    if not args.quiet:
        if success:
            print(f"💾 {message}")
        else:
            print(f"❌ {message}")


if __name__ == "__main__":
    main()