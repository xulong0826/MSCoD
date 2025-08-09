import os
from rdkit import Chem
from rdkit import RDLogger
import torch
import numpy as np
import contextlib
from tqdm import tqdm
from posecheck import PoseCheck
from rdkit.Chem.QED import qed
from core.evaluation.utils.sascorer import compute_sa_score
import argparse
import time
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import copy
import warnings
import sys

# 关闭所有警告和提示
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def is_posecheck_complete(pose_check_data):
    """检查PoseCheck数据是否完整 - 基于参数数量判断"""
    if not pose_check_data or not isinstance(pose_check_data, dict):
        return False
    
    # 统计参数数量
    param_count = len(pose_check_data)
    
    # 如果只有1个参数（通常是strain），说明未完整处理
    # 如果有多个参数，说明已经完整处理
    if param_count <= 1:
        return False
    
    # 进一步检查是否有关键字段
    required_fields = ['clash', 'strain']
    for field in required_fields:
        if field not in pose_check_data:
            return False
    
    return True

def is_molecule_successful(mol_data):
    """判断分子是否处理成功
    成功标准：
    1. complete = True
    2. validity = True  
    3. 有完整的PoseCheck数据
    4. 有化学性质数据
    5. PoseCheck指标在合理范围内
    """
    try:
        # 检查基本完整性
        is_complete = False
        is_valid = False
        
        if isinstance(mol_data, dict):
            is_complete = mol_data.get('complete', False)
            is_valid = mol_data.get('validity', False)
        else:
            is_complete = getattr(mol_data, 'complete', False)
            is_valid = getattr(mol_data, 'validity', False)
        
        if not (is_complete and is_valid):
            return False
        
        # 检查PoseCheck数据
        pose_check = mol_data.get('pose_check') if isinstance(mol_data, dict) else getattr(mol_data, 'pose_check', None)
        if not is_posecheck_complete(pose_check):
            return False
        
        # 检查化学性质数据
        chem_results = mol_data.get('chem_results') if isinstance(mol_data, dict) else getattr(mol_data, 'chem_results', None)
        if not chem_results:
            return False
        
        # 检查PoseCheck指标是否在合理范围内
        clash = pose_check.get('clash')
        strain = pose_check.get('strain')
        
        if clash is None or strain is None:
            return False
        
        if clash >= 999.0 or strain >= 1e9:
            return False
        
        if clash < 0 or strain < 0:
            return False
        
        # 检查化学性质是否有效
        qed_val = chem_results.get('qed')
        sa_val = chem_results.get('sa')
        
        if qed_val is None or sa_val is None:
            return False
        
        if not (0 <= qed_val <= 1):
            return False
        
        if sa_val < 0:
            return False
        
        # 检查是否有vina数据
        vina_data = mol_data.get('vina') if isinstance(mol_data, dict) else getattr(mol_data, 'vina', None)
        if not vina_data:
            return False
        
        return True
        
    except Exception:
        return False

def filter_successful_molecules(all_data):
    """过滤出处理成功的分子"""
    successful_molecules = []
    failed_molecules = []
    
    for mol_data in all_data:
        if is_molecule_successful(mol_data):
            successful_molecules.append(mol_data)
        else:
            failed_molecules.append(mol_data)
    
    return successful_molecules, failed_molecules

def load_results_from_pt(pt_path):
    """从PT文件加载结果"""
    print(f"📁 加载PT文件: {os.path.basename(pt_path)}")
    file_size = os.path.getsize(pt_path) / (1024**3)
    print(f"📦 文件大小: {file_size:.2f} GB")
    
    with open(pt_path, 'rb') as f:
        try:
            from rdkit.Chem import rdchem
            torch.serialization.add_safe_globals([rdchem.Mol])
            original_data = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            f.seek(0)
            original_data = torch.load(f, map_location='cpu')
        
        if isinstance(original_data, dict) and 'all_results' in original_data:
            results = original_data['all_results']
            data_format = 'targetdiff'
            print(f"✅ TargetDiff格式，找到 {len(results):,} 个分子")
        elif isinstance(original_data, list):
            results = original_data
            data_format = 'list'
            print(f"✅ 列表格式，找到 {len(results):,} 个分子")
        else:
            results = []
            data_format = 'custom'
            for item in original_data:
                if hasattr(item, 'mol') or (isinstance(item, dict) and 'mol' in item):
                    results.append(item)
            print(f"✅ 自定义格式，提取了 {len(results):,} 个分子")
    
    # 分析PoseCheck状态
    sample_size = min(100, len(results))
    complete_count = 0
    incomplete_count = 0
    missing_count = 0
    param_counts = {}
    
    print(f"📋 分析PoseCheck状态 (前{sample_size}个样本):")
    
    for item in results[:sample_size]:
        pose_check = item.get('pose_check') if isinstance(item, dict) else getattr(item, 'pose_check', None)
        
        if pose_check is None:
            missing_count += 1
        else:
            param_count = len(pose_check)
            param_counts[param_count] = param_counts.get(param_count, 0) + 1
            
            if is_posecheck_complete(pose_check):
                complete_count += 1
            else:
                incomplete_count += 1
    
    print(f"   完整PoseCheck: {complete_count}")
    print(f"   不完整PoseCheck: {incomplete_count}")
    print(f"   缺失PoseCheck: {missing_count}")
    print(f"   参数数量分布: {param_counts}")
    
    return results, original_data, data_format

def process_molecule_batch_update_pt(batch_info):
    """处理分子批次并直接更新PT数据结构"""
    mol_data_batch, protein_root, batch_idx, force_recompute, debug_mode = batch_info
    
    # 为每个进程创建独立的PoseCheck实例
    pc = PoseCheck()
    current_protein = None
    batch_processed = 0
    batch_updated = 0
    batch_successful = 0
    updated_batch = []
    
    debug_first_few = debug_mode and batch_idx == 0
    
    for i, mol_data in enumerate(mol_data_batch):
        try:
            updated_mol_data = copy.deepcopy(mol_data)
            debug_this = debug_first_few and i < 5  # 调试前5个分子
            
            if debug_this:
                print(f"\n🔍 调试分子 {batch_idx * len(mol_data_batch) + i + 1}:")
            
            # 提取分子信息
            mol = None
            ligand_filename = None
            
            if isinstance(mol_data, dict):
                mol = mol_data.get('mol')
                ligand_filename = mol_data.get('ligand_filename', '')
            else:
                mol = getattr(mol_data, 'mol', None)
                ligand_filename = getattr(mol_data, 'ligand_filename', '')
            
            if mol is None:
                if debug_this:
                    print(f"  ❌ 没有分子对象")
                updated_batch.append(updated_mol_data)
                batch_processed += 1
                continue
            
            # 检查当前PoseCheck状态
            existing_pose_check = None
            existing_chem_results = None
            
            if isinstance(updated_mol_data, dict):
                existing_pose_check = updated_mol_data.get('pose_check')
                existing_chem_results = updated_mol_data.get('chem_results')
            else:
                existing_pose_check = getattr(updated_mol_data, 'pose_check', None)
                existing_chem_results = getattr(updated_mol_data, 'chem_results', None)
            
            if debug_this:
                print(f"  📝 配体文件: {ligand_filename}")
                print(f"  🧪 分子: {Chem.MolToSmiles(mol)[:50]}...")
                if existing_pose_check:
                    param_count = len(existing_pose_check)
                    print(f"  📊 现有PoseCheck参数数量: {param_count}")
                    print(f"  📊 PoseCheck字段: {list(existing_pose_check.keys())}")
                else:
                    print(f"  📊 无PoseCheck数据")
                print(f"  📊 有化学性质数据: {existing_chem_results is not None}")
            
            # 检查是否需要处理
            needs_processing = force_recompute
            
            if not force_recompute:
                pose_check_complete = is_posecheck_complete(existing_pose_check)
                has_chem_results = existing_chem_results is not None
                
                if not pose_check_complete or not has_chem_results:
                    needs_processing = True
                    if debug_this:
                        print(f"  🔄 需要处理: PoseCheck完整={pose_check_complete}, 有化学性质={has_chem_results}")
                else:
                    if debug_this:
                        print(f"  ✅ 跳过，已有完整数据")
            
            if not needs_processing:
                updated_batch.append(updated_mol_data)
                batch_processed += 1
                continue
            
            # 设置分子属性
            mol.SetProp('_Name', ligand_filename)
            
            # 添加vina分数
            try:
                vina_data = None
                if isinstance(mol_data, dict) and 'vina' in mol_data:
                    vina_data = mol_data['vina']
                elif hasattr(mol_data, 'vina'):
                    vina_data = mol_data.vina
                
                if vina_data:
                    if 'score_only' in vina_data and len(vina_data['score_only']) > 0:
                        mol.SetProp('vina_score', str(vina_data['score_only'][0]['affinity']))
                    if 'minimize' in vina_data and len(vina_data['minimize']) > 0:
                        mol.SetProp('vina_minimize', str(vina_data['minimize'][0]['affinity']))
                    if 'dock' in vina_data and len(vina_data['dock']) > 0:
                        mol.SetProp('vina_dock', str(vina_data['dock'][0]['affinity']))
            except Exception as e:
                if debug_this:
                    print(f"  ⚠️ 设置vina分数失败: {e}")
            
            # 检查分子有效性
            if not mol.HasProp('vina_score'):
                if debug_this:
                    print(f"  ❌ 没有vina_score属性")
                if isinstance(updated_mol_data, dict):
                    updated_mol_data['complete'] = False
                    updated_mol_data['validity'] = False
                else:
                    updated_mol_data.complete = False
                    updated_mol_data.validity = False
                updated_batch.append(updated_mol_data)
                batch_processed += 1
                continue
            
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                if debug_this:
                    print(f"  ❌ 分子包含多个片段")
                if isinstance(updated_mol_data, dict):
                    updated_mol_data['complete'] = False
                    updated_mol_data['validity'] = False
                else:
                    updated_mol_data.complete = False
                    updated_mol_data.validity = False
                updated_batch.append(updated_mol_data)
                batch_processed += 1
                continue
            
            # 加载蛋白质
            protein_loaded = False
            try:
                protein_fn = os.path.join(
                    protein_root,
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:10] + '.pdb'
                )
                
                if debug_this:
                    print(f"  🔄 蛋白质文件: {protein_fn}")
                    print(f"  🔍 文件存在: {os.path.exists(protein_fn)}")
                
                if protein_fn != current_protein:
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            pc.load_protein_from_pdb(protein_fn)
                    current_protein = protein_fn
                    protein_loaded = True
                    if debug_this:
                        print(f"  ✅ 蛋白质加载成功")
                else:
                    protein_loaded = True
                    if debug_this:
                        print(f"  ✅ 蛋白质已加载（缓存）")
                        
            except Exception as e:
                protein_loaded = False
                if debug_this:
                    print(f"  ❌ 蛋白质加载失败: {e}")
            
            # 计算PoseCheck指标
            posecheck_success = False
            clash = 999.0
            strain = 1e10
            
            if protein_loaded:
                try:
                    if debug_this:
                        print(f"  🔄 开始PoseCheck计算...")
                    
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            pc.load_ligands_from_mols([mol])
                            clash_results = pc.calculate_clashes()
                            strain_results = pc.calculate_strain_energy()
                            
                            clash = clash_results[0] if clash_results and len(clash_results) > 0 else 999.0
                            strain = strain_results[0] if strain_results and len(strain_results) > 0 else 1e10
                    
                    if debug_this:
                        print(f"  📊 PoseCheck原始结果: clash={clash}, strain={strain}")
                    
                    # 处理NaN
                    if clash != clash:  # NaN check
                        clash = 999.0
                    if strain != strain:  # NaN check
                        strain = 1e10
                    
                    # 验证结果是否合理
                    if clash < 999.0 and strain < 1e9 and clash >= 0 and strain >= 0:
                        posecheck_success = True
                        batch_successful += 1
                        if debug_this:
                            print(f"  ✅ PoseCheck成功: clash={clash:.3f}, strain={strain:.1f}")
                    else:
                        if debug_this:
                            print(f"  ❌ PoseCheck结果异常: clash={clash}, strain={strain}")
                    
                except Exception as e:
                    posecheck_success = False
                    if debug_this:
                        print(f"  ❌ PoseCheck计算异常: {e}")
            else:
                if debug_this:
                    print(f"  ❌ 蛋白质未加载，跳过PoseCheck")
            
            # 计算化学属性
            qed_val = 0.0
            sa_val = 0.0
            atom_num = 0
            
            try:
                qed_val = qed(mol)
                sa_val = compute_sa_score(mol)
                atom_num = mol.GetNumAtoms()
                if debug_this:
                    print(f"  🧮 化学属性: QED={qed_val:.3f}, SA={sa_val:.3f}, 原子数={atom_num}")
            except Exception as e:
                if debug_this:
                    print(f"  ⚠️ 化学属性计算失败: {e}")
            
            # 构建完整的PoseCheck数据结构（包含多个参数）
            pose_check_data = {
                'clash': float(clash),
                'strain': float(strain),
                'hb_acceptor': 0.0,
                'hb_donor': 0.0,
                'hydrophobic': 0.0,
                'vdw': 0.0
            }
            
            chem_results_data = {
                'qed': float(qed_val),
                'sa': float(sa_val),
                'atom_num': int(atom_num)
            }
            
            if debug_this:
                print(f"  📊 最终PoseCheck参数数量: {len(pose_check_data)}")
                print(f"  📊 PoseCheck字段: {list(pose_check_data.keys())}")
            
            # 更新数据结构
            if isinstance(updated_mol_data, dict):
                updated_mol_data['pose_check'] = pose_check_data
                updated_mol_data['chem_results'] = chem_results_data
                updated_mol_data['complete'] = True
                updated_mol_data['validity'] = posecheck_success
            else:
                updated_mol_data.pose_check = pose_check_data
                updated_mol_data.chem_results = chem_results_data
                updated_mol_data.complete = True
                updated_mol_data.validity = posecheck_success
            
            batch_updated += 1
            updated_batch.append(updated_mol_data)
            batch_processed += 1
            
            # 内存清理
            if i % 10 == 0:
                gc.collect()
            
        except Exception as e:
            if debug_first_few and i < 5:
                print(f"  ❌ 处理异常: {str(e)}")
            
            try:
                failed_mol_data = copy.deepcopy(mol_data)
                if isinstance(failed_mol_data, dict):
                    failed_mol_data['complete'] = False
                    failed_mol_data['validity'] = False
                else:
                    failed_mol_data.complete = False
                    failed_mol_data.validity = False
                updated_batch.append(failed_mol_data)
            except:
                updated_batch.append(copy.deepcopy(mol_data))
            batch_processed += 1
            continue
    
    # 清理
    del pc
    gc.collect()
    
    return updated_batch, batch_updated, batch_processed, batch_successful, batch_idx

def save_updated_pt_data(successful_molecules, original_data, data_format, output_pt):
    """保存更新后的PT数据 - 只保存成功的分子"""
    try:
        print(f"💾 保存PT文件（只包含成功分子）...")
        
        if data_format == 'targetdiff':
            output_data = copy.deepcopy(original_data)
            output_data['all_results'] = successful_molecules
        elif data_format == 'list':
            output_data = successful_molecules
        else:
            output_data = successful_molecules
        
        torch.save(output_data, output_pt)
        
        file_size = os.path.getsize(output_pt) / (1024**3)
        print(f"✅ 文件保存成功: {os.path.basename(output_pt)} ({file_size:.2f} GB)")
        print(f"📊 保存的成功分子数: {len(successful_molecules):,}")
        
        return True, "保存成功"
        
    except Exception as e:
        return False, f"保存失败: {str(e)}"

def save_failed_molecules_report(failed_molecules, output_dir):
    """保存失败分子的报告"""
    try:
        if not failed_molecules:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'failed_molecules_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"失败分子报告\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"总失败分子数: {len(failed_molecules)}\n\n")
            
            # 统计失败原因
            reasons = {}
            for mol_data in failed_molecules:
                if isinstance(mol_data, dict):
                    complete = mol_data.get('complete', False)
                    validity = mol_data.get('validity', False)
                    pose_check = mol_data.get('pose_check')
                    chem_results = mol_data.get('chem_results')
                else:
                    complete = getattr(mol_data, 'complete', False)
                    validity = getattr(mol_data, 'validity', False)
                    pose_check = getattr(mol_data, 'pose_check', None)
                    chem_results = getattr(mol_data, 'chem_results', None)
                
                if not complete:
                    reason = "分子不完整"
                elif not validity:
                    reason = "PoseCheck验证失败"
                elif not is_posecheck_complete(pose_check):
                    reason = "PoseCheck数据不完整"
                elif not chem_results:
                    reason = "缺少化学性质数据"
                else:
                    reason = "其他原因"
                
                reasons[reason] = reasons.get(reason, 0) + 1
            
            f.write("失败原因统计:\n")
            for reason, count in reasons.items():
                f.write(f"  {reason}: {count} ({count/len(failed_molecules)*100:.1f}%)\n")
        
        print(f"📝 失败分子报告已保存: {report_path}")
        
    except Exception as e:
        print(f"⚠️ 保存失败分子报告时出错: {e}")

def calculate_final_statistics(all_updated_data, successful_molecules, failed_molecules):
    """计算最终统计指标"""
    print("\n" + "="*60)
    print("📊 最终统计结果")
    
    total_molecules = len(all_updated_data)
    successful_count = len(successful_molecules)
    failed_count = len(failed_molecules)
    
    print(f"📈 处理结果:")
    print(f"   总分子数: {total_molecules:,}")
    print(f"   成功分子: {successful_count:,} ({successful_count/total_molecules*100:.1f}%)")
    print(f"   失败分子: {failed_count:,} ({failed_count/total_molecules*100:.1f}%)")
    print()
    
    if not successful_molecules:
        print("⚠️ 没有成功处理的分子")
        return
    
    # BF和SR统计
    bf_total = 0
    bf_success = 0
    sr_total = 0
    sr_success = 0
    
    # 指标统计
    vina_scores = []
    vina_docks = []
    clashes = []
    strains = []
    qeds = []
    sas = []
    
    # PoseCheck状态统计
    complete_posecheck = 0
    param_counts = {}
    
    for mol_data in successful_molecules:
        # PoseCheck状态统计
        pose_check = mol_data.get('pose_check') if isinstance(mol_data, dict) else getattr(mol_data, 'pose_check', None)
        if pose_check:
            param_count = len(pose_check)
            param_counts[param_count] = param_counts.get(param_count, 0) + 1
            
            if is_posecheck_complete(pose_check):
                complete_posecheck += 1
        
        # 提取指标
        vina_score = None
        vina_dock = None
        strain = None
        clash = None
        qed_val = None
        sa_val = None
        
        # 从vina数据提取
        vina_data = mol_data.get('vina') if isinstance(mol_data, dict) else getattr(mol_data, 'vina', None)
        if vina_data:
            if 'score_only' in vina_data and len(vina_data['score_only']) > 0:
                vina_score = vina_data['score_only'][0]['affinity']
                vina_scores.append(vina_score)
            if 'dock' in vina_data and len(vina_data['dock']) > 0:
                vina_dock = vina_data['dock'][0]['affinity']
                vina_docks.append(vina_dock)
        
        # 从pose_check数据提取
        if pose_check and is_posecheck_complete(pose_check):
            strain = pose_check.get('strain')
            clash = pose_check.get('clash')
            if strain is not None and strain < 1e9:
                strains.append(strain)
            if clash is not None and clash < 999:
                clashes.append(clash)
        
        # 从chem_results数据提取
        chem_results = mol_data.get('chem_results') if isinstance(mol_data, dict) else getattr(mol_data, 'chem_results', None)
        if chem_results:
            qed_val = chem_results.get('qed')
            sa_val = chem_results.get('sa')
            if qed_val is not None:
                qeds.append(qed_val)
            if sa_val is not None:
                sas.append(sa_val)
        
        # BF计算
        if vina_score is not None and strain is not None and strain < 1e9:
            bf_total += 1
            if vina_score < -2.49 and strain < 836:
                bf_success += 1
        
        # SR计算
        if vina_dock is not None and qed_val is not None and sa_val is not None:
            sr_total += 1
            if vina_dock < -8.18 and qed_val > 0.25 and sa_val > 0.59:
                sr_success += 1
    
    # 计算比例
    bf_rate = (bf_success / bf_total * 100) if bf_total > 0 else 0
    sr_rate = (sr_success / sr_total * 100) if sr_total > 0 else 0
    
    print(f"📊 成功分子的PoseCheck状态:")
    print(f"   完整PoseCheck: {complete_posecheck:,} ({complete_posecheck/successful_count*100:.1f}%)")
    print(f"   参数数量分布: {param_counts}")
    print()
    
    print(f"🎯 关键指标（基于成功分子）:")
    print(f"   BF (结合可行性): {bf_success:,}/{bf_total:,} = {bf_rate:.2f}%")
    print(f"   SR (成功率): {sr_success:,}/{sr_total:,} = {sr_rate:.2f}%")
    print()
    
    # 基础统计
    if vina_scores:
        print(f"📊 指标统计（成功分子）:")
        print(f"   Vina Score: μ={np.mean(vina_scores):.3f}, M={np.median(vina_scores):.3f} (n={len(vina_scores):,})")
        if vina_docks:
            print(f"   Vina Dock:  μ={np.mean(vina_docks):.3f}, M={np.median(vina_docks):.3f} (n={len(vina_docks):,})")
        if clashes:
            print(f"   Clash:      μ={np.mean(clashes):.3f}, M={np.median(clashes):.3f} (n={len(clashes):,})")
        if strains:
            print(f"   Strain:     μ={np.mean(strains):.1f}, M={np.median(strains):.1f} (n={len(strains):,})")
        if qeds:
            print(f"   QED:        μ={np.mean(qeds):.3f}, M={np.median(qeds):.3f} (n={len(qeds):,})")
        if sas:
            print(f"   SA:         μ={np.mean(sas):.3f}, M={np.median(sas):.3f} (n={len(sas):,})")

def main():
    parser = argparse.ArgumentParser(description='PoseCheck处理PT文件 - 只保存成功分子版本')
    parser.add_argument('--pt_file', type=str, required=True, help='输入PT文件路径')
    parser.add_argument('--output_pt', type=str, required=True, help='输出PT文件路径')
    parser.add_argument('--protein_root', type=str, default='./data/test_set', help='蛋白质文件根目录')
    parser.add_argument('--n_workers', type=int, default=None, help='并行工作进程数')
    parser.add_argument('--batch_size', type=int, default=50, help='每个进程处理的分子数')
    parser.add_argument('--max_molecules', type=int, default=None, help='最大处理分子数')
    parser.add_argument('--backup', action='store_true', help='创建输入文件备份')
    parser.add_argument('--force_recompute', action='store_true', help='强制重新计算所有PoseCheck')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--quiet', action='store_true', help='更安静的输出')
    parser.add_argument('--save_failed_report', action='store_true', help='保存失败分子报告')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("=" * 60)
        print("🧬 PoseCheck处理器 - 只保存成功分子版本")
        print("=" * 60)
        print(f"⏰ 当前时间: 2025-06-05 04:37:49")
        print(f"👤 用户: xulong0826")
        print(f"📥 输入: {os.path.basename(args.pt_file)}")
        print(f"📤 输出: {os.path.basename(args.output_pt)}")
        print(f"💾 内存: {get_memory_usage():.0f} MB")
        if args.debug:
            print("🐛 调试模式启用")
        if args.force_recompute:
            print("🔄 强制重新计算")
        if args.save_failed_report:
            print("📝 将保存失败分子报告")
        print("⚠️  注意：只会保存处理成功的分子到输出文件")
        print()
    
    # 检查输入文件
    if not os.path.exists(args.pt_file):
        print(f"❌ 错误: 输入PT文件不存在: {args.pt_file}")
        return
    
    # 检查蛋白质根目录
    if not os.path.exists(args.protein_root):
        print(f"❌ 错误: 蛋白质根目录不存在: {args.protein_root}")
        return
    
    # 创建备份
    if args.backup:
        backup_path = args.pt_file + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(args.pt_file, backup_path)
            if not args.quiet:
                print(f"💾 已创建备份: {os.path.basename(backup_path)}")
    
    # 加载PT文件
    results, original_data, data_format = load_results_from_pt(args.pt_file)
    
    # 应用最大分子数限制
    if args.max_molecules and args.max_molecules < len(results):
        results = results[:args.max_molecules]
        print(f"⚠️  限制为前 {args.max_molecules:,} 个分子")
    
    # 设置工作进程数
    if args.n_workers is None:
        args.n_workers = min(mp.cpu_count() // 3, 4)
    
    if not args.quiet:
        print(f"⚙️  并行配置: {args.n_workers} 进程 × {args.batch_size} 分子/批次")
        print(f"🗂️  蛋白质目录: {args.protein_root}")
    
    # 分批准备任务
    batches = []
    for i in range(0, len(results), args.batch_size):
        batch_data = results[i:i+args.batch_size]
        batches.append((batch_data, args.protein_root, i // args.batch_size, args.force_recompute, args.debug))
    
    if not args.quiet:
        print(f"📦 分批处理: {len(batches):,} 个批次")
        print()
    
    # 并行处理
    completed_batches = 0
    total_updated = 0
    total_processed = 0
    total_successful = 0
    all_updated_data = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        future_to_batch = {executor.submit(process_molecule_batch_update_pt, batch): batch for batch in batches}
        
        desc = "🔬 PoseCheck处理" if not args.quiet else "处理中"
        with tqdm(total=len(batches), desc=desc, unit="batch", disable=args.quiet) as pbar:
            for future in as_completed(future_to_batch):
                try:
                    updated_batch, batch_updated, batch_processed, batch_successful, batch_idx = future.result()
                    
                    # 按顺序合并结果
                    all_updated_data.extend(updated_batch)
                    total_updated += batch_updated
                    total_processed += batch_processed
                    total_successful += batch_successful
                    completed_batches += 1
                    
                    pbar.update(1)
                    
                    if not args.quiet:
                        elapsed_time = time.time() - start_time
                        if completed_batches > 0:
                            avg_time_per_batch = elapsed_time / completed_batches
                            remaining_batches = len(batches) - completed_batches
                            eta = remaining_batches * avg_time_per_batch
                            
                            pbar.set_postfix({
                                'updated': f'{batch_updated}/{batch_processed}',
                                'success': batch_successful,
                                'total': f'{total_updated:,}',
                                'mem': f'{get_memory_usage():.0f}MB',
                                'ETA': f'{eta/60:.1f}min'
                            })
                    
                except Exception as e:
                    completed_batches += 1
                    pbar.update(1)
                    if not args.quiet:
                        print(f"批次处理失败: {e}")
    
    elapsed_time = time.time() - start_time
    
    if not args.quiet:
        print(f"\n✅ 处理完成!")
        print(f"   耗时: {elapsed_time/60:.1f} 分钟")
        print(f"   更新: {total_updated:,}/{total_processed:,} 个分子 ({total_updated/total_processed*100:.1f}%)")
        print(f"   PoseCheck成功: {total_successful:,} 个分子 ({total_successful/total_processed*100:.1f}%)")
        print(f"   速度: {total_updated/(elapsed_time/60):.0f} 分子/分钟")
    
    # 过滤成功和失败的分子
    print(f"\n🔍 过滤处理结果...")
    successful_molecules, failed_molecules = filter_successful_molecules(all_updated_data)
    
    print(f"📊 过滤结果:")
    print(f"   成功分子: {len(successful_molecules):,}")
    print(f"   失败分子: {len(failed_molecules):,}")
    print(f"   成功率: {len(successful_molecules)/len(all_updated_data)*100:.1f}%")
    
    # 保存失败分子报告
    if args.save_failed_report and failed_molecules:
        output_dir = os.path.dirname(args.output_pt) or '.'
        save_failed_molecules_report(failed_molecules, output_dir)
    
    # 保存更新后的PT文件（只包含成功分子）
    if successful_molecules:
        success, message = save_updated_pt_data(successful_molecules, original_data, data_format, args.output_pt)
        
        if success:
            # 计算最终统计指标
            calculate_final_statistics(all_updated_data, successful_molecules, failed_molecules)
            
            if not args.quiet:
                print(f"\n💾 最终内存使用: {get_memory_usage():.0f} MB")
                print("=" * 60)
            
        else:
            print(f"❌ {message}")
    else:
        print("❌ 没有成功处理的分子，不保存输出文件")

if __name__ == "__main__":
    main()