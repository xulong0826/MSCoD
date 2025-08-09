import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from openbabel import openbabel as ob
from spyrmsd import rmsd, molecule
import copy
import warnings
import gc
import psutil
from datetime import datetime

# 禁用警告
warnings.filterwarnings('ignore')
ob.obErrorLog.SetOutputLevel(0)

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_symmetry_rmsd(mol, ref):
    """计算对称RMSD"""
    try:
        mol_spyrmsd = molecule.Molecule.from_rdkit(mol)
        ref_spyrmsd = molecule.Molecule.from_rdkit(ref)
        coords_ref = ref_spyrmsd.coordinates
        anum_ref = ref_spyrmsd.atomicnums
        adj_ref = ref_spyrmsd.adjacency_matrix
        coords = mol_spyrmsd.coordinates
        anum = mol_spyrmsd.atomicnums
        adj = mol_spyrmsd.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords_ref,
            coords,
            anum_ref,
            anum,
            adj_ref,
            adj,
        )
        return RMSD
    except Exception as e:
        raise e

def get_rmsd(gen_mol, dock_mol):
    """计算简单RMSD"""
    gen_pose = gen_mol.GetConformer().GetPositions()
    dock_pose = dock_mol.GetConformer().GetPositions()
    return np.sqrt(np.mean(np.sum((gen_pose - dock_pose)**2, axis=1)))

def get_pdbqt_mol(pdbqt_block: str, temp_dir="tmp") -> Chem.Mol:
    """将PDBQT块转换为RDKit分子"""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 生成随机文件名
    random_name = np.random.randint(0, 1000000)
    pdbqt_name = os.path.join(temp_dir, f"test_pdbqt_{random_name}.pdbqt")
    
    try:
        # 写入PDBQT文件
        with open(pdbqt_name, "w") as f:
            f.write(pdbqt_block)

        # 读取PDBQT文件
        mol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("pdbqt", "pdb")
        obConversion.ReadFile(mol, pdbqt_name)

        # 转换为RDKit
        mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

        return mol
    
    finally:
        # 清理临时文件
        if os.path.exists(pdbqt_name):
            os.remove(pdbqt_name)

def calculate_rmsd_for_molecule(mol_data, temp_dir="tmp", debug=False):
    """为单个分子计算RMSD"""
    try:
        # 提取分子和对接姿态
        if isinstance(mol_data, dict):
            mol = mol_data.get('mol')
            vina_data = mol_data.get('vina', {})
        else:
            mol = getattr(mol_data, 'mol', None)
            vina_data = getattr(mol_data, 'vina', {})
        
        if mol is None:
            if debug:
                print("    ❌ 没有分子对象")
            return None, "没有分子对象"
        
        # 检查是否有dock数据
        if not vina_data or 'dock' not in vina_data or len(vina_data['dock']) == 0:
            if debug:
                print("    ❌ 没有dock数据")
            return None, "没有dock数据"
        
        # 获取对接姿态
        dock_data = vina_data['dock'][0]
        if 'pose' not in dock_data:
            if debug:
                print("    ❌ 没有pose数据")
            return None, "没有pose数据"
        
        docked_pdbqt = dock_data['pose']
        
        # 转换PDBQT为分子
        docked_mol = get_pdbqt_mol(docked_pdbqt, temp_dir)
        if docked_mol is None:
            if debug:
                print("    ❌ PDBQT转换失败")
            return None, "PDBQT转换失败"
        
        # 去除氢原子
        mol_clean = Chem.RemoveAllHs(mol)
        docked_mol_clean = Chem.RemoveAllHs(docked_mol)
        
        if mol_clean is None or docked_mol_clean is None:
            if debug:
                print("    ❌ 去氢失败")
            return None, "去氢失败"
        
        # 首先尝试对称RMSD
        try:
            rmsd_val = get_symmetry_rmsd(docked_mol_clean, mol_clean)
            if debug:
                print(f"    ✅ 对称RMSD成功: {rmsd_val:.3f}")
            return rmsd_val, "对称RMSD成功"
        except Exception as e:
            if debug:
                print(f"    ⚠️ 对称RMSD失败: {e}")
        
        # 回退到简单RMSD
        try:
            rmsd_val = get_rmsd(mol_clean, docked_mol_clean)
            if debug:
                print(f"    ✅ 简单RMSD成功: {rmsd_val:.3f}")
            return rmsd_val, "简单RMSD成功"
        except Exception as e:
            if debug:
                print(f"    ❌ 简单RMSD失败: {e}")
            return None, f"简单RMSD失败: {e}"
    
    except Exception as e:
        if debug:
            print(f"    ❌ 计算异常: {e}")
        return None, f"计算异常: {e}"

def process_pt_file_with_rmsd(input_pt, output_pt, temp_dir="tmp", debug=False, max_molecules=None):
    """处理PT文件，添加RMSD计算"""
    
    print("=" * 60)
    print("🧬 PT文件RMSD计算器")
    print("=" * 60)
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 用户: xulong0826")
    print(f"📥 输入: {os.path.basename(input_pt)}")
    print(f"📤 输出: {os.path.basename(output_pt)}")
    print(f"💾 初始内存: {get_memory_usage():.0f} MB")
    print()
    
    # 检查输入文件
    if not os.path.exists(input_pt):
        print(f"❌ 错误: 输入文件不存在: {input_pt}")
        return False
    
    # 创建临时目录
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 加载数据
    print("📁 加载PT文件...")
    file_size = os.path.getsize(input_pt) / (1024**3)
    print(f"📦 文件大小: {file_size:.2f} GB")
    
    try:
        # 尝试安全加载
        try:
            from rdkit.Chem import rdchem
            torch.serialization.add_safe_globals([rdchem.Mol])
            original_data = torch.load(input_pt, map_location='cpu', weights_only=False)
        except:
            original_data = torch.load(input_pt, map_location='cpu')
        
        # 确定数据格式
        if isinstance(original_data, dict) and 'all_results' in original_data:
            results = original_data['all_results']
            data_format = 'targetdiff'
            print(f"✅ TargetDiff格式，找到 {len(results):,} 个分子")
        elif isinstance(original_data, list):
            results = original_data
            data_format = 'list'
            print(f"✅ 列表格式，找到 {len(results):,} 个分子")
        else:
            print("❌ 不支持的数据格式")
            return False
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 应用最大分子数限制
    if max_molecules and max_molecules < len(results):
        results = results[:max_molecules]
        print(f"⚠️  限制为前 {max_molecules:,} 个分子")
    
    # 统计变量
    total_molecules = len(results)
    processed = 0
    success_count = 0
    symmetric_count = 0
    simple_count = 0
    failed_count = 0
    already_has_rmsd = 0
    
    rmsds = []
    updated_results = []
    
    print(f"🔬 开始RMSD计算...")
    print()
    
    # 处理每个分子
    with tqdm(results, desc="计算RMSD", unit="mol") as pbar:
        for i, mol_data in enumerate(pbar):
            try:
                # 创建副本
                updated_mol_data = copy.deepcopy(mol_data)
                
                debug_this = debug and i < 5  # 只调试前5个
                
                if debug_this:
                    print(f"\n🔍 调试分子 {i+1}:")
                
                # 检查是否已有RMSD
                existing_rmsd = None
                if isinstance(mol_data, dict):
                    existing_rmsd = mol_data.get('rmsd')
                else:
                    existing_rmsd = getattr(mol_data, 'rmsd', None)
                
                if existing_rmsd is not None:
                    already_has_rmsd += 1
                    rmsds.append(existing_rmsd)
                    if debug_this:
                        print(f"  ✅ 已有RMSD: {existing_rmsd:.3f}")
                    updated_results.append(updated_mol_data)
                    processed += 1
                    success_count += 1
                    continue
                
                # 计算RMSD
                rmsd_val, message = calculate_rmsd_for_molecule(mol_data, temp_dir, debug_this)
                
                if rmsd_val is not None:
                    # 成功计算RMSD
                    if isinstance(updated_mol_data, dict):
                        updated_mol_data['rmsd'] = float(rmsd_val)
                    else:
                        updated_mol_data.rmsd = float(rmsd_val)
                    
                    rmsds.append(rmsd_val)
                    success_count += 1
                    
                    if "对称RMSD" in message:
                        symmetric_count += 1
                    elif "简单RMSD" in message:
                        simple_count += 1
                    
                    if debug_this:
                        print(f"  ✅ RMSD计算成功: {rmsd_val:.3f} ({message})")
                
                else:
                    # 计算失败
                    failed_count += 1
                    if debug_this:
                        print(f"  ❌ RMSD计算失败: {message}")
                
                updated_results.append(updated_mol_data)
                processed += 1
                
                # 更新进度
                pbar.set_postfix({
                    'success': success_count,
                    'failed': failed_count,
                    'mem': f'{get_memory_usage():.0f}MB'
                })
                
                # 内存清理
                if i % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                if debug and i < 5:
                    print(f"  ❌ 处理异常: {e}")
                
                # 保存原始数据
                updated_results.append(copy.deepcopy(mol_data))
                processed += 1
                failed_count += 1
                continue
    
    print(f"\n✅ RMSD计算完成!")
    print(f"   处理分子: {processed:,}/{total_molecules:,}")
    print(f"   成功计算: {success_count:,} ({success_count/total_molecules*100:.1f}%)")
    print(f"   已有RMSD: {already_has_rmsd:,}")
    print(f"   对称RMSD: {symmetric_count:,}")
    print(f"   简单RMSD: {simple_count:,}")
    print(f"   计算失败: {failed_count:,}")
    
    # RMSD统计
    if rmsds:
        rmsd_array = np.array(rmsds)
        rmsd_lt_2 = np.sum(rmsd_array < 2.0)
        print(f"\n📊 RMSD统计:")
        print(f"   平均值: {np.mean(rmsd_array):.3f}")
        print(f"   中位数: {np.median(rmsd_array):.3f}")
        print(f"   Q25/Q75: {np.quantile(rmsd_array, 0.25):.3f}/{np.quantile(rmsd_array, 0.75):.3f}")
        print(f"   RMSD < 2Å: {rmsd_lt_2:,}/{len(rmsds):,} ({rmsd_lt_2/len(rmsds)*100:.1f}%)")
    
    # 保存结果
    print(f"\n💾 保存结果...")
    try:
        if data_format == 'targetdiff':
            output_data = copy.deepcopy(original_data)
            output_data['all_results'] = updated_results
        else:
            output_data = updated_results
        
        torch.save(output_data, output_pt)
        
        output_size = os.path.getsize(output_pt) / (1024**3)
        print(f"✅ 文件保存成功: {os.path.basename(output_pt)} ({output_size:.2f} GB)")
        
        # 保存RMSD数组
        rmsd_file = output_pt.replace('.pt', '_rmsds.npy')
        if rmsds:
            np.save(rmsd_file, np.array(rmsds))
            print(f"📊 RMSD数组已保存: {os.path.basename(rmsd_file)}")
        
        print(f"💾 最终内存使用: {get_memory_usage():.0f} MB")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='为PT文件添加RMSD计算')
    parser.add_argument('--input', type=str, required=True, help='输入PT文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出PT文件路径')
    parser.add_argument('--temp_dir', type=str, default='tmp', help='临时文件目录')
    parser.add_argument('--max_molecules', type=int, default=None, help='最大处理分子数')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 执行处理
    success = process_pt_file_with_rmsd(
        input_pt=args.input,
        output_pt=args.output,
        temp_dir=args.temp_dir,
        debug=args.debug,
        max_molecules=args.max_molecules
    )
    
    if success:
        print("🎉 处理完成!")
    else:
        print("❌ 处理失败!")

if __name__ == "__main__":
    main()