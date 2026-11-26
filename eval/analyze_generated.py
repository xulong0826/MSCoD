import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Reuse the exact evaluation pipeline from sample_for_pocket.py
from pathlib import Path

import sample_for_pocket as sfp


def main():
    ap = argparse.ArgumentParser(description='Evaluate generated SDFs in a directory (reuses sample_for_pocket Metrics)')
    ap.add_argument('--sdf_dir', required=True, help='directory containing 0.sdf, 1.sdf, ...')
    ap.add_argument('--protein', required=True)
    ap.add_argument('--ref_ligand', required=True)
    ap.add_argument('--max_samples', type=int, default=100)
    args = ap.parse_args()

    sdf_dir = os.path.abspath(args.sdf_dir)
    all_metrics = []
    for i in range(args.max_samples):
        out_fn = os.path.join(sdf_dir, f'{i}.sdf')
        if os.path.exists(out_fn):
            try:
                metrics = sfp.Metrics(args.protein, args.ref_ligand, out_fn).evaluate()
                metrics['sample_id'] = i
                all_metrics.append(metrics)
                print(f"Sample {i}: 完成评估", flush=True)
            except Exception as e:
                print(f"Sample {i}: 评估失败 - {e}", flush=True)
        else:
            print(f"Sample {i}: 文件不存在", flush=True)

    print(f"\n总评估样本数: {len(all_metrics)}", flush=True)
    if all_metrics:
        stats_results = sfp.analyze_all_metrics(all_metrics)
        if stats_results:
            sfp.all_metrics = all_metrics  # print_key_insights reads this module global
            sfp.print_key_insights(stats_results)
            prefix = f'docking_analysis_{Path(args.ref_ligand).stem}'
            sfp.save_results(all_metrics, stats_results, prefix=prefix, out_dir=sdf_dir)
            print(f"\n分析完成！共分析了 {len(all_metrics)} 个分子的 {len(stats_results)} 项指标。", flush=True)
    else:
        print("没有成功评估任何分子，请检查文件路径和格式。", flush=True)


if __name__ == '__main__':
    main()
