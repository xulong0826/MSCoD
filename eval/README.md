# eval/ — 敏感性分析与评估脚本

对应论文 **补充材料 Sec. 5（消融与敏感性分析）** 与 **Sec. 8（逐口袋 Vina Dock 分布）**。

## 脚本 ↔ 论文对照

| 脚本 | 产出 | 对应论文 |
|---|---|---|
| `msfp_sensitivity.sh` | 一键跑完整 MSFP leave-one-out 流水线 | SM Sec 5.2 |
| `msfp_eval_light.py` | Vina Score/Min + QED/SA（快速，81 公共口袋） | **Table S4** 的 Vina Score/QED/SA 列 |
| `msfp_dock_rmsd.py` | Vina Dock + 生成-对接 RMSD（姿态稳定性） | **Table S4** 的 Vina Dock/RMSD 列 |
| `msfp_clash.py` | PoseCheck clash（89 口袋，可断点续跑） | **Table S4** 的 Clash 列 |
| `msfp_posecheck.py` | 完整 PoseCheck（strain + clash + 相互作用） | **Table S5/S6**（7XKJ） |
| `msfp_probe_batch.py` | MSFP 瓶颈表示探针（region/element 标签解码） | SM Sec 5.2 表示探针 |
| `msfp_sensitivity_full.py` | 全量逐口袋敏感性聚合 | **Table S4** |
| `msfp_eval_only.py` | Vina Score + QED/SA + clash（旧版） | **Table S4** |

## 数据（中间结果）

`eval/7xkj/output/msfp_sensitivity_full/{pocket}_{cfg}/`：
- `docking_analysis_light.json` — Vina Score/Min + QED/SA/logP
- `dock_rmsd_stats.json` — Vina Dock + RMSD<2Å
- `0-9.sdf` — 生成的 10 个分子/口袋/配置

## 复现

```bash
bash eval/msfp_sensitivity.sh
```

> 注意：Table S4 是**配对分析**，只保留四个配置都完成的公共口袋（对接类 81 个、clash 89 个），每口袋 10 分子；绝对数值与 Table I（100 口袋 × 100 分子）不可直接比较，只看四配置间的配对差异。
