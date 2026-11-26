# eval/caculate_all_metrics/ — 全指标后处理

对应论文 **Table I（最终指标汇总）** 与 **SM Sec. 8（Table S8 逐口袋 Vina Dock 分布）**。

## 流程

1. `1_add_rmsd_pt.py` — 给 `vina_docked.pt` 加 RMSD 字段
2. `2_add_pose_pt.py` — 加 PoseCheck（clash/strain）字段 → `vina_docked_pose_checked.pt`
3. `3_all_metrics.py` — 汇总所有指标（Vina/QED/SA/Div/SE/Clash/RMSD）

## 脚本 ↔ 论文对照

| 脚本 | 产出 | 对应论文 |
|---|---|---|
| `1_add_rmsd_pt.py` | RMSD 字段 | Table I 的 RMSD |
| `2_add_pose_pt.py` | PoseCheck 字段（clash/strain/hb_*） | Table I 的 SE/Clash |
| `3_all_metrics.py` | 全指标汇总 + 逐口袋 Vina Dock 分布 | Table I + **Table S8** |

## 数据

- 输入：`vina_docked.pt`（Vina 评估结果）
- 输出：`checkpoints/mscod_vina_docked_pose_checked.pt`（91 口袋、9443 分子）