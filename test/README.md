# test/ — 标准评估脚本

对应论文 **Table I（主对比表）** 的指标计算。

## 脚本 ↔ 论文对照

| 脚本 | 产出 | 对应论文 |
|---|---|---|
| `eval_dock_vina.py` | AutoDock Vina（Score / Min / Dock） | Table I 的 Vina 三列 |
| `eval_posecheck.py` | PoseCheck（strain energy / clash） | Table I 的 SE / Clash 列 |
| `eval_rmsd.py` | 对称校正 RMSD | Table I 的 RMSD 列 |
| `eval_utils.py` | 共享评估辅助（含 Diversity、QED、SA） | Table I 的 QED/SA/Div 列 |
| `eval_distribution.ipynb` | 属性分布分析 notebook | Fig. S1–S3 辅助 |

## 数据

CrossDocked2020 `test_set/`（100 口袋 × 100 分子）。
