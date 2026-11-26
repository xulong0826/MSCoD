# 论文图/表 ↔ 代码 ↔ 数据 对应关系

本文档把论文中的每一张图、每一个表映射到本仓库里的生成脚本与数据来源，方便复现与审计。

> 说明：Fig. 1、Fig. 2 是架构示意图（MSCoD 框架 / MSFP+MHCA 模块），由绘图工具制作，不涉及代码；其余图表均可用下述脚本复现。

---

## 主文（Main Text）

### 图（Figures）

| 编号 | 内容 | 生成脚本 | 数据来源 |
|---|---|---|---|
| Fig. 1 | MSCoD 框架总览（Sender/Receiver/Update） | 示意图（非代码） | — |
| Fig. 2 | MSFP 与 MHCA 模块架构 | 示意图（非代码） | — |
| Fig. 3 | 8 个测试口袋的对接视图（口袋表面/配体/接触残基/氢键） | `visualization/fig3_export.py` → `fig3_render.py` → `fig3_label_overlay.py` → `fig3_montage.py` | 各口袋生成的 SDF + 口袋 PDB（`test_set/`） |
| Fig. 4 | KRAS G12D (7XKJ) 两口袋的属性小提琴图（QED/SA/Vina Score/Min/Dock/SE/Clash） | `eval/7xkj/view.py` | `eval/7xkj/output/{mscod,molcraft}_{gdp,6ic}/docking_analysis_*_raw_data.json` |
| Fig. 5 | MHCA 配体→蛋白注意力热图 + 区域图例 | `eval/7xkj/attention_heatmap.py` + `eval/7xkj/make_region_legend.py` | 模型 `co_attention._last_attentions`（7XKJ 两口袋） |

### 表（Tables）

| 编号 | 内容 | 生成脚本 | 数据来源 |
|---|---|---|---|
| Table I | 主对比表（Reference + 10 个基线 + MSCoD，Vina/SE/Clash/RMSD/QED/SA/Div/BF/SR） | `test/eval_dock_vina.py` + `test/eval_posecheck.py` + `test/eval_rmsd.py` + `test/eval_utils.py` | CrossDocked2020 `test_set/`（100 口袋，100 分子/口袋） |
| Table II | MSFP / MHCA 组件消融（Baseline、w/o MHCA、w/o MSFP、MSCoD） | 消融模型分别采样 + 同 Table I 的评估 | 消融 checkpoint（w/o MSFP / w/o MHCA） |
| Table III | DTA 预测（Davis / Kiba，GraphDTA 四 backbone + MSCoD block） | DTA 训练/评估脚本（GraphDTA 变体） | Davis / Kiba 数据集 |

---

## 补充材料（Supplementary Material）

### 图（Figures）

| 编号 | 内容 | 生成脚本 |
|---|---|---|
| Fig. S1 | 键长分布（BL.png） | `core/evaluation/utils/eval_bond_length.py` |
| Fig. S2 | 键角分布（BA1/2/3.png） | `core/evaluation/utils/eval_bond_angle.py` |
| Fig. S3 | 二面角分布（TA1/2.png） | `core/evaluation/utils/eval_torsion_angle.py` |
| Fig. S4 | MSFP 三个瓶颈表示的主成分分析（按 KRAS 结构域着色，GDP/6IC） | `eval/7xkj/msfp_scale_viz.py` |
| Fig. S5 | 区域注意力份额对比（GDP vs 6IC） | `eval/7xkj/attention_region_compare.py` |

### 表（Tables）

| 编号 | 内容 | 生成脚本 | 数据来源 |
|---|---|---|---|
| Table S1 | 可训练参数量（MolCRAFT 2.82M → MSCoD 3.20M，+379K/+13.4%） | `eval/benchmark_inference.py`（或直接 `torch` 统计 state_dict） | `checkpoints/mscod_last.ckpt` |
| Table S2 | 推理 / 训练 profile（135 s/100 分子、+4.9% 开销、25 epochs/5 天） | `sample_for_pocket.py --timing-only [--backbone-only]` | RTX 5070 Ti（推理）/ RTX 4060 Ti（训练） |
| Table S3 | MHCA 注意力头数消融（4/8/16 heads） | 头数消融模型分别采样 | 4/8/16-head checkpoint |
| Table S4 | MSFP 压缩通路 leave-one-out 敏感性（Vina Score/Dock、QED、SA、Diversity、Clash、RMSD） | `eval/msfp_sensitivity.sh` → `msfp_eval_light.py` + `msfp_dock_rmsd.py` + `msfp_clash.py` + `msfp_sensitivity_full.py` | `eval/7xkj/output/msfp_sensitivity_full/`（81/89 公共口袋） |
| Table S5 | 7XKJ 逐口袋 strain energy（mean±SD、median、SE>836） | `eval/7xkj` 评估数据 | `eval/7xkj/output/{mscod,molcraft}_{gdp,6ic}/docking_analysis_*_raw_data.json` |
| Table S6 | 7XKJ 逐口袋 clash 计数 | `eval/msfp_clash.py` / PoseCheck | 同上 |
| Table S7 | 各 KRAS 结构域注意力份额（%） | `eval/7xkj/attention_region_compare.py` | 模型 `_last_attentions` |
| Table S8 | 逐口袋 Vina Dock 分布（91 口袋、9443 分子） | `eval/caculate_all_metrics/3_all_metrics.py` | `checkpoints/mscod_vina_docked_pose_checked.pt` |

---

## 关键数据文件位置

- 权重：`checkpoints/mscod_last.ckpt`（3,204,063 参数；需单独下载，见 README「Checkpoints」）
- 已评估结果：`checkpoints/mscod_vina_docked_pose_checked.pt`（91 口袋、9443 分子的 Vina/PoseCheck/QED/SA）
- MSFP 敏感性中间结果：`eval/7xkj/output/msfp_sensitivity_full/{pocket}_{cfg}/`（`docking_analysis_light.json`=Vina Score/Min+QED/SA，`dock_rmsd_stats.json`=Vina Dock+RMSD，`0-9.sdf`=生成分子）
- 7XKJ 案例结果：`eval/7xkj/output/{model}_{pocket}/docking_analysis_*_raw_data.json`

---

## 复现要点

1. **Table S4（MSFP 敏感性）** 是配对分析：只保留四个配置都完成评估的公共口袋（对接类指标 81 个、PoseCheck clash 89 个），每口袋 10 分子；绝对数值与 Table I（100 口袋 × 100 分子）不可直接比较，只比较四配置间的配对差异。
2. **Table S2 推理开销**：在同一张卡上分别跑 `--timing-only`（MSCoD）与 `--timing-only --backbone-only`（MolCRAFT 等价 backbone），比值即 +4.9%。
3. **Fig. 5 / Table S7** 读 `co_attention._last_attentions`；新版向量化后其格式为 `[B, H, Nl, Np]`（B=分子数，H=头数），`attention_heatmap.py` / `attention_region_compare.py` 已按此读取。

---

## 详细复现命令

### 训练
```bash
python train_bfn.py --exp_name ${EXP_NAME}
```

### 生成（单口袋）
```bash
python sample_for_pocket.py <protein.pdb> <reference_ligand.sdf> \
    --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --sample_steps 100 --out_dir output/pocket
```

### 推理开销（Table S2）
```bash
# MSCoD
python sample_for_pocket.py <pocket.pdb> <lig.sdf> --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --timing-only
# MolCRAFT 等价 backbone
python sample_for_pocket.py <pocket.pdb> <lig.sdf> --ckpt checkpoints/mscod_last.ckpt \
    --num_samples 100 --timing-only --backbone-only
```

### MSFP 敏感性（Table S4，SM Sec 5）
```bash
bash eval/msfp_sensitivity.sh
# 或分阶段：
python eval/msfp_eval_light.py    # Vina Score/Min + QED/SA（81 公共口袋）
python eval/msfp_dock_rmsd.py     # Vina Dock + 姿态稳定性 RMSD
python eval/msfp_clash.py         # PoseCheck clash（89 口袋）
python eval/msfp_probe_batch.py   # 瓶颈表示探针
python eval/msfp_sensitivity_full.py
```

### KRAS G12D (7XKJ) 案例（Fig. 4/5，Table S5–S7）
```bash
python eval/7xkj/sample_for_pocket.py eval/7xkj/7xkj_A_rec.pdb eval/7xkj/7xkj_A_rec_7xkj_lig_gdp.sdf
python eval/7xkj/view.py                       # Fig. 4 小提琴图
python eval/7xkj/attention_heatmap.py          # Fig. 5 热图
python eval/7xkj/attention_region_compare.py   # Fig. S5 / Table S7
python eval/7xkj/msfp_scale_viz.py             # Fig. S4
```

### 全指标后处理（Table I / Table S8）
```bash
python eval/caculate_all_metrics/1_add_rmsd_pt.py
python eval/caculate_all_metrics/2_add_pose_pt.py
python eval/caculate_all_metrics/3_all_metrics.py
```

### Fig. 3 生成
```bash
python visualization/fig3_export.py
python visualization/fig3_render.py       # 需要 PyMOL
python visualization/fig3_label_overlay.py
python visualization/fig3_montage.py
```
