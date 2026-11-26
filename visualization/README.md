# visualization/ — Figure 3 生成流水线

对应论文 **Fig. 3**（8 个测试口袋的对接视图：口袋表面 / 配体 / 接触残基 / 氢键）。

## 四步流水线（均从仓库根目录运行）

| 步骤 | 脚本 | 作用 |
|---|---|---|
| 1 | `python visualization/fig3_export.py` | 从 pose-checked 结果导出每个案例的 `protein.pdb` / `ligand.pdb` / `complex.pdb` + `manifest.jsonl` |
| 2 | `python visualization/fig3_render.py` | 用 PyMOL 渲染结构 → `panel_nolabel.png` + `labels.json` |
| 3 | `python visualization/fig3_label_overlay.py` | 叠加蛋白名 / PDB ID / Vina Dock / QED / SA 文字 → `panel.png` |
| 4 | `python visualization/fig3_montage.py` | 拼成 2×4 网格 → `fig3_montage.png` |

## 依赖

- 步骤 1：`checkpoints/mscod_vina_docked_pose_checked.pt` + `data/test_set`
- 步骤 2：PyMOL（开源版，无水印）
- 步骤 3/4：PIL + matplotlib

## 路径说明

默认输出目录是 `../paper/fig3/`（与论文 LaTeX 工程相邻）。可用 `--out_dir`（步骤 1）或 `--fig3_dir`（步骤 3/4）覆盖到任意目录。
