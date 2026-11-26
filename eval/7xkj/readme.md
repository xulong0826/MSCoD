# eval/7xkj/ — KRAS G12D (7XKJ) 案例研究

对应论文 **Fig. 4、Fig. 5** 与 **SM Sec. 6（逐口袋统计）、Sec. 7（区域注意力）、Fig. S4/S5**。

## 脚本 ↔ 论文对照

| 脚本 | 产出 | 对应论文 |
|---|---|---|
| `sample_for_pocket.py` | 口袋条件生成 | 案例研究生成 |
| `view.py` | 属性小提琴图（QED/SA/Vina/SE/Clash） | **Fig. 4** |
| `attention_heatmap.py` | 配体→蛋白注意力热图 | **Fig. 5(a)(b)** |
| `make_region_legend.py` | 热图区域图例 | **Fig. 5 图例** |
| `attention_region_compare.py` | 区域注意力份额对比 | **Fig. S5** + **Table S7** |
| `msfp_scale_viz.py` | MSFP 瓶颈 PCA（按结构域着色） | **Fig. S4** |
| `msfp_probe.py` | MSFP 瓶颈表示探针 | SM Sec 5.2 探针 |

## 数据

评估结果：`eval/7xkj/output/{model}_{pocket}/docking_analysis_*_raw_data.json`
（字段：qed/sa/vina_score/vina_minimize/vina_dock/strain/clash/hb_*/vdw/hydrophobic）。

## 用法

```bash
python sample_for_pocket.py 7xkj_A_rec.pdb 7xkj_A_rec_7xkj_lig_gdp.sdf
python view.py                       # Fig. 4
python attention_heatmap.py          # Fig. 5
python attention_region_compare.py   # Fig. S5 / Table S7
python msfp_scale_viz.py             # Fig. S4
```