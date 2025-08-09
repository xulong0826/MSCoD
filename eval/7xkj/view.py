import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid", palette="muted")

# Multiple JSON files and their labels for comparison
json_files = [
    ('6ic_molcraft.json', '6ic_molcraft'),
    ('6ic_mscod.json', '6ic_mscod'),
    ('gdp_molcraft.json', 'gdp_molcraft'),
    ('gdp_mscod.json', 'gdp_mscod'),
]

# Reference ligand data
reference_data = {
    'gdp': {'qed': 0.229, 'sa': 0.64, 'lipinski': 2, 'vina_score': -8.84, 'vina_minimize': -9.407, 'vina_dock': -9.549},
    '6ic': {'qed': 0.319, 'sa': 0.47, 'lipinski': 4, 'vina_score': -9.359, 'vina_minimize': -12.234, 'vina_dock': -12.507}
}

# Color mapping: blue for 6ic, orange/red for gdp
color_map = {
    '6ic_molcraft': '#4C72B0',
    '6ic_mscod': '#55A868',
    'gdp_molcraft': '#E17C05',
    'gdp_mscod': '#C44E52',
}

# Read all data
all_data = {}
for file, label in json_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        all_data[label] = data
    else:
        print(f"File not found: {file}")

metrics = [
    'qed', 'sa', 'lipinski',
    'vina_score', 'vina_minimize', 'vina_dock'
]

metric_labels = {
    'qed': 'QED',
    'sa': 'SA',
    'lipinski': 'Lipinski',
    'vina_score': 'Vina Score',
    'vina_minimize': 'Vina Minimize',
    'vina_dock': 'Vina Dock'
}

num_metrics = len(metrics)
group_labels = list(all_data.keys())

# Horizontal compact layout: all violin plots in one row
fig_violin, axes_violin = plt.subplots(1, num_metrics, figsize=(3*num_metrics, 4), constrained_layout=True)
if num_metrics == 1:
    axes_violin = [axes_violin]

for idx, metric in enumerate(metrics):
    data_to_plot = []
    palette = []
    for label in group_labels:
        data = all_data[label]
        values = [item[metric] for item in data if metric in item]
        data_to_plot.append(values)
        palette.append(color_map.get(label, None))
    
    # Plot violin plot
    sns.violinplot(data=data_to_plot, ax=axes_violin[idx], palette=palette, inner="box")
    
    # 对 lipinski 指标设置 y 轴范围限制
    if metric == 'lipinski':
        axes_violin[idx].set_ylim(-0.5, 5.5)
        # 添加最大值参考线
        axes_violin[idx].axhline(y=5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        axes_violin[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        # 在右侧添加范围标注
        axes_violin[idx].text(len(group_labels)-0.5, 5.2, 'Max=5', 
                             fontsize=8, ha='center', color='red')
    
    # Add reference ligand points
    for i, label in enumerate(group_labels):
        # Determine which reference data to use based on label
        if '6ic' in label.lower():
            ref_value = reference_data['6ic'][metric]
            color = 'red'
            marker = 's'  # square
        elif 'gdp' in label.lower():
            ref_value = reference_data['gdp'][metric]
            color = 'black'
            marker = 'o'  # circle
        else:
            continue
        
        # Plot reference point
        axes_violin[idx].scatter(i, ref_value, color=color, s=60, marker=marker, 
                               edgecolors='white', linewidth=2, zorder=10, 
                               label=f'Ref {label.split("_")[0].upper()}' if i == 0 or (i == 2 and 'gdp' in label.lower()) else "")
    
    axes_violin[idx].set_xticks(range(len(group_labels)))
    axes_violin[idx].set_xticklabels(group_labels, rotation=30, fontsize=8)
    axes_violin[idx].set_ylabel(metric_labels.get(metric, metric))
    axes_violin[idx].set_title(metric_labels.get(metric, metric), fontsize=10)
    axes_violin[idx].grid(axis='y')
    
    # Add legend only to the first subplot
    if idx == 0:
        axes_violin[idx].legend(loc='upper right', fontsize=6)

fig_violin.suptitle("Comparison with Reference Ligands (QED/SA/Lipinski: higher is better, Vina: lower is better)", fontsize=14, y=1.05)
plt.savefig('all_metrics_violin_with_reference.png', dpi=200, bbox_inches='tight')
plt.close(fig_violin)