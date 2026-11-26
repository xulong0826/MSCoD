import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid", palette="muted")

# Multiple JSON files and their labels for comparison
json_files = [
    ('output/molcraft_6ic/docking_analysis_7xkj_A_rec_7xkj_lig_6ic_raw_data.json', '6ic_molcraft'),
    ('output/mscod_6ic/docking_analysis_7xkj_A_rec_7xkj_lig_6ic_raw_data.json', '6ic_mscod'),
    ('output/molcraft_gdp/docking_analysis_7xkj_A_rec_7xkj_lig_gdp_raw_data.json', 'gdp_molcraft'),
    ('output/mscod_gdp/docking_analysis_7xkj_A_rec_7xkj_lig_gdp_raw_data.json', 'gdp_mscod'),
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
    'qed', 'sa',
    'vina_score', 'vina_minimize', 'vina_dock',
    'strain', 'clash'
]

metric_labels = {
    'qed': 'QED',
    'sa': 'SA',
    'vina_score': 'Vina Score',
    'vina_minimize': 'Vina Minimize',
    'vina_dock': 'Vina Dock',
    'strain': 'SE',
    'clash': 'Clash'
}

num_metrics = len(metrics)
group_labels = list(all_data.keys())

# Two-row layout: 7 violin plots + 1 clean legend cell
fig_violin, axes_violin = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)
axes_violin = axes_violin.ravel()

for idx, metric in enumerate(metrics):
    ax = axes_violin[idx]
    data_to_plot = []
    palette = []
    for label in group_labels:
        data = all_data[label]
        values = [item[metric] for item in data if metric in item]
        data_to_plot.append(values)
        palette.append(color_map.get(label, None))

    # Plot violin plot
    sns.violinplot(data=data_to_plot, ax=ax, palette=palette, inner="box")

    # Add reference ligand points (only for metrics with known reference values)
    for i, label in enumerate(group_labels):
        if '6ic' in label.lower():
            ref_value = reference_data['6ic'].get(metric)
            color, marker = 'red', 's'
        elif 'gdp' in label.lower():
            ref_value = reference_data['gdp'].get(metric)
            color, marker = 'black', 'o'
        else:
            continue
        if ref_value is None:
            continue
        ax.scatter(i, ref_value, color=color, s=60, marker=marker,
                   edgecolors='white', linewidth=2, zorder=10)

    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels, rotation=30, fontsize=11)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.grid(axis='y')

# ---- clean legend in the eighth cell ----
ax_leg = axes_violin[7]
ax_leg.axis('off')
handles = [Patch(color=color_map[l], label=l) for l in group_labels]
handles += [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Ref 6IC'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Ref GDP'),
]
ax_leg.legend(handles=handles, loc='center', fontsize=11, frameon=True, title=None)

fig_violin.suptitle("Comparison with Reference Ligands (QED/SA: higher is better, Vina/SE/Clash: lower is better)", fontsize=14, y=1.06)
plt.savefig('7xkj.png', dpi=200, bbox_inches='tight')
plt.close(fig_violin)