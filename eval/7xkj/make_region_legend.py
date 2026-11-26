"""Generate a single shared structural-region colour legend for Fig. 5."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

region_colors = {
    'P-loop': '#1f77b4', 'switch-I': '#ff7f0e', 'switch-II': '#d62728',
    'b4-b5': '#2ca02c', 'a3': '#9467bd', 'NKXD loop': '#8c564b', 'a3-b5': '#e377c2',
}

fig = plt.figure(figsize=(6.5, 0.35))
handles = [Patch(color=c, label=r) for r, c in region_colors.items()]
fig.legend(handles=handles, loc='center', ncol=7, frameon=False,
           fontsize=9, handlelength=1.0, handleheight=0.7, columnspacing=1.0)
plt.savefig('attention_region_legend.png', dpi=300, bbox_inches='tight')
print('saved attention_region_legend.png')
