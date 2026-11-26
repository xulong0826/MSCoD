import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


# ============================================================
# 模块级辅助函数（新增，不改变任何类的名字/签名）
# ============================================================
def _pad_group(feat, group_id, num_groups, max_size):
    """把按 batch 连续的特征 pad 成 [B, max_size, F]，返回 (padded, mask)。"""
    N, F = feat.shape[0], feat.shape[-1]
    device = feat.device
    counts = torch.bincount(group_id, minlength=num_groups)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(N, device=device) - starts[group_id]
    padded = torch.zeros(num_groups, max_size, F, dtype=feat.dtype, device=device)
    mask = torch.ones(num_groups, max_size, dtype=torch.bool, device=device)
    padded[group_id, pos] = feat
    mask[group_id, pos] = False
    return padded, mask


def _unpad_group(padded, mask):
    """[B, max_size, F] -> [N_valid, F]，顺序 = 原 batch 连续顺序。"""
    return padded[~mask]


def _ligand_coattention_batched(co_attn, h_spp, batch, is_protein, is_ligand,
                                pad_ratio_threshold=3.0):
    """批量配体协同注意力：padding + 块对角注意力；浪费过大时回退循环。"""
    protein_feat = h_spp[is_protein]
    ligand_feat = h_spp[is_ligand]
    protein_batch = batch[is_protein]
    ligand_batch = batch[is_ligand]

    B = int(batch.max().item()) + 1
    np_counts = torch.bincount(protein_batch, minlength=B)
    nl_counts = torch.bincount(ligand_batch, minlength=B)
    max_np = int(np_counts.max().item())
    max_nl = int(nl_counts.max().item())

    total_pairs = int((np_counts * nl_counts).sum().item())
    batched_pairs = B * max_np * max_nl
    if batched_pairs > pad_ratio_threshold * total_pairs:
        # 回退：逐 batch 循环（语义与原实现完全一致）
        ligand_out = torch.empty_like(ligand_feat)
        ligand_batch_ids = batch[is_ligand]
        for b in range(B):
            protein_mask = (batch == b) & is_protein
            ligand_mask = (batch == b) & is_ligand
            _, lf = co_attn(h_spp[protein_mask], h_spp[ligand_mask])
            ligand_out[ligand_batch_ids == b] = lf
        return ligand_out

    protein_padded, protein_mask = _pad_group(protein_feat, protein_batch, B, max_np)
    ligand_padded, ligand_mask = _pad_group(ligand_feat, ligand_batch, B, max_nl)
    _, ligand_final_padded = co_attn(
        protein_padded, ligand_padded,
        protein_pad_mask=protein_mask, ligand_pad_mask=ligand_mask)
    return _unpad_group(ligand_final_padded, ligand_mask)


class MultiScaleInformationBottleneck(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bottleneck_dims = [16, 32, 64]
        self.bottlenecks = nn.ModuleList()

        for bottleneck_dim in self.bottleneck_dims:
            self.bottlenecks.append(nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, output_dim),
                nn.ReLU()
            ))
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, h):
        out = h
        for bottleneck in self.bottlenecks:
            out = out + bottleneck(h)
        out = self.norm(out)
        out = self.ffn(out)
        return out


class MultiHeadCoAttentionWithGating(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.attention_dim = feature_dim // num_heads

        # 多头线性变换（保留 ModuleList，权重名不变，checkpoint 兼容）
        self.protein_linears = nn.ModuleList([nn.Linear(feature_dim, self.attention_dim, bias=False) for _ in range(num_heads)])
        self.ligand_linears = nn.ModuleList([nn.Linear(feature_dim, self.attention_dim, bias=False) for _ in range(num_heads)])

        # 更新层
        self.ligand_updates = nn.ModuleList([nn.Linear(self.attention_dim, feature_dim) for _ in range(num_heads)])

        # 门控机制
        self.ligand_gate = nn.Linear(feature_dim, self.attention_dim)
        self.gate_merge = nn.Linear(2 * self.attention_dim, self.attention_dim)  # 保留（原代码死代码，checkpoint 兼容）

        # 输出融合层
        self.final_layer = nn.Linear(num_heads * feature_dim, feature_dim)

        # 归一化层
        self.norm = nn.LayerNorm(feature_dim)

        # 前馈神经网络
        self.ffn_pre = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.ffn_post = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self._inv_scale = 1.0 / math.sqrt(self.attention_dim)
        self._last_attentions = None

    def forward(self, h_protein, h_ligand, protein_pad_mask=None, ligand_pad_mask=None):
        # 单对（2D）自动升维为 B=1，保持向后兼容
        squeeze = h_protein.dim() == 2
        if squeeze:
            h_protein = h_protein.unsqueeze(0)
            h_ligand = h_ligand.unsqueeze(0)
            protein_pad_mask = torch.zeros(1, h_protein.shape[1], dtype=torch.bool, device=h_protein.device)
            ligand_pad_mask = torch.zeros(1, h_ligand.shape[1], dtype=torch.bool, device=h_ligand.device)

        B, Np, F = h_protein.shape
        B, Nl, _ = h_ligand.shape
        H, A = self.num_heads, self.attention_dim

        ligand_gate = torch.sigmoid(self.ligand_gate(h_ligand))          # [B, Nl, A]

        # 向量化多头投影
        Wp = torch.stack([m.weight for m in self.protein_linears])       # [H, A, F]
        Wl = torch.stack([m.weight for m in self.ligand_linears])        # [H, A, F]
        protein_feat = torch.einsum('bnf,haf->bhna', h_protein, Wp)      # [B, H, Np, A]
        ligand_feat = torch.einsum('bnf,haf->bhna', h_ligand, Wl)        # [B, H, Nl, A]

        # 批量注意力得分
        scores = torch.einsum('bhpa,bhla->bhpl', protein_feat, ligand_feat) * self._inv_scale  # [B,H,Np,Nl]

        if protein_pad_mask is not None:
            scores = scores.masked_fill(protein_pad_mask[:, None, :, None], float('-inf'))

        ligand_attention = scores.softmax(dim=2)                         # [B,H,Np,Nl]

        self._last_attentions = ligand_attention.detach().transpose(-1, -2)  # [B,H,Nl,Np]

        # 批量上下文
        ligand_context = torch.einsum('bhpl,bhpa->bhla', ligand_attention, protein_feat)  # [B,H,Nl,A]
        ligand_context = ligand_context * ligand_gate[:, None]

        # 向量化多头更新
        Wu = torch.stack([m.weight for m in self.ligand_updates])        # [H, F, A]
        bu = torch.stack([m.bias for m in self.ligand_updates])          # [H, F]
        ligand_contexts = torch.einsum('bhla,hfa->bhlf', ligand_context, Wu) + bu.view(1, H, 1, F)

        # 合并头
        ligand_final = ligand_contexts.transpose(1, 2).reshape(B, Nl, H * F)
        ligand_final = self.final_layer(ligand_final)

        ligand_final = self.ffn_pre(ligand_final)
        ligand_final = torch.sigmoid(ligand_final)
        ligand_final = ligand_final + h_ligand
        ligand_final = self.norm(ligand_final)
        ligand_final = self.ffn_post(ligand_final)

        if squeeze:
            return h_protein.squeeze(0), ligand_final.squeeze(0)
        return h_protein, ligand_final


class CombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim, num_heads):
        super(CombinedModel, self).__init__()
        self.spp = MultiScaleInformationBottleneck(input_dim, output_dim)
        self.co_attention = MultiHeadCoAttentionWithGating(feature_dim, num_heads)

        self.ffn_pre = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.ffn_post = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, h, mask_ligand, batch):
        is_protein = mask_ligand == 0
        is_ligand = mask_ligand == 1

        h_spp = self.spp(h)

        # ---- 蛋白分支 ----
        protein_features = h_spp[is_protein]
        protein_final = self.ffn_pre(protein_features)
        protein_final = torch.sigmoid(protein_final)
        protein_final = protein_final + protein_features
        protein_final = self.norm(protein_final)
        protein_final = self.ffn_post(protein_final)

        # ---- 配体分支：批量 + 自适应回退 ----
        ligand_features = h_spp[is_ligand]
        ligand_out = _ligand_coattention_batched(
            self.co_attention, h_spp, batch, is_protein, is_ligand)

        ligand_final = self.ffn_pre(ligand_out)
        ligand_final = torch.sigmoid(ligand_final)
        ligand_final = ligand_final + ligand_features
        ligand_final = self.norm(ligand_final)
        ligand_final = self.ffn_post(ligand_final)

        # ---- 重组 ----
        h_updated = h.clone()
        h_updated[is_protein] = protein_final
        h_updated[is_ligand] = ligand_final
        return h_updated


class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32, bottleneck_dim=32, num_heads=8,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        self.combinedmodel = CombinedModel(input_dim=hidden_dim, output_dim=hidden_dim, feature_dim=hidden_dim, num_heads=num_heads)

    def _apply_interaction_pipeline(self, h, mask_ligand, batch, num_interactions=1):
        # 串联协同注意力和信息瓶颈模块，并重复 num_interactions 次
        for i in range(num_interactions):
            # 协同注意
            h = self.combinedmodel(h, mask_ligand, batch)
        return h

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.knn, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                # 更新特征
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
                h = self._apply_interaction_pipeline(h, mask_ligand, batch, 1)

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
