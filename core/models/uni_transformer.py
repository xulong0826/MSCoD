import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


class FeatureCompressionMLP(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim, activation_fn=F.relu):
        super().__init__()

        self.activation_fn = activation_fn

        # 压缩层：逐步将输入特征压缩到瓶颈维度
        self.fc_compress_1 = nn.Linear(input_dim, input_dim // 2)  # 第一层压缩
        self.fc_compress_2 = nn.Linear(input_dim // 2, input_dim // 4)  # 第二层压缩
        self.fc_compress_3 = nn.Linear(input_dim // 4, bottleneck_dim)  # 第三层压缩，瓶颈维度

        # 扩展层：逐步将瓶颈维度恢复到输出维度
        self.fc_expand_1 = nn.Linear(bottleneck_dim, input_dim // 4)  # 第一层恢复
        self.fc_expand_2 = nn.Linear(input_dim // 4, input_dim // 2)  # 第二层恢复
        self.fc_expand_3 = nn.Linear(input_dim // 2, output_dim)  # 第三层恢复，输出维度

        # 可学习的参数 alpha，用于残差连接
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5，控制残差连接的权重

    def forward(self, h, mask_ligand, batch_mask, custom_activation=None):
        act_fn = custom_activation if custom_activation else self.activation_fn

        # 使用 mask_ligand 来区分蛋白质和配体
        protein_mask = (mask_ligand == 0) & batch_mask
        ligand_mask = (mask_ligand == 1) & batch_mask

        # 提取蛋白质和配体特征
        h_protein = h * protein_mask.unsqueeze(-1)
        h_ligand = h * ligand_mask.unsqueeze(-1)

        # 特征压缩：逐步将输入特征压缩到瓶颈维度
        h_protein_compressed = act_fn(self.fc_compress_1(h_protein))
        h_protein_compressed = act_fn(self.fc_compress_2(h_protein_compressed))
        h_protein_compressed = act_fn(self.fc_compress_3(h_protein_compressed))

        h_ligand_compressed = act_fn(self.fc_compress_1(h_ligand))
        h_ligand_compressed = act_fn(self.fc_compress_2(h_ligand_compressed))
        h_ligand_compressed = act_fn(self.fc_compress_3(h_ligand_compressed))

        # 特征恢复：逐步将瓶颈维度的特征恢复到输出维度
        h_protein_expanded = act_fn(self.fc_expand_1(h_protein_compressed))
        h_protein_expanded = act_fn(self.fc_expand_2(h_protein_expanded))
        h_protein_expanded = self.fc_expand_3(h_protein_expanded)

        h_ligand_expanded = act_fn(self.fc_expand_1(h_ligand_compressed))
        h_ligand_expanded = act_fn(self.fc_expand_2(h_ligand_expanded))
        h_ligand_expanded = self.fc_expand_3(h_ligand_expanded)

        # 使用残差连接：更新蛋白质和配体的特征
        h_updated = h.clone()
        h_updated[protein_mask] = self.alpha * h_protein_expanded + (1 - self.alpha) * h_protein
        h_updated[ligand_mask] = self.alpha * h_ligand_expanded + (1 - self.alpha) * h_ligand

        return h_updated

# 协同注意力模块
class MultiHeadCoAttentionWithGating(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = feature_dim // num_heads

        # 多头线性变换
        self.protein_linears = nn.ModuleList(
            [nn.Linear(feature_dim, self.attention_dim, bias=False) for _ in range(num_heads)])
        self.ligand_linears = nn.ModuleList(
            [nn.Linear(feature_dim, self.attention_dim, bias=False) for _ in range(num_heads)])

        # 上下文特征更新层
        self.protein_updates = nn.ModuleList([nn.Linear(self.attention_dim, feature_dim) for _ in range(num_heads)])
        self.ligand_updates = nn.ModuleList([nn.Linear(self.attention_dim, feature_dim) for _ in range(num_heads)])

        # 注意力权重和归一化
        self.attention_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(self.attention_dim)) for _ in range(num_heads)])
        self.norm_layer = nn.LayerNorm(self.attention_dim)  # 归一化层

        # 门控机制
        self.protein_gate = nn.Linear(feature_dim, self.attention_dim)  # 蛋白质门控
        self.ligand_gate = nn.Linear(feature_dim, self.attention_dim)  # 配体门控
        self.gate_merge = nn.Linear(2 * self.attention_dim, self.attention_dim)  # 融合门控

        # FFN 层用于瓶颈调整
        self.ffn_bottleneck = nn.Linear(self.attention_dim, self.attention_dim)
        self.sigmoid_bottleneck = nn.Sigmoid()

        # 输出融合
        self.final_layer = nn.Linear(num_heads * feature_dim, feature_dim)  # 多头输出融合

    def forward(self, h, mask_ligand, batch_mask):
        protein_batch_mask = (mask_ligand == 0) & batch_mask
        ligand_batch_mask = (mask_ligand == 1) & batch_mask

        h_protein = h[protein_batch_mask]
        h_ligand = h[ligand_batch_mask]

        # 多头投影和上下文特征计算
        protein_contexts, ligand_contexts = [], []
        for head_idx in range(self.num_heads):
            # 多头投影
            protein_feat = self.protein_linears[head_idx](h_protein)  # (N_p, attention_dim)
            ligand_feat = self.ligand_linears[head_idx](h_ligand)  # (N_l, attention_dim)

            # 计算注意力得分
            attention_scores = torch.matmul(protein_feat, ligand_feat.T) / (self.attention_dim ** 0.5)  # (N_p, N_l)
            protein_attention = F.softmax(attention_scores, dim=-1)  # 对配体归一化
            ligand_attention = F.softmax(attention_scores.T, dim=-1)  # 对蛋白质归一化

            # 上下文特征
            protein_context = torch.matmul(protein_attention, ligand_feat)  # (N_p, attention_dim)
            ligand_context = torch.matmul(ligand_attention, protein_feat)  # (N_l, attention_dim)

            # 门控机制
            protein_gate = torch.sigmoid(self.protein_gate(h_protein))  # (N_p, attention_dim)
            ligand_gate = torch.sigmoid(self.ligand_gate(h_ligand))  # (N_l, attention_dim)

            protein_context = self.norm_layer(protein_context * protein_gate)  # 归一化加权
            ligand_context = self.norm_layer(ligand_context * ligand_gate)

            # 引入 FFN 和 Sigmoid 激活进行调整
            protein_context = self.ffn_bottleneck(protein_context)
            protein_context = self.sigmoid_bottleneck(protein_context)

            ligand_context = self.ffn_bottleneck(ligand_context)
            ligand_context = self.sigmoid_bottleneck(ligand_context)

            protein_contexts.append(self.protein_updates[head_idx](protein_context))  # (N_p, feature_dim)
            ligand_contexts.append(self.ligand_updates[head_idx](ligand_context))  # (N_l, feature_dim)

        # 汇总所有头
        protein_final = torch.cat(protein_contexts, dim=-1)  # (N_p, num_heads * feature_dim)
        ligand_final = torch.cat(ligand_contexts, dim=-1)  # (N_l, num_heads * feature_dim)

        # 多头融合
        protein_final = self.final_layer(protein_final)  # (N_p, feature_dim)
        ligand_final = self.final_layer(ligand_final)  # (N_l, feature_dim)

        # 残差连接：将更新后的特征与输入特征相加
        h_updated = h.clone()
        h_updated[protein_batch_mask] = protein_final + h_protein  # 蛋白质残差连接
        h_updated[ligand_batch_mask] = ligand_final + h_ligand  # 配体残差连接

        return h_updated

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
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
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32, bottleneck_dim=32, num_heads=4,
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

        self.multimlp = FeatureCompressionMLP(input_dim=hidden_dim, bottleneck_dim=bottleneck_dim, output_dim=hidden_dim)
        self.cottention = MultiHeadCoAttentionWithGating(feature_dim=hidden_dim, num_heads=num_heads)

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

    def _apply_interaction_pipeline(self, h, mask_ligand, batch, num_interactions):
        # 串联协同注意力和信息瓶颈模块，并重复 num_interactions 次。

        # Parameters:
        # - h: 当前特征
        # - mask_ligand: 是否为配体的掩码
        # - batch: 节点批次
        # - num_interactions: 重复次数

        # Returns:
        # - h: 更新后的特征
        # original_h = h.clone()  # 保存原始特征
        for i in range(num_interactions):
            # 协同注意力
            h = self.cottention(h, mask_ligand, batch)
            # 信息瓶颈
            h = self.multimlp(h, mask_ligand, batch)

            # 局部特征残差更新
            # h = h + original_h  # 加入残差连接，结合原始特征

        return h

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
                # h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)

                # 只有前几个block才应用残差连接
                if b_idx > self.num_blocks // 3:  # 例如在后半部分使用残差连接
                    original_h = h.clone()

                # 更新特征
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
                # 协同注意力与信息瓶颈的串联
                h = self._apply_interaction_pipeline(h, mask_ligand, batch, num_interactions=1)

                # 局部特征更新：将更新后的特征与原始特征相加（可选）
                if b_idx > self.num_blocks // 3:  # 仅在后半部分加入残差连接
                    h = h + original_h  # 加入残差连接，结合原始特征
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
