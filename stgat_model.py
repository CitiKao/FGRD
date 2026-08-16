"""
stgat_model.py — Spatio-Temporal Graph Attention Network 預測模型

架構（對應論文 Section 4.2）：
  ┌─────────────────────────────────────────────────┐
  │  Node Stream (需求 / 空車)                       │
  │    Path-Fixed   : [GTCN → GAT(A_fixed, edge)]×L │
  │    Path-Adaptive: [GTCN → GAT(A_adp)]×L         │
  │    → GatedFusion → demand_head / supply_head     │
  ├─────────────────────────────────────────────────┤
  │  Edge Stream (道路速度)                          │
  │    [GTCN → GAT(line_graph)]×L                    │
  │    → speed_head                                  │
  └─────────────────────────────────────────────────┘

輸出：predicted demand, vacant taxis, road speeds for next p steps
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════════════

def build_line_graph_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    從邊列表建構線圖 (line graph) 邊列表。
    兩條有向邊 e_i=(u,v) 與 e_j=(v,w) 相鄰，表示線圖中的 i <- j。
    """
    dst = edge_index[:, 1]  # (|E|,)
    src = edge_index[:, 0]  # (|E|,)
    recv, send = torch.nonzero(dst.unsqueeze(1) == src.unsqueeze(0), as_tuple=True)
    return torch.stack([recv, send], dim=0).long()


# ════════════════════════════════════════════════════════════════
#  GTCN — Gated Temporal Convolutional Network
# ════════════════════════════════════════════════════════════════

class GTCNLayer(nn.Module):
    """
    單層門控時間卷積（Eq. 1-2）。
    output = gate ⊗ filter + (1 - gate) ⊗ residual
    其中 gate = σ(conv(x))
    """

    def __init__(
        self, in_c: int, out_c: int, kernel_size: int = 3, dilation: int = 1
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.gate_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.filter_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)       # (B*N, C, T)
        x = F.pad(x, (self.pad, 0))                         # causal left-pad
        gate = torch.sigmoid(self.gate_conv(x))
        out = gate * self.filter_conv(x) + (1 - gate) * self.skip_conv(x)
        return out.permute(0, 2, 1).reshape(B, N, T, -1)


class GTCN(nn.Module):
    """多層 GTCN，含殘差連接與 LayerNorm。"""

    def __init__(
        self,
        in_c: int,
        hid_c: int,
        out_c: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            ic = in_c if i == 0 else hid_c
            oc = hid_c if i < num_layers - 1 else out_c
            self.layers.append(GTCNLayer(ic, oc, kernel_size, dilation=2 ** i))
            self.norms.append(nn.LayerNorm(oc))
        self.skip = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        res = self.skip(x)
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x)))
        return x + res


# ════════════════════════════════════════════════════════════════
#  GAT — Graph Attention Layer
# ════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    通用圖注意力層，支援可選的邊特徵（Eq. 3-7 / 8-12）。

    若 edge_in > 0 → 區域級 GAT（含邊特徵 len, speed）
    若 edge_in == 0 → 邊級 GAT（不含額外邊特徵）
    """

    def __init__(
        self,
        in_c: int,
        d_out: int,
        num_heads: int = 4,
        edge_in: int = 0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.concat = concat
        self.has_edge = edge_in > 0
        total = d_out * num_heads

        self.W_q = nn.Linear(in_c, total, bias=False)
        self.W_k = nn.Linear(in_c, total, bias=False)
        self.W_v = nn.Linear(in_c, total, bias=False)

        self.a_q = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        self.a_k = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        nn.init.xavier_normal_(self.a_q)
        nn.init.xavier_normal_(self.a_k)

        if self.has_edge:
            self.W_e = nn.Linear(edge_in, total, bias=False)
            self.a_e = nn.Parameter(torch.empty(1, 1, 1, num_heads, d_out))
            nn.init.xavier_normal_(self.a_e)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h         : (B, N, C_in)
        adj       : (N, N) binary mask
        edge_feat : (B, N, N, edge_in) or None
        return    : (B, N, H*d_out) if concat else (B, N, d_out)
        """
        B, N, _ = h.shape
        H, D = self.num_heads, self.d_out

        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        # additive attention: a_q·q_i + a_k·k_j (+ a_e·e_{i,j})
        s_q = (q * self.a_q).sum(-1)  # (B, N, H)
        s_k = (k * self.a_k).sum(-1)  # (B, N, H)
        scores = s_q.unsqueeze(2) + s_k.unsqueeze(1)  # (B, N, N, H)

        if self.has_edge and edge_feat is not None:
            e = self.W_e(edge_feat).view(B, N, N, H, D)
            s_e = (e * self.a_e).sum(-1)  # (B, N, N, H)
            scores = scores + s_e

        scores = F.leaky_relu(scores, 0.2)

        # mask non-neighbors and bias attention by adjacency weights
        adj = adj.to(device=h.device, dtype=scores.dtype)
        mask = adj > 0
        scores = scores + adj.clamp_min(1e-12).log().unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), -1e9)

        alpha = F.softmax(scores, dim=2)  # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # aggregate: s'_i = Σ_j α_{i,j} · v_j
        out = torch.einsum("bnjh,bnjhd->bnhd", alpha, v.unsqueeze(1).expand(-1, N, -1, -1, -1))

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return F.elu(out)


class SparseGATLayer(nn.Module):
    """
    Sparse GAT on an explicit edge list.

    Instead of materializing a dense NxN attention matrix and masking most
    entries away, this layer only computes attention on real graph neighbors.
    """

    def __init__(
        self,
        in_c: int,
        d_out: int,
        num_heads: int = 4,
        edge_in: int = 0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.concat = concat
        self.has_edge = edge_in > 0
        total = d_out * num_heads

        self.W_q = nn.Linear(in_c, total, bias=False)
        self.W_k = nn.Linear(in_c, total, bias=False)
        self.W_v = nn.Linear(in_c, total, bias=False)

        self.a_q = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        self.a_k = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        nn.init.xavier_normal_(self.a_q)
        nn.init.xavier_normal_(self.a_k)

        if self.has_edge:
            self.W_e = nn.Linear(edge_in, total, bias=False)
            self.a_e = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
            nn.init.xavier_normal_(self.a_e)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h         : (B, N, C_in)
        edge_index: (2, L) with [recv, send]
        edge_feat : (B, L, edge_in) or None
        return    : (B, N, H*d_out) if concat else (B, N, d_out)
        """
        B, N, _ = h.shape
        H, D = self.num_heads, self.d_out
        recv = edge_index[0]
        send = edge_index[1]
        num_edges = recv.shape[0]

        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        q_recv = q[:, recv]  # (B, L, H, D)
        k_send = k[:, send]  # (B, L, H, D)
        scores = (q_recv * self.a_q).sum(-1) + (k_send * self.a_k).sum(-1)
        if self.has_edge and edge_feat is not None:
            e = self.W_e(edge_feat).view(B, num_edges, H, D)
            scores = scores + (e * self.a_e).sum(-1)
        scores = F.leaky_relu(scores, 0.2).float()
        if edge_weight is not None:
            weight = edge_weight.to(device=h.device, dtype=scores.dtype)
            if weight.ndim == 1:
                weight = weight.view(1, num_edges, 1)
            elif weight.ndim == 2:
                weight = weight.view(B, num_edges, 1)
            else:
                raise ValueError("edge_weight must have shape (L,) or (B, L).")
            scores = scores + weight.clamp_min(1e-12).log()

        recv_index = recv.view(1, num_edges, 1).expand(B, num_edges, H)
        max_scores = torch.full(
            (B, N, H),
            -torch.inf,
            device=h.device,
            dtype=scores.dtype,
        )
        max_scores.scatter_reduce_(1, recv_index, scores, reduce="amax", include_self=True)

        attn_logits = scores - max_scores.gather(1, recv_index)
        attn_exp = torch.exp(attn_logits)
        denom = torch.zeros((B, N, H), device=h.device, dtype=attn_exp.dtype)
        denom.scatter_add_(1, recv_index, attn_exp)
        alpha = attn_exp / denom.gather(1, recv_index).clamp_min(1e-12)
        alpha = torch.nan_to_num(alpha, nan=0.0).to(v.dtype)

        messages = alpha.unsqueeze(-1) * v[:, send]
        out = torch.zeros((B, N, H, D), device=h.device, dtype=messages.dtype)
        out.scatter_add_(
            1,
            recv.view(1, num_edges, 1, 1).expand(B, num_edges, H, D),
            messages,
        )

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return F.elu(out)


# ════════════════════════════════════════════════════════════════
#  Gated Fusion（Eq. 13-14）
# ════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        self.W3 = nn.Linear(dim, dim)

    def forward(self, h_fixed: torch.Tensor, h_adp: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W1(h_fixed + h_adp))
        return torch.tanh(gate * self.W2(h_fixed) + (1 - gate) * self.W3(h_adp))


# ════════════════════════════════════════════════════════════════
#  STGATPredictor — 完整預測模型
# ════════════════════════════════════════════════════════════════

class STGATPredictor(nn.Module):
    """
    Parameters
    ----------
    num_nodes       : N — 區域數
    edge_index      : (|E|, 2) long — 有向邊列表
    edge_lengths    : (|E|,) float — 靜態路長
    adj_matrix      : (N, N) float — 物理鄰接 (0/1)
    hidden_dim      : 隱藏層維度（也是 GTCN 輸出與 GAT 的 heads × d_out）
    num_heads       : 注意力頭數
    num_st_blocks   : ST-Block 堆疊數
    pred_horizon    : 預測未來 p 步
    hist_len        : 歷史窗口長度 h（僅用於說明，不影響模型定義）
    adaptive_emb    : 自適應鄰接 embedding 維度
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        adj_matrix: torch.Tensor,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_st_blocks: int = 2,
        num_gtcn_layers: int = 2,
        kernel_size: int = 3,
        pred_horizon: int = 3,
        adaptive_emb: int = 10,
        adaptive_topk: int = 16,
        speed_adaptive_topk: int | None = None,
        node_feat_dim: int = 2,
        edge_feat_dim: int = 2,
        adj_weight_matrix: torch.Tensor | None = None,
        node_use_fixed_graph: bool = True,
        node_use_adaptive: bool = True,
        speed_use_adaptive: bool = False,
        speed_use_fixed_graph: bool = True,
        use_speed_history_mask: bool = False,
        use_fixed_edge_length_feature: bool = False,
        v_domain: str = "edge",
    ) -> None:
        super().__init__()
        N = num_nodes
        d_per_head = hidden_dim // num_heads
        assert hidden_dim == d_per_head * num_heads, "hidden_dim must be divisible by num_heads"
        cleaned_v_domain = str(v_domain).strip().lower()
        if cleaned_v_domain not in {"edge", "node"}:
            raise ValueError("v_domain must be either 'edge' or 'node'.")
        self.v_domain = cleaned_v_domain
        self.node_use_fixed_graph = bool(node_use_fixed_graph)
        self.node_use_adaptive = bool(node_use_adaptive)
        if not self.node_use_fixed_graph and not self.node_use_adaptive:
            raise ValueError("At least one node path must be enabled.")
        self.speed_use_adaptive = bool(speed_use_adaptive)
        self.speed_use_fixed_graph = bool(speed_use_fixed_graph)
        if not self.speed_use_fixed_graph and not self.speed_use_adaptive:
            raise ValueError("At least one speed path must be enabled.")
        self.use_speed_history_mask = bool(use_speed_history_mask)
        self.use_fixed_edge_length_feature = bool(use_fixed_edge_length_feature)
        self.speed_adaptive_topk = adaptive_topk if speed_adaptive_topk is None else int(speed_adaptive_topk)

        # ── 圖結構（不需要梯度） ──
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_lengths", edge_lengths.float())
        adj_binary_with_self = (adj_matrix + torch.eye(N)).clamp(max=1.0)
        if adj_weight_matrix is not None:
            adj_weighted = adj_weight_matrix.float().clone()
            adj_weighted = torch.where(adj_matrix > 0, adj_weighted, torch.zeros_like(adj_weighted))
            adj_weighted = adj_weighted + torch.eye(N, dtype=adj_weighted.dtype, device=adj_weighted.device)
            adj_with_self = torch.where(adj_binary_with_self > 0, adj_weighted.clamp_min(0.0), torch.zeros_like(adj_weighted))
        else:
            adj_with_self = adj_binary_with_self
        self.register_buffer("adj_fixed", adj_with_self)
        self.register_buffer("adj_full", torch.ones(N, N))
        fixed_recv, fixed_send = torch.nonzero(adj_binary_with_self > 0, as_tuple=True)
        self.register_buffer("fixed_edge_index", torch.stack([fixed_recv, fixed_send], dim=0).long())
        self.register_buffer("line_edge_index", build_line_graph_edge_index(edge_index))
        edge_lookup = torch.full((N, N), -1, dtype=torch.long)
        edge_lookup[edge_index[:, 0].long(), edge_index[:, 1].long()] = torch.arange(
            edge_index.shape[0],
            dtype=torch.long,
        )
        fixed_edge_ids = edge_lookup[fixed_recv, fixed_send]
        fixed_edge_lengths = torch.zeros(fixed_edge_ids.shape[0], dtype=torch.float32)
        fixed_has_road = fixed_edge_ids >= 0
        fixed_edge_lengths[fixed_has_road] = edge_lengths.float()[fixed_edge_ids[fixed_has_road]]
        fixed_edge_weights = torch.ones(fixed_edge_ids.shape[0], dtype=torch.float32)
        if adj_weight_matrix is not None:
            fixed_edge_weights = adj_with_self[fixed_recv, fixed_send].float()
            fixed_edge_weights = torch.where(
                fixed_edge_weights > 0,
                fixed_edge_weights,
                torch.ones_like(fixed_edge_weights),
            )
        self.register_buffer("fixed_edge_ids", fixed_edge_ids)
        self.register_buffer("fixed_edge_lengths", fixed_edge_lengths)
        self.register_buffer("fixed_edge_has_road", fixed_has_road)
        self.register_buffer("fixed_edge_weights", fixed_edge_weights)

        # ── 自適應鄰接 ──
        self.emb_src = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
        self.adaptive_topk = adaptive_topk

        # ── 輸入投影 ──
        self.time_feat_dim = max(node_feat_dim - 2, 0)
        self.use_edge_domain_length_feature = (
            self.v_domain == "edge" and self.use_fixed_edge_length_feature
        )
        self.edge_input_dim = (
            1
            + (1 if self.use_edge_domain_length_feature else 0)
            + (1 if self.use_speed_history_mask else 0)
            + self.time_feat_dim
        )
        self.fixed_edge_feat_dim = 1 + (1 if self.use_fixed_edge_length_feature else 0)
        self.speed_node_input_dim = (
            1
            + (1 if self.use_speed_history_mask else 0)
            + self.time_feat_dim
        )
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(self.edge_input_dim, hidden_dim)
        self.speed_node_proj = nn.Linear(self.speed_node_input_dim, hidden_dim)

        # ── Node path — fixed topology ──
        self.n_gtcn_fix = nn.ModuleList()
        self.n_gat_fix = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_fix.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_fix.append(
                SparseGATLayer(hidden_dim, d_per_head, num_heads, edge_in=edge_feat_dim, concat=True)
            )

        # ── Node path — adaptive topology ──
        self.n_gtcn_adp = nn.ModuleList()
        self.n_gat_adp = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_adp.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_adp.append(
                GATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
            )

        # ── Fusion ──
        self.fusion = GatedFusion(hidden_dim)

        # ── Edge path ──
        self.e_gtcn = nn.ModuleList()
        self.e_gat = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.e_gtcn.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.e_gat.append(
                SparseGATLayer(hidden_dim, d_per_head, num_heads, concat=True)
            )

        # ── Speed-only node path for sensor-node datasets ──
        self.vn_gtcn_fix = nn.ModuleList()
        self.vn_gat_fix = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.vn_gtcn_fix.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.vn_gat_fix.append(
                SparseGATLayer(
                    hidden_dim,
                    d_per_head,
                    num_heads,
                    edge_in=self.fixed_edge_feat_dim,
                    concat=True,
                )
            )

        if self.speed_use_adaptive:
            speed_items = int(edge_index.shape[0])
            self.speed_node_emb_src = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
            self.speed_node_emb_dst = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
            self.vn_gtcn_adp = nn.ModuleList()
            self.vn_gat_adp = nn.ModuleList()
            for _ in range(num_st_blocks):
                self.vn_gtcn_adp.append(
                    GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
                )
                self.vn_gat_adp.append(
                    GATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
                )
            self.speed_node_fusion = GatedFusion(hidden_dim)

            self.speed_emb_src = nn.Parameter(torch.randn(speed_items, adaptive_emb) * 0.1)
            self.speed_emb_dst = nn.Parameter(torch.randn(speed_items, adaptive_emb) * 0.1)
            self.e_gtcn_adp = nn.ModuleList()
            self.e_gat_adp = nn.ModuleList()
            for _ in range(num_st_blocks):
                self.e_gtcn_adp.append(
                    GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
                )
                self.e_gat_adp.append(
                    SparseGATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
                )
            self.speed_fusion = GatedFusion(hidden_dim)

        # ── 輸出頭 ──
        self.demand_head = nn.Linear(hidden_dim, pred_horizon)
        self.supply_head = nn.Linear(hidden_dim, pred_horizon)
        self.speed_head = nn.Linear(hidden_dim, pred_horizon)

    # ── helpers ────────────────────────────────────────────────

    def _adaptive_adj_from(
        self,
        emb_src: torch.Tensor,
        emb_dst: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        scores = F.relu(emb_src @ emb_dst.T)
        num_nodes = scores.shape[0]
        keep_topk = topk > 0 and topk < num_nodes
        if keep_topk:
            diag_idx = torch.arange(num_nodes, device=scores.device)
            scores_for_topk = scores.clone()
            scores_for_topk[diag_idx, diag_idx] = torch.inf
            _, topk_idx = scores_for_topk.topk(topk, dim=1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
        else:
            mask = torch.ones_like(scores, dtype=torch.bool)

        diag_idx = torch.arange(num_nodes, device=scores.device)
        mask[diag_idx, diag_idx] = True
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        adj = F.softmax(masked_scores, dim=1)
        return torch.nan_to_num(adj, nan=0.0)

    def _adaptive_adj(self) -> torch.Tensor:
        return self._adaptive_adj_from(self.emb_src, self.emb_dst, self.adaptive_topk)

    def _speed_node_adaptive_adj(self) -> torch.Tensor:
        return self._adaptive_adj_from(
            self.speed_node_emb_src,
            self.speed_node_emb_dst,
            self.speed_adaptive_topk,
        )

    def _speed_edge_adaptive_adj(self) -> torch.Tensor:
        return self._adaptive_adj_from(
            self.speed_emb_src,
            self.speed_emb_dst,
            self.speed_adaptive_topk,
        )

    def _fixed_edge_feat(self, speed: torch.Tensor) -> torch.Tensor:
        """
        speed : (BT, |E|)  — 每條邊在某時間步的速度
        return: (BT, |A_fixed|, 2) — [road_length, speed]
        """
        BT = speed.shape[0]
        num_fixed_edges = self.fixed_edge_ids.shape[0]
        feat = torch.zeros(BT, num_fixed_edges, 2, device=speed.device, dtype=speed.dtype)
        feat[:, :, 0] = self.fixed_edge_lengths.to(speed.dtype).unsqueeze(0)
        feat[:, self.fixed_edge_has_road, 1] = speed[:, self.fixed_edge_ids[self.fixed_edge_has_road]]
        return feat

    def _fixed_speed_node_edge_feat(
        self,
        batch_time: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_fixed_edge_length_feature:
            parts.append(self.fixed_edge_lengths.to(device=device, dtype=dtype).unsqueeze(0))
        parts.append(self.fixed_edge_weights.to(device=device, dtype=dtype).unsqueeze(0))
        feat = torch.stack(parts, dim=-1)
        return feat.expand(batch_time, -1, -1)

    def _build_edge_input(
        self,
        speed_seq: torch.Tensor,
        temporal_feat_seq: torch.Tensor | None = None,
        speed_history_mask_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [speed_seq.unsqueeze(-1)]
        if self.use_edge_domain_length_feature:
            edge_lengths = self.edge_lengths.to(device=speed_seq.device, dtype=speed_seq.dtype)
            length_feat = edge_lengths.view(1, -1, 1, 1).expand(
                speed_seq.shape[0],
                -1,
                speed_seq.shape[2],
                -1,
            )
            parts.append(length_feat)
        if self.use_speed_history_mask:
            if speed_history_mask_seq is None:
                mask_feat = torch.ones_like(speed_seq, dtype=speed_seq.dtype).unsqueeze(-1)
            else:
                mask_feat = speed_history_mask_seq.to(dtype=speed_seq.dtype).unsqueeze(-1)
            parts.append(mask_feat)
        if self.time_feat_dim > 0:
            if temporal_feat_seq is None:
                temporal_feat_seq = speed_seq.new_zeros(
                    speed_seq.shape[0],
                    speed_seq.shape[2],
                    self.time_feat_dim,
                )
            temporal_feat = temporal_feat_seq.unsqueeze(1).expand(-1, speed_seq.shape[1], -1, -1)
            parts.append(temporal_feat)
        return torch.cat(parts, dim=-1)

    def _build_speed_node_input(
        self,
        speed_seq: torch.Tensor,
        temporal_feat_seq: torch.Tensor | None = None,
        speed_history_mask_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [speed_seq.unsqueeze(-1)]
        if self.use_speed_history_mask:
            if speed_history_mask_seq is None:
                mask_feat = torch.ones_like(speed_seq, dtype=speed_seq.dtype).unsqueeze(-1)
            else:
                mask_feat = speed_history_mask_seq.to(dtype=speed_seq.dtype).unsqueeze(-1)
            parts.append(mask_feat)
        if self.time_feat_dim > 0:
            if temporal_feat_seq is None:
                temporal_feat_seq = speed_seq.new_zeros(
                    speed_seq.shape[0],
                    speed_seq.shape[2],
                    self.time_feat_dim,
                )
            temporal_feat = temporal_feat_seq.unsqueeze(1).expand(-1, speed_seq.shape[1], -1, -1)
            parts.append(temporal_feat)
        return torch.cat(parts, dim=-1)

    def _run_fixed_node_path(
        self,
        node_h: torch.Tensor,
        speed_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-topology node path with sparse edge-aware attention."""
        B, N, T, _ = node_h.shape
        x = node_h
        sp_flat = speed_seq.permute(0, 2, 1).reshape(B * T, -1)
        edge_feat = self._fixed_edge_feat(sp_flat)
        for gtcn, gat in zip(self.n_gtcn_fix, self.n_gat_fix):
            x = gtcn(x)                                              # (B, N, T, C)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)    # (BT, N, C)
            x_flat = gat(x_flat, self.fixed_edge_index, edge_feat)   # (BT, N, C)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)     # (B, N, T, C)
        return x[:, :, -1, :]                                        # (B, N, C)

    def _run_adaptive_node_path(
        self,
        node_h: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive node path driven by a learned sparse adjacency prior."""
        B, N, T, _ = node_h.shape
        x = node_h
        adj_adp = self._adaptive_adj()
        for gtcn, gat in zip(self.n_gtcn_adp, self.n_gat_adp):
            x = gtcn(x)                                              # (B, N, T, C)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)    # (BT, N, C)
            x_flat = gat(x_flat, adj_adp)                            # (BT, N, C)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)     # (B, N, T, C)
        return x[:, :, -1, :]                                        # (B, N, C)

    def _run_fixed_speed_node_path(
        self,
        speed_node_h: torch.Tensor,
    ) -> torch.Tensor:
        B, N, T, _ = speed_node_h.shape
        x = speed_node_h
        edge_feat = self._fixed_speed_node_edge_feat(
            B * T,
            device=speed_node_h.device,
            dtype=speed_node_h.dtype,
        )
        for gtcn, gat in zip(self.vn_gtcn_fix, self.vn_gat_fix):
            x = gtcn(x)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)
            x_flat = gat(x_flat, self.fixed_edge_index, edge_feat)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        return x[:, :, -1, :]

    def _run_adaptive_speed_node_path(
        self,
        speed_node_h: torch.Tensor,
    ) -> torch.Tensor:
        B, N, T, _ = speed_node_h.shape
        x = speed_node_h
        adj_adp = self._speed_node_adaptive_adj()
        for gtcn, gat in zip(self.vn_gtcn_adp, self.vn_gat_adp):
            x = gtcn(x)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)
            x_flat = gat(x_flat, adj_adp)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        return x[:, :, -1, :]

    def _run_fixed_edge_path(
        self,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed line-graph edge path."""
        B, nE, T, _ = edge_h.shape
        x_e = edge_h
        for gtcn, gat in zip(self.e_gtcn, self.e_gat):
            x_e = gtcn(x_e)                                         # (B, |E|, T, C)
            x_flat = x_e.permute(0, 2, 1, 3).reshape(B * T, nE, -1)
            x_flat = gat(x_flat, self.line_edge_index)
            x_e = x_flat.reshape(B, T, nE, -1).permute(0, 2, 1, 3)
        return x_e[:, :, -1, :]                                     # (B, |E|, C)

    def _run_adaptive_edge_path(
        self,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        B, nE, T, _ = edge_h.shape
        x_e = edge_h
        adj_adp = self._speed_edge_adaptive_adj()
        recv, send = torch.nonzero(adj_adp > 0, as_tuple=True)
        adaptive_edge_index = torch.stack([recv, send], dim=0).long()
        adaptive_edge_weight = adj_adp[recv, send]
        for gtcn, gat in zip(self.e_gtcn_adp, self.e_gat_adp):
            x_e = gtcn(x_e)
            x_flat = x_e.permute(0, 2, 1, 3).reshape(B * T, nE, -1)
            x_flat = gat(x_flat, adaptive_edge_index, edge_weight=adaptive_edge_weight)
            x_e = x_flat.reshape(B, T, nE, -1).permute(0, 2, 1, 3)
        return x_e[:, :, -1, :]

    def forward_v(
        self,
        speed_seq: torch.Tensor,
        temporal_feat_seq: torch.Tensor | None = None,
        speed_history_mask_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.v_domain == "node":
            speed_node_h = self.speed_node_proj(
                self._build_speed_node_input(
                    speed_seq,
                    temporal_feat_seq,
                    speed_history_mask_seq,
                )
            )
            h_fix = self._run_fixed_speed_node_path(speed_node_h) if self.speed_use_fixed_graph else None
            h_adp = self._run_adaptive_speed_node_path(speed_node_h) if self.speed_use_adaptive else None
            if h_fix is not None and h_adp is not None:
                h_speed = self.speed_node_fusion(h_fix, h_adp)
            elif h_fix is not None:
                h_speed = h_fix
            elif h_adp is not None:
                h_speed = h_adp
            else:
                raise RuntimeError("No speed node path is enabled.")
            return self.speed_head(h_speed)

        edge_h = self.edge_proj(
            self._build_edge_input(
                speed_seq,
                temporal_feat_seq,
                speed_history_mask_seq,
            )
        )
        h_edge = self._run_fixed_edge_path(edge_h) if self.speed_use_fixed_graph else None
        h_adp = self._run_adaptive_edge_path(edge_h) if self.speed_use_adaptive else None
        if h_edge is not None and h_adp is not None:
            h_edge = self.speed_fusion(h_edge, h_adp)
        elif h_edge is None and h_adp is not None:
            h_edge = h_adp
        elif h_edge is None:
            raise RuntimeError("No speed edge path is enabled.")
        return self.speed_head(h_edge)

    def speed_adaptive_graph_summary(self) -> dict[str, float | int | str] | None:
        if not self.speed_use_adaptive:
            return None
        with torch.no_grad():
            if self.v_domain == "node":
                adj = self._speed_node_adaptive_adj()
                domain = "node"
            else:
                adj = self._speed_edge_adaptive_adj()
                domain = "edge"
            kept = adj > 0
            degree = kept.sum(dim=1).float()
            diag = torch.arange(adj.shape[0], device=adj.device)
            self_loops = int(kept[diag, diag].sum().item())
            return {
                "num_edges": int(kept.sum().item()),
                "degree_mean": float(degree.mean().item()),
                "degree_min": int(degree.min().item()),
                "degree_max": int(degree.max().item()),
                "self_loops_kept": self_loops,
                "density": float(kept.float().mean().item()),
                "domain": domain,
            }

    # ── forward ───────────────────────────────────────────────

    def forward(
        self, node_seq: torch.Tensor, speed_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_seq  : (B, N, T, 2) — [demand, vacant] 歷史序列
        speed_seq : (B, |E|, T)  — 歷史邊速度

        Returns
        -------
        demand_pred : (B, N, p)
        supply_pred : (B, N, p)
        speed_pred  : (B, |E|, p)
        """
        B, N, T, _ = node_seq.shape

        temporal_feat_seq = node_seq[:, 0, :, 2:] if self.time_feat_dim > 0 else None

        node_h = self.node_proj(node_seq)                            # (B, N, T, C)
        edge_h = self.edge_proj(
            self._build_edge_input(speed_seq, temporal_feat_seq)
        )                                                            # (B, |E|, T, C)

        if self.node_use_fixed_graph:
            h_fix = self._run_fixed_node_path(node_h, speed_seq)
        else:
            h_fix = None
        if self.node_use_adaptive:
            h_adp = self._run_adaptive_node_path(node_h)
        else:
            h_adp = None
        if h_fix is not None and h_adp is not None:
            h_node = self.fusion(h_fix, h_adp)                      # (B, N, C)
        elif h_fix is not None:
            h_node = h_fix
        elif h_adp is not None:
            h_node = h_adp
        else:
            raise RuntimeError("No node path is enabled.")

        # ── Edge path ──
        h_edge = self._run_fixed_edge_path(edge_h) if self.speed_use_fixed_graph else None
        h_edge_adp = self._run_adaptive_edge_path(edge_h) if self.speed_use_adaptive else None
        if h_edge is not None and h_edge_adp is not None:
            h_edge = self.speed_fusion(h_edge, h_edge_adp)
        elif h_edge is None and h_edge_adp is not None:
            h_edge = h_edge_adp
        elif h_edge is None:
            raise RuntimeError("No speed edge path is enabled.")

        # ── 輸出 ──
        demand_pred = self.demand_head(h_node)                       # (B, N, p)
        supply_pred = self.supply_head(h_node)                       # (B, N, p)
        speed_pred = self.speed_head(h_edge)                         # (B, |E|, p)

        return demand_pred, supply_pred, speed_pred
