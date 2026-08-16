#!/usr/bin/env python3
"""Build the public ST-GAPV core and exercise GAP/V forward paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stgat_model import STGATPredictor


def main() -> int:
    torch.manual_seed(42)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    edge_lengths = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
    adjacency = torch.zeros((3, 3), dtype=torch.float32)
    adjacency[edge_index[:, 0], edge_index[:, 1]] = 1.0
    model = STGATPredictor(
        num_nodes=3,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        adj_matrix=adjacency,
        hidden_dim=8,
        num_heads=2,
        num_st_blocks=2,
        num_gtcn_layers=1,
        kernel_size=3,
        pred_horizon=4,
        adaptive_emb=4,
        adaptive_topk=3,
        node_feat_dim=4,
        speed_use_fixed_graph=True,
        speed_use_adaptive=True,
        v_domain="edge",
    ).eval()
    node_sequence = torch.randn(2, 3, 6, 4)
    speed_sequence = torch.rand(2, 3, 6) * 30.0
    with torch.no_grad():
        demand, supply, speed = model(node_sequence, speed_sequence)
        speed_only = model.forward_v(
            speed_sequence,
            temporal_feat_seq=node_sequence[:, 0, :, 2:],
        )
    expected = ((2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4))
    actual = tuple(tuple(tensor.shape) for tensor in (demand, supply, speed, speed_only))
    if actual != expected or not all(
        torch.isfinite(tensor).all().item()
        for tensor in (demand, supply, speed, speed_only)
    ):
        raise RuntimeError(f"ST-GAPV smoke test failed: shapes={actual}")
    print("ST-GAPV smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
