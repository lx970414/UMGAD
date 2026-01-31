from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

def normalize_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return edge weights for GCN normalization on given edge_index."""
    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    w = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return ei, w

class SimplifiedGCN(nn.Module):
    """Simplified GCN encoder.

    参考 AnomMAN 的分析：层间非线性并非关键，移除后相当于更强的低通滤波，有利于异常检测。
    """

    """Simplified GCN encoder (no inter-layer nonlinearity) used widely in GAD papers."""
    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0, final_activation: str | None = None):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.num_layers = num_layers
        self.dropout = dropout
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        ei, ew = normalize_adj(edge_index, num_nodes)
        h = x
        for _ in range(self.num_layers):
            h = self.propagate(h, ei, ew)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin(h)
        if self.final_activation == 'relu':
            h = F.relu(h)
        elif self.final_activation == 'tanh':
            h = torch.tanh(h)
        return h

    @staticmethod
    def propagate(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))
        return out

class AttrDecoder(nn.Module):
    def __init__(self, hid_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(hid_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h)

class UMGAD(nn.Module):
    """UMGAD core model.

    It maintains per-relation encoders for:
      - original attribute masking (enc1)
      - original edge masking (enc2)
      - attribute-augmented view (enc3)
      - subgraph-augmented view (enc4)

    In the paper, weights a_r and b_r are learnable scalars to fuse relations (Eq. 3,8,12,15).
    """
    def __init__(self, in_dim: int, hid_dim: int, num_relations: int, enc_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_relations = num_relations

        def make_enc():
            return SimplifiedGCN(in_dim, hid_dim, num_layers=enc_layers, dropout=dropout)

        self.enc_attr_orig = nn.ModuleList([make_enc() for _ in range(num_relations)])
        self.enc_edge_orig = nn.ModuleList([make_enc() for _ in range(num_relations)])
        self.enc_attr_aug  = nn.ModuleList([make_enc() for _ in range(num_relations)])
        self.enc_sub_aug   = nn.ModuleList([make_enc() for _ in range(num_relations)])

        self.dec_attr_orig = AttrDecoder(hid_dim, in_dim)
        self.dec_attr_aug  = AttrDecoder(hid_dim, in_dim)
        self.dec_attr_sub  = AttrDecoder(hid_dim, in_dim)

        # Learnable fusion weights
        self.a = nn.Parameter(torch.randn(num_relations))
        self.b = nn.Parameter(torch.randn(num_relations))

        # mask token (learnable) used for attribute masking
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def fuse_attr(self, xs: List[torch.Tensor]) -> torch.Tensor:
        w = F.softmax(self.a, dim=0)  # stable, keeps weights positive
        out = 0.0
        for r, x in enumerate(xs):
            out = out + w[r] * x
        return out

    def fuse_struct_loss(self, losses: List[torch.Tensor]) -> torch.Tensor:
        w = F.softmax(self.b, dim=0)
        out = 0.0
        for r, l in enumerate(losses):
            out = out + w[r] * l
        return out

    # forward helpers
    def encode_attr_orig(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(x, edge_index_list[r]) for r, enc in enumerate(self.enc_attr_orig)]

    def encode_edge_orig(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(x, edge_index_list[r]) for r, enc in enumerate(self.enc_edge_orig)]

    def encode_attr_aug(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(x, edge_index_list[r]) for r, enc in enumerate(self.enc_attr_aug)]

    def encode_sub_aug(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(x, edge_index_list[r]) for r, enc in enumerate(self.enc_sub_aug)]
