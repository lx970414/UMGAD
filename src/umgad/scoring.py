from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn.functional as F

def anomaly_score(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    z_list: List[torch.Tensor],
    edge_index_list: List[torch.Tensor],
    eps: float = 0.5
) -> torch.Tensor:
    """Compute node anomaly score S(i)* for one view (Eq. 19) without dense adjacency.

    - attribute error: L1 distance between x_hat and x per node
    - structure error: for each relation r, measure reconstruction error on observed edges:
        per node i: mean_{(i,j) in E_r} (1 - sigmoid(z_i · z_j))
      This is a proxy for ||A_hat_row - A_row|| and works without dense matrices.
    """
    attr_err = (x_hat - x).abs().sum(dim=-1)  # L1

    struct_errs = []
    for z, ei in zip(z_list, edge_index_list):
        src, dst = ei
        p = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
        # edge error: want 1, so error is (1-p)
        e_err = (1.0 - p)
        # aggregate per source node
        N = x.size(0)
        ssum = torch.zeros(N, device=x.device)
        scnt = torch.zeros(N, device=x.device)
        ssum.index_add_(0, src, e_err)
        scnt.index_add_(0, src, torch.ones_like(e_err))
        per_node = ssum / (scnt + 1e-12)
        struct_errs.append(per_node)
    struct_err = torch.stack(struct_errs, dim=0).mean(dim=0)

    return eps * attr_err + (1.0 - eps) * struct_err

def mean_multi_view(scores: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(scores, dim=0).mean(dim=0)
