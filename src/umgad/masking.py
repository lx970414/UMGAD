from __future__ import annotations
from typing import Tuple, List, Optional
import torch

def mask_node_attributes(
    x: torch.Tensor,
    mask_ratio: float,
    mask_token: torch.Tensor,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform random node masking without replacement.
    Returns (x_masked, mask_idx).
    """
    N = x.size(0)
    m = max(1, int(mask_ratio * N))
    perm = torch.randperm(N, generator=generator, device=x.device)
    mask_idx = perm[:m]
    x_masked = x.clone()
    x_masked[mask_idx] = mask_token
    return x_masked, mask_idx

def mask_edges(
    edge_index: torch.Tensor,
    mask_ratio: float,
    num_nodes: int,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask edges by dropping a random subset.
    Returns (edge_index_remain, edge_index_masked_pos).
    """
    E = edge_index.size(1)
    m = max(1, int(mask_ratio * E))
    perm = torch.randperm(E, generator=generator, device=edge_index.device)
    mask_e_idx = perm[:m]
    remain_e_idx = perm[m:]
    masked_pos = edge_index[:, mask_e_idx]
    remain = edge_index[:, remain_e_idx]
    return remain, masked_pos
