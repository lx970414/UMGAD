from __future__ import annotations
from typing import Tuple, Optional, Set
import torch

def attribute_swap_augment(
    x: torch.Tensor,
    swap_ratio: float,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replacement-based augmentation (Eq. 10).
    Pick subset V_aa, for each i in V_aa replace x(i) with x(j) from random j.
    Returns (x_aug, idx_aug_nodes).
    """
    N = x.size(0)
    m = max(1, int(swap_ratio * N))
    perm = torch.randperm(N, generator=generator, device=x.device)
    idx = perm[:m]
    j = torch.randint(0, N, (m,), generator=generator, device=x.device)
    x_aug = x.clone()
    x_aug[idx] = x[j]
    return x_aug, idx

def rwr_subgraph_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    target_size: int,
    restart: float,
    steps: int,
    seed_nodes: torch.Tensor,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Random-walk-with-restart subgraph node sampling.
    Returns sampled node indices (unique).
    """
    # build adjacency list (CPU-friendly); for huge graphs, use CSR in practice.
    # Here we implement a simple torch-based neighbor sampling.
    row, col = edge_index
    # create neighbor lists via scatter; fallback to python lists on cpu
    device = edge_index.device
    if device.type != "cpu":
        # move to cpu for neighbor list (lightweight)
        row_cpu, col_cpu = row.cpu(), col.cpu()
    else:
        row_cpu, col_cpu = row, col

    neighbors = [[] for _ in range(num_nodes)]
    for r, c in zip(row_cpu.tolist(), col_cpu.tolist()):
        neighbors[r].append(c)

    sampled: Set[int] = set(int(s) for s in seed_nodes.cpu().tolist())
    cur = int(seed_nodes[0].item())
    for _ in range(steps):
        if torch.rand((), generator=generator).item() < restart:
            cur = int(seed_nodes[torch.randint(0, seed_nodes.numel(), (1,), generator=generator)].item())
        else:
            nb = neighbors[cur]
            if len(nb) > 0:
                cur = nb[int(torch.randint(0, len(nb), (1,), generator=generator).item())]
        sampled.add(cur)
        if len(sampled) >= target_size:
            break
    return torch.tensor(sorted(sampled), device=device, dtype=torch.long)

def mask_subgraph_edges(
    edge_index: torch.Tensor,
    sub_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove edges whose both endpoints are inside the sampled subgraph.
    Returns (edge_index_remain, masked_edges).
    """
    device = edge_index.device
    Nsub = sub_nodes.numel()
    sub_set = torch.zeros(int(edge_index.max().item()) + 1, device=device, dtype=torch.bool)
    sub_set[sub_nodes] = True
    src, dst = edge_index
    inside = sub_set[src] & sub_set[dst]
    masked = edge_index[:, inside]
    remain = edge_index[:, ~inside]
    return remain, masked


def subgraph_swap_and_mask(
    x: torch.Tensor,
    sub_nodes: torch.Tensor,
    mask_token: torch.Tensor,
    swap_inside: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate subgraph-level augmented node attributes X^r,k_s and apply masking.

    In UMGAD, the subgraph-level augmented graph is defined as
      G^{r,k}_s = (V, E^r \ E^r_s, X^{r,k}_s)
    where the sampled subgraph nodes are perturbed and then masked for reconstruction.

    We follow the paper's wording consistently with the attribute-level augmentation:
      1) replacement-based augmentation (swap attributes with random nodes, no new content)
      2) mask the augmented node subset (here: sampled subgraph nodes)

    Args:
        x: Original node features [N, F]
        sub_nodes: Sampled subgraph node indices
        mask_token: Learnable [F] token
        swap_inside: Whether to perform attribute swapping before masking.
        generator: Random generator.

    Returns:
        x_masked: Augmented+masked feature matrix.
    """
    x_aug = x.clone()
    if swap_inside and sub_nodes.numel() > 0:
        N = x.size(0)
        j = torch.randint(0, N, (sub_nodes.numel(),), generator=generator, device=x.device)
        x_aug[sub_nodes] = x[j]
    # mask exactly the sampled subgraph nodes
    if sub_nodes.numel() > 0:
        x_aug[sub_nodes] = mask_token
    return x_aug
