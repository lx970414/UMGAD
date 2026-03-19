from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

def cosine_recon_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    eta: float = 1.0
) -> torch.Tensor:
    """Cosine distance ^ eta averaged over idx nodes (Eq. 4/13/15)."""
    xh = x_hat[idx]
    xo = x[idx]
    cos = F.cosine_similarity(xh, xo, dim=-1).clamp(-1, 1)
    loss = (1.0 - cos).pow(eta).mean()
    return loss

def sampled_edge_softmax_loss(
    z: torch.Tensor,
    pos_edges: torch.Tensor,
    edge_index_all: torch.Tensor,
    num_nodes: int,
    num_neg: int = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cross-entropy on masked edges with negative sampling (Eq. 7 / 15).
    For each positive edge (v,u), sample `num_neg` negative endpoints u'.
    Denominator approximates sum over unmasked edges.
    """
    v = pos_edges[0]
    u = pos_edges[1]
    # dot for positives
    pos_score = (z[v] * z[u]).sum(dim=-1)  # [P]

    # negative sampling: keep v fixed, sample random u'
    P = v.numel()
    neg_u = torch.randint(0, num_nodes, (P, num_neg), device=z.device, generator=generator)
    neg_score = (z[v].unsqueeze(1) * z[neg_u]).sum(dim=-1)  # [P, num_neg]

    # log-softmax over (pos + neg)
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # [P, 1+num_neg]
    log_prob_pos = F.log_softmax(logits, dim=1)[:, 0]
    return (-log_prob_pos).mean()

def contrastive_pair_loss(
    h_anchor: torch.Tensor,
    h_pos: torch.Tensor,
    num_neg: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Implements Eq. 17 in the paper in a simplified way.
    For each node i: positive (i in anchor, i in pos). Negatives: random j.
    Denominator uses exp(anchor·anchor_j) + exp(anchor·pos_j).
    """
    N = h_anchor.size(0)
    # sample negatives j (one per i by default, consistent with Eq.17 notation)
    j = torch.randint(0, N, (N,), device=h_anchor.device, generator=generator)
    sim_pos = (h_anchor * h_pos).sum(dim=-1)                # [N]
    sim_aa  = (h_anchor * h_anchor[j]).sum(dim=-1)          # [N]
    sim_ap  = (h_anchor * h_pos[j]).sum(dim=-1)             # [N]
    denom = torch.exp(sim_aa) + torch.exp(sim_ap)
    loss = -(sim_pos - torch.log(denom + 1e-12)).mean()
    return loss
