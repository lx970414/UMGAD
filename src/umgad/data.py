from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch

@dataclass
class MultiplexGraph:
    x: torch.Tensor                     # [N, F]
    edge_index: List[torch.Tensor]      # length R, each [2, E_r]
    y: Optional[torch.Tensor] = None    # [N]

def load_npz(path: str, device: Optional[str] = None) -> MultiplexGraph:
    d = np.load(path, allow_pickle=True)
    x = torch.from_numpy(d["x"]).float()
    y = torch.from_numpy(d["y"]).long() if "y" in d.files else None

    # infer relations by edge_index_{r}
    rel_keys = sorted([k for k in d.files if k.startswith("edge_index_")],
                      key=lambda s: int(s.split("_")[-1]))
    edge_index = []
    for k in rel_keys:
        ei = torch.from_numpy(d[k]).long()
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"{k} must have shape [2, E], got {ei.shape}")
        edge_index.append(ei)
    if len(edge_index) == 0:
        raise ValueError("No edge_index_{r} found in npz.")

    if device is not None:
        x = x.to(device)
        edge_index = [ei.to(device) for ei in edge_index]
        if y is not None:
            y = y.to(device)

    return MultiplexGraph(x=x, edge_index=edge_index, y=y)

def inject_anomalies(
    x: np.ndarray,
    edge_index_list: List[np.ndarray],
    num_struct_cliques: int,
    clique_size: int,
    seed: int = 0
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Inject structural + attribute anomalies following the paper's description.

    Structural: create `num_struct_cliques` cliques of size `clique_size`, assign clique edges to random relation(s).
    Attribute: choose same number of nodes and replace one node's attribute with another node's attribute.

    Returns (x_new, edge_index_list_new, y) where y is 0/1.
    """
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    R = len(edge_index_list)

    y = np.zeros(N, dtype=np.int64)

    # structural anomalies
    anomalies = set()
    for _ in range(num_struct_cliques):
        nodes = rng.choice(N, size=clique_size, replace=False)
        for n in nodes:
            anomalies.add(int(n))
        # fully connect clique
        # pick one or more relation types; here choose 1 relation uniformly (simplification)
        r = int(rng.integers(0, R))
        ei = edge_index_list[r]
        # add undirected edges (u,v) for all pairs
        new_edges = []
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                u, v = int(nodes[i]), int(nodes[j])
                new_edges.append((u, v))
                new_edges.append((v, u))
        if len(new_edges) > 0:
            add = np.array(new_edges, dtype=np.int64).T
            edge_index_list[r] = np.concatenate([ei, add], axis=1)

    # attribute anomalies (same count)
    num_attr = len(anomalies)
    attr_nodes = rng.choice(N, size=num_attr, replace=False)
    x_new = x.copy()
    for i in attr_nodes:
        j = int(rng.integers(0, N))
        x_new[i] = x_new[j]

    for i in anomalies:
        y[i] = 1
    for i in attr_nodes:
        y[int(i)] = 1

    return x_new, edge_index_list, y
