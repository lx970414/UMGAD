from dataclasses import dataclass

@dataclass
class UMGADConfig:
    # model
    in_dim: int
    hid_dim: int = 128
    out_dim: int = 128
    enc_layers: int = 2              # encoder depth L in paper
    dropout: float = 0.0

    # multiplex
    num_relations: int = 3

    # masking / augmentation
    mask_ratio_attr: float = 0.3     # r_m for attribute masking
    mask_ratio_edge: float = 0.3     # r_m for edge masking
    num_repeats: int = 2             # K repeats
    aug_ratio_attr_swap: float = 0.1 # |V_aa|/|V|
    aug_ratio_subgraph: float = 0.15 # approx subgraph node fraction

    # random-walk-with-restart
    rwr_restart: float = 0.5
    rwr_steps: int = 50
    subgraph_seed_per_repeat: int = 1

    # losses
    alpha: float = 0.5               # balance LA vs LS in original view (Eq. 9)
    beta: float = 0.5                # balance Lsa vs Lss in subgraph aug (Eq. 16)
    lam: float = 1.0                 # λ for LA_Aug (Eq. 18)
    mu: float = 1.0                  # μ for LS_Aug (Eq. 18)
    theta: float = 1.0               # Θ for contrastive (Eq. 18)

    eta: float = 2.0                 # scaling factor in cosine loss (Eq. 4,13,15)
    eps: float = 0.5                 # ε in anomaly score (Eq. 19)

    # structure loss sampling
    num_neg: int = 50                # negatives per positive (edge prediction)

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    seed: int = 42
