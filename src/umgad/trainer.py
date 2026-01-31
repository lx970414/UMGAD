from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import UMGADConfig
from .model import UMGAD
from .masking import mask_node_attributes, mask_edges
from .augment import attribute_swap_augment, rwr_subgraph_nodes, mask_subgraph_edges, subgraph_swap_and_mask
from .losses import cosine_recon_loss, sampled_edge_softmax_loss, contrastive_pair_loss
from .scoring import anomaly_score, mean_multi_view

class UMGADTrainer:
    def __init__(self, model: UMGAD, cfg: UMGADConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(cfg.seed)

    def train(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> None:
        cfg = self.cfg
        model = self.model

        for epoch in tqdm(range(cfg.epochs), desc="train"):
            model.train()
            self.opt.zero_grad()

            # ----- Original-view attribute masking (Eq.1-4) -----
            attr_losses = []
            x_ma_recons = []
            z_ma_list_all = []
            mask_idx_list = []

            for k in range(cfg.num_repeats):
                x_masked, mask_idx = mask_node_attributes(x, cfg.mask_ratio_attr, model.mask_token, self.g)
                mask_idx_list.append(mask_idx)
                # per relation encode
                z_list = model.encode_attr_orig(x_masked, edge_index_list)
                z_ma_list_all.append(z_list)
                # decode attribute from fused embedding
                z_fused = model.fuse_attr(z_list)
                x_hat = model.dec_attr_orig(z_fused)
                x_ma_recons.append((x_hat, mask_idx))
                attr_losses.append(cosine_recon_loss(x_hat, x, mask_idx, eta=cfg.eta))

            LA = torch.stack(attr_losses).mean()

            # ----- Original-view edge masking (Eq.5-8) -----
            struct_losses_r = []
            z_ms_list_all = []
            for r, ei in enumerate(edge_index_list):
                l_r = []
                z_r_allk = []
                for k in range(cfg.num_repeats):
                    ei_remain, ei_pos = mask_edges(ei, cfg.mask_ratio_edge, x.size(0), self.g)
                    z = model.enc_edge_orig[r](x, ei_remain)
                    z_r_allk.append(z)
                    l_r.append(sampled_edge_softmax_loss(z, ei_pos, ei_remain, x.size(0), num_neg=cfg.num_neg, generator=self.g))
                struct_losses_r.append(torch.stack(l_r).mean())
                z_ms_list_all.append(torch.stack(z_r_allk).mean(dim=0))
            LS = model.fuse_struct_loss(struct_losses_r)
            LO = cfg.alpha * LA + (1.0 - cfg.alpha) * LS  # Eq.9

            # ----- Attribute-level augmented view (Eq.10-13) -----
            la_aug_list = []
            z_aa_fused_list = []
            for k in range(cfg.num_repeats):
                x_aug, idx_aug = attribute_swap_augment(x, cfg.aug_ratio_attr_swap, self.g)
                # mask exactly augmented nodes
                x_aug_masked = x_aug.clone()
                x_aug_masked[idx_aug] = model.mask_token
                z_list = model.encode_attr_aug(x_aug_masked, edge_index_list)
                z_fused = model.fuse_attr(z_list)
                x_hat = model.dec_attr_aug(z_fused)
                la_aug_list.append(cosine_recon_loss(x_hat, x, idx_aug, eta=cfg.eta))
                z_aa_fused_list.append(z_fused)
            LA_Aug = torch.stack(la_aug_list).mean()

            # ----- Subgraph-level augmented view (Eq.14-16) -----
            # In the paper, subgraph-level augmentation is repeated K times for each relation.
            # We sample subgraphs via RWR (cf. DualGAD) and apply subgraph masking:
            #   - mask edges inside the sampled subgraph (E^r_s)
            #   - generate X^{r,k}_s via replacement-based perturbation, then mask the same nodes
            lsa_allk: List[torch.Tensor] = []
            lss_allk_rel = [ [] for _ in range(model.num_relations) ]
            z_sa_fused_list: List[torch.Tensor] = []

            target_size = max(5, int(cfg.aug_ratio_subgraph * x.size(0)))
            for k in range(cfg.num_repeats):
                z_rel_k = []
                for r, ei in enumerate(edge_index_list):
                    seed_nodes = torch.randint(
                        0, x.size(0), (cfg.subgraph_seed_per_repeat,),
                        generator=self.g, device=x.device
                    )
                    sub_nodes = rwr_subgraph_nodes(
                        ei, x.size(0), target_size,
                        restart=cfg.rwr_restart,
                        steps=cfg.rwr_steps,
                        seed_nodes=seed_nodes,
                        generator=self.g,
                    )
                    # subgraph masking: drop edges inside subgraph
                    ei_remain, ei_masked = mask_subgraph_edges(ei, sub_nodes)
                    # generated attribute matrix + mask subgraph nodes (consistent with Eq.10-13 style)
                    x_s_masked = subgraph_swap_and_mask(
                        x, sub_nodes, model.mask_token, swap_inside=True, generator=self.g
                    )
                    # encode on remaining edges with masked attributes
                    z = model.enc_sub_aug[r](x_s_masked, ei_remain)
                    z_rel_k.append(z)

                    # attribute reconstruction on sampled nodes (Eq.15 Lsa)
                    x_hat = model.dec_attr_sub(z)
                    lsa_allk.append(cosine_recon_loss(x_hat, x, sub_nodes, eta=cfg.eta))

                    # structure reconstruction: predict masked edges (Eq.15 Lss)
                    if ei_masked.numel() > 0:
                        lss_allk_rel[r].append(
                            sampled_edge_softmax_loss(
                                z, ei_masked, ei_remain, x.size(0),
                                num_neg=cfg.num_neg, generator=self.g
                            )
                        )
                    else:
                        lss_allk_rel[r].append(torch.zeros((), device=x.device))

                # one fused subgraph-aug embedding per repeat (for contrastive loss)
                z_sa_fused_list.append(model.fuse_attr(z_rel_k))

            Lsa = torch.stack(lsa_allk).mean() if len(lsa_allk) else torch.zeros((), device=x.device)
            Lss_rel = [torch.stack(v).mean() for v in lss_allk_rel]
            Lss = model.fuse_struct_loss(Lss_rel)
            LS_Aug = cfg.beta * Lsa + (1.0 - cfg.beta) * Lss  # Eq.16

            # ----- Dual-view contrastive (Eq.17) -----
            # Use one representative original-view embedding: average over repeats, then fuse
            z_ma_fused = torch.stack([model.fuse_attr(z_list) for z_list in z_ma_list_all], dim=0).mean(dim=0)
            z_aa_fused = torch.stack(z_aa_fused_list, dim=0).mean(dim=0)
            z_sa_fused = torch.stack(z_sa_fused_list, dim=0).mean(dim=0)

            L_oa = contrastive_pair_loss(z_ma_fused, z_aa_fused, generator=self.g)
            L_os = contrastive_pair_loss(z_ma_fused, z_sa_fused, generator=self.g)
            LCL = L_oa + L_os

            # ----- Total loss (Eq.18) -----
            loss = LO + cfg.lam * LA_Aug + cfg.mu * LS_Aug + cfg.theta * LCL
            loss.backward()
            self.opt.step()

    @torch.no_grad()
    def score(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> np.ndarray:
        """Compute final anomaly score S(i) as mean over three views (Eq.19 + mean)."""
        cfg = self.cfg
        model = self.model
        model.eval()

        # Original-view embeddings: use unmasked x for scoring, but reuse encoders/decoders
        z_o_list = model.encode_attr_orig(x, edge_index_list)
        z_o = model.fuse_attr(z_o_list)
        x_hat_o = model.dec_attr_orig(z_o)

        # Attribute-aug view: one swap sample for scoring
        x_aug, idx_aug = attribute_swap_augment(x, cfg.aug_ratio_attr_swap, self.g)
        z_a_list = model.encode_attr_aug(x_aug, edge_index_list)
        z_a = model.fuse_attr(z_a_list)
        x_hat_a = model.dec_attr_aug(z_a)

        # Subgraph-aug view: sample subgraphs via RWR and apply subgraph masking
        # (consistent with training; cf. Section IV-B2).
        z_s_list: List[torch.Tensor] = []
        target_size = max(5, int(cfg.aug_ratio_subgraph * x.size(0)))
        for r, ei in enumerate(edge_index_list):
            seed_nodes = torch.randint(0, x.size(0), (cfg.subgraph_seed_per_repeat,), generator=self.g, device=x.device)
            sub_nodes = rwr_subgraph_nodes(
                ei, x.size(0), target_size,
                restart=cfg.rwr_restart, steps=cfg.rwr_steps,
                seed_nodes=seed_nodes, generator=self.g
            )
            ei_remain, _ = mask_subgraph_edges(ei, sub_nodes)
            x_s_masked = subgraph_swap_and_mask(x, sub_nodes, model.mask_token, swap_inside=True, generator=self.g)
            z_s_list.append(model.enc_sub_aug[r](x_s_masked, ei_remain))

        z_s = model.fuse_attr(z_s_list)
        x_hat_s = model.dec_attr_sub(z_s)

        s_o = anomaly_score(x_hat_o, x, z_o_list, edge_index_list, eps=cfg.eps)
        s_a = anomaly_score(x_hat_a, x, z_a_list, edge_index_list, eps=cfg.eps)
        s_s = anomaly_score(x_hat_s, x, z_s_list, edge_index_list, eps=cfg.eps)

        s = mean_multi_view([s_o, s_a, s_s]).detach().cpu().numpy()
        return s
