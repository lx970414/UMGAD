import argparse, json, os
import numpy as np
import torch

from src.umgad.config import UMGADConfig
from src.umgad.data import load_npz
from src.umgad.model import UMGAD
from src.umgad.trainer import UMGADTrainer
from src.umgad.threshold import predict
from src.umgad.metrics import evaluate
from src.umgad.utils import set_seed, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to .npz multiplex graph")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hid_dim", type=int, default=128)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--num_neg", type=int, default=50)
    ap.add_argument("--mask_ratio_attr", type=float, default=0.3)
    ap.add_argument("--mask_ratio_edge", type=float, default=0.3)
    ap.add_argument("--num_repeats", type=int, default=2)
    args = ap.parse_args()

    ensure_dir(args.out)
    device = torch.device(args.device)

    g = load_npz(args.data, device=str(device))
    cfg = UMGADConfig(
        in_dim=g.x.size(1),
        hid_dim=args.hid_dim,
        num_relations=len(g.edge_index),
        enc_layers=args.enc_layers,
        num_neg=args.num_neg,
        mask_ratio_attr=args.mask_ratio_attr,
        mask_ratio_edge=args.mask_ratio_edge,
        num_repeats=args.num_repeats,
    )
    if args.epochs is not None:
        cfg.epochs = args.epochs

    set_seed(cfg.seed)

    model = UMGAD(in_dim=cfg.in_dim, hid_dim=cfg.hid_dim, num_relations=cfg.num_relations, enc_layers=cfg.enc_layers)
    trainer = UMGADTrainer(model, cfg, device)

    trainer.train(g.x, g.edge_index)

    scores = trainer.score(g.x, g.edge_index)
    pred, thr, k = predict(scores)

    np.save(os.path.join(args.out, "scores.npy"), scores)
    np.save(os.path.join(args.out, "pred.npy"), pred)
    meta = {"threshold": thr, "num_pred": k}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if g.y is not None:
        y = g.y.detach().cpu().numpy()
        metrics = evaluate(y, scores, pred)
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
