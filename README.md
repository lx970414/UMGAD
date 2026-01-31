# UMGAD (ICDE 2025) — Reproduction Code (Unofficial)

This repo re-implements **UMGAD: Unsupervised Multiplex Graph Anomaly Detection** as described in the ICDE'25 paper.
It follows the paper's training objective (original-view reconstruction + augmented-view reconstruction + dual-view contrastive learning)
and the **label-free anomaly-score threshold selection** strategy.

> **Note**: Datasets are not bundled. This repo supports a simple `.npz` format for multiplex graphs so you can plug in your data.
> You can also use the built-in anomaly injection routine described in the paper.

## 1. Install

```bash
pip install -r requirements.txt
```

PyG wheels depend on your CUDA/torch version. If `pip` fails for `torch-geometric`,
install from the official PyG instructions.

## 2. Data format

Provide a `.npz` with:
- `x`: float32 array of shape `[N, F]`
- `y` (optional): int64 array of shape `[N]` (0=normal, 1=anomaly) for evaluation only
- For each relation `r` (0..R-1): `edge_index_r`: int64 array of shape `[2, E_r]` (COO, undirected or directed)

Example keys:
`x`, `edge_index_0`, `edge_index_1`, `edge_index_2`, `y`

## 3. Train

```bash
python train.py --data path/to/graph.npz --epochs 300 --device cuda
```

Outputs:
- `scores.npy`: node anomaly scores
- `pred.npy`: predicted labels from the paper's thresholding strategy
- `metrics.json` (if `y` exists): AUC, Macro-F1, etc.

## 4. Key implementation notes

- **Original-view attribute masking**: mask nodes with learnable `[MASK]` token and reconstruct masked attributes (Eq. 1–4).
- **Original-view edge masking**: mask edges and predict them with sampled softmax / cross-entropy (Eq. 5–8).
- **Augmented-view attribute-level**: swap features for a node subset, then mask exactly those swapped nodes (Eq. 10–13).
- **Augmented-view subgraph-level**: sample a subgraph with random-walk-with-restart and mask its edges (Eq. 14–16).
- **Dual-view contrastive**: contrast original-view vs attribute-aug view, and original-view vs subgraph-aug view (Eq. 17).
- **Total loss**: Eq. 18.
- **Anomaly score**: combines attribute reconstruction error + structure reconstruction error (Eq. 19).
- **Unsupervised threshold**: moving average smoothing + 2nd-order difference inflection point (Eq. 20–23).

## 5. Repro tips

- For large graphs (millions of nodes), full-matrix reconstruction is infeasible.
  This code uses **edge-wise** scoring and **negative sampling** to avoid dense adjacency materialization.
- Use `--num_neg` to trade speed vs accuracy.

## 6. Citation

If you use this repo, please cite the original ICDE 2025 paper (UMGAD).


## 推荐运行方式（与 BPHGNN 类似的 src 结构）

```bash
bash run.sh --data data/your_graph.npz --device cuda
```
