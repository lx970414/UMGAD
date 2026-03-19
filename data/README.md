# data/

把数据集放在这里（不随代码仓库分发）。

推荐格式：`.npz`，包含：
- `x`: [N,F] float32
- `edge_index_0 ... edge_index_{R-1}`: [2,E] int64
- `y`(可选): [N] int64，仅用于评估

示例：
```
data/
  yelp.npz
  amazon.npz
```
