# UMGAD

This is the code for the paper:
> UMGAD: Unsupervised Multiplex Graph Anomaly Detection (ICDE 2025)

## Framework
![Framework](./Framework.png)
We propose a novel framework named UMGAD for unsupervised graph anomaly detection tasks. By integrating graph autoencoders with diverse masking strategies across multiple relation types, UMGAD captures rich anomaly signals while suppressing redundancy. It leverages attribute and structure reconstruction in the augmented view, followed by anomaly aggregation via dual-view contrastive learning, ensuring robust detection.

## Dependencies
Recent versions of the following packages for Python 3.10/3.11 are required:
# torch-geometric wheels must match your torch/CUDA.
* networkx==2.8.4
* numpy==1.26.4
* PyYAML==6.0.1
* Requests==2.31.0
* scikit_learn==1.3.2
* scipy==1.11.4
* setuptools==60.2.0
* sphinx_gallery==0.15.0
* tensorboardX==2.6.2
* tqdm==4.66.4
* dgl==0.4.1

* torch>=2.1.0,<2.6.0
* torch-geometric>=2.5.0,<2.6.0

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Retail_Rocket https://tianchi.aliyun.com/competition/entrance/231719/information/
* Alibaba https://github.com/xuehansheng/DualHGCN
* Amazon https://github.com/YingtongDou/CARE-GNN/tree/master/data
* YelpChi https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset

Kindly note that there may be two versions of node features for YelpChi. The old version has a dimension of 100 and the new version is 32. In our paper, the results are reported based on the new features.

## Citation
If you find UMGAD useful in your research, please cite the following paper:
```bibtex
@inproceedings{li2025umgad,
  title={Umgad: Unsupervised multiplex graph anomaly detection},
  author={Li, Xiang and Qi, Jianpeng and Zhao, Zhongying and Zheng, Guanjie and Cao, Lei and Dong, Junyu and Yu, Yanwei},
  booktitle={2025 IEEE 41st International Conference on Data Engineering (ICDE)},
  pages={3724--3737},
  year={2025},
  organization={IEEE}
}
```
