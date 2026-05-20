## [ICML 2026] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm

This repository contains the official implementation of paper [SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](https://arxiv.org/abs/2602.08064).

### News

May 1, 2026: Paper accepted to ICML 2026.

### Abstract

------------

Modern Transformers predominantly adopt the Pre-Norm paradigm for its optimization stability, foregoing the superior potential of the unstable Post-Norm architecture. Prior attempts to combine their strengths typically lead to a stability-performance trade-off. We attribute this phenomenon to a structural incompatibility within a *single-stream* design: Any application of the Post-Norm operation inevitably obstructs the clean identity gradient preserved by Pre-Norm. To fundamentally reconcile these paradigms, we propose SiameseNorm, a *two-stream* architecture that couples Pre-Norm-like and Post-Norm-like streams with shared parameters. This design decouples the optimization dynamics of the two streams, retaining the distinct characteristics of both Pre-Norm and Post-Norm by enabling all residual blocks to receive combined gradients inherited from both paradigms, where one stream secures stability while the other enhances expressivity. Extensive pre-training experiments on 1.3B-parameter models demonstrate that SiameseNorm exhibits exceptional optimization robustness and consistently outperforms strong baselines.

<div align="center">
  <img src="assets/norms.jpg" alt="image">
</div>

Architectural comparison of Post-Norm, Pre-Norm and SiameseNorm. In SiameseNorm, the input is duplicated into parallel streams sharing identical residual updates, where distinct LN positioning differentiates the hidden states across layers.

### Installation

-----------

#### 1. Clone this repository
```
git clone https://github.com/Qwen-Applications/SiameseNorm.git
cd SiameseNorm
```
#### 2. Apply modifications
```
# 1. Initialize the submodule
git submodule update --init --recursive

# 2. Enter the submodule directory
cd OLMo

# 3. Apply the patch file
git apply ../changes.patch
# 4. Install dependencies
pip install -e .[all]
```

### Training

First, prepare the data and fill in the address in the config.


```
cd OLMo

# For Pre-Norm
export MODEL_TYPE="pre"
torchrun --nproc_per_node 8 scripts/train.py configs/exps/OLMo-1B-2e-3.yaml
# For Hyper-Connection
export MODEL_TYPE="hc"
torchrun --nproc_per_node 8 scripts/train.py configs/exps/OLMo-1B-2e-3.yaml
# For SiameseNorm(post-pre)
export MODEL_TYPE="post_pre"
torchrun --nproc_per_node 8 scripts/train.py configs/exps/OLMo-1B-2e-3.yaml
# For SiameseNorm(hybrid-pre)
export MODEL_TYPE="hybrid_pre"
torchrun --nproc_per_node 8 scripts/train.py configs/exps/OLMo-1B-hybrid-2e-3.yaml
```

### Citing this work

-----------

If you find this work helpful or use it in your research, please consider citing our paper:

```
@article{li2026siamesenorm,
  title={SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm},
  author={Li, Tianyu and Han, Dongchen and Cao, Zixuan and Huang, Haofeng and Zhou, Mengyu and Chen, Ming and Zhao, Erchao and Jiang, Xiaoxi and Jiang, Guanjun and Huang, Gao},
  journal={arXiv preprint arXiv:2602.08064},
  year={2026}
}
```

### Acknowledgment

-----------

The code is based on [OLMo](https://github.com/allenai/OLMo) and [HybridNorm](https://github.com/BryceZhuo/HybridNorm).
