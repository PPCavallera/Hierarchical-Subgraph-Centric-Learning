# [Vision Paper] Hierarchical Subgraph-Centric Learning: A Vision for Scalable and Explainable Temporal Graph Machine Learning


## Abstract

Graph-structured data are increasingly central to domains such as social networks, transportation, and epidemiology. While temporal graph neural networks (TGNNs) effectively capture evolving node or edge dynamics, they remain large, complex black-box models that are difficult to scale, analyze, and reason about.  We argue that the main limitation in temporal graph learning lies not in the choice of TGNN models, but in how graphs are represented and structured for learning. Real-world temporal graphs exhibit inherent multi-scale and hierarchical organization, yet existing approaches largely ignore this structure in favor of global end-to-end learning. In this vision paper, we introduce  Hierarchical Subgraph-Centric Learning (HSCL),  a learning paradigm that redefines subgraphs as first-class units of representation, learning, and explanation. HSCL builds on a hybrid hierarchical clustering process that integrates structural, attribute, and temporal similarity, capabilities largely absent from existing TGNN frameworks. This subgraph-centric perspective supports modular subgraph-level models that can be composed across scales and  enables multi-level explanation, from global system behavior to subgraph dynamics and individual nodes or edges. We position HSCL as a forward-looking framework to guide future research at the intersection of temporal graph learning, hierarchical clustering, and explanability . Preliminary experiments on four public benchmarks illustrate the promising potential of HSCL.

## Datasets

DATASETS AVAIBLE AT : https://drive.google.com/drive/folders/1qvfHgZf4vDhBtwi1potBh0ZAXET-9Mvz?usp=drive_link

| Dataset Name | Number of nodes | Number of timestamps |
| -------- | :--------: | :--------: |
| ChickenPox  Hungary | 20 | 522 |
| Wikipedia Math | 1068 | 731 |
| Windmill | 319 | 17472 |
| METR-LA | 207 | 34272 |

## Models

**LSTM** : Simple auto-encoder based on LSTM layers

**DCRNN**[1] : TGNN implementation from [2]

## Experimental setup 

Experiments had been run using :

- **CPU** : Intel Xeon Gold 6230R (Cascade Lake-SP), x86_64, 2.10GHz, 2 CPUs/node, 26 cores/CPU
- **GPU** : Nvidia Quadro RTX 8000 (45 GiB)
- **RAM** : 384 GiB


## How to install

Install packages from requirement.txt  

```
pip install -r requirements.txt
```


[1]: Li, Y et al. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*

[2]: Rozemberczki, B et al. (2021). *PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models*

