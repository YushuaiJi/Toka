# Toka

## 1 Introduction

**Toka** is a **T**PU-**o**ptimized **𝑘**-means **a**lgorithm. This repo holds the source code and scripts for reproducing the key experiments of our paper:  <u>Boosting Incremental Balanced Clustering via TPU-Optimized 𝑘-means Algorithm</u>

## 2 Data Process

- Upload the notebook to `Colab` or any other platforms which have `CPU`, `GPU`, and `TPU` processor.
- Put all the datasets into `./content/sample_data/`.
- Then you can run the notebooks to get the results.

The datasets we use are all high-dimensional, with brief information as shown in the list below:

|                           Datasets                           | Dataset Scale | Dimension |                         Description                          |
| :----------------------------------------------------------: | :-----------: | :-------: | :----------------------------------------------------------: |
| **[Huatuo](https://github.com/FreedomIntelligence/Huatuo-26M)** |     26M     |      1024     | A large-scale Chinese medical text embedding dataset designed for healthcare-related NLP tasks. |
| **[LiveChat](https://github.com/gaojingsheng/LiveChat)** |     1.39M     |     1024    | A conversational dataset containing chat logs or embeddings from real-time customer support interactions. |
|      **[Deep](https://www.tensorflow.org/datasets/catalog/deep1b)**      |    1B     |     96      | A large-scale dataset of deep learning-based feature vectors designed for billion-scale approximate nearest neighbor search. |
| **[GloVe](https://nlp.stanford.edu/projects/glove)** |    5.32M       |    300       | A pre-trained word embedding dataset capturing semantic relationships between words using vector representations. |
| **[SIFT](https://archive.ics.uci.edu/dataset/353/sift10m)** |    10M     |     128        | A dataset of image patches used for extracting SIFT descriptors for visual feature matching and retrieval. |

## 3 Comparison Algorithms

### CPU based method

1. [Least squares quantization in PCM](https://hal.science/hal-04614938/document)
1. [Coordinate Descent Method for k-means](https://ieeexplore.ieee.org/abstract/document/9444882/)
1. [Balanced clustering with least square regression](https://ojs.aaai.org/index.php/AAAI/article/view/10877)
1. [Fast clustering with flexible balance constraints](https://ieeexplore.ieee.org/abstract/document/8621917/)
1. [Balanced k-means with a Novel Constraint](https://www.sciencedirect.com/science/article/pii/S0165168422001141)

### GPU based method

1. [GPU-based k-Means algorithm](https://www.sciencedirect.com/science/article/pii/S0022000012000992)
1. [Accelerating the yinyang k-means algorithm using the GPU](https://ieeexplore.ieee.org/abstract/document/9458604)

### TPU based method

1. [Enhancing k-Means Algorithm with Tensor Processing Unit](https://ieeexplore.ieee.org/abstract/document/10020427)

### Incremental strategy

1. [Web-scale k-means clustering](https://dl.acm.org/doi/abs/10.1145/1772690.1772862)
1. [Fully Dynamic k-Means Coreset in Near-Optimal Update Time](https://arxiv.org/abs/2406.19926)
1. [Novel partitional color quantization algorithm](https://www.sciencedirect.com/science/article/pii/S0957417422011708)

## 4 How to Run Toka

You can run `Toka` and all the comparison algorithms in `jupyter notebook`

**Parameter configuration:**

```matlab
% 0.test time
n_runs = 10;

% 1.comparison algorithms
models = {'Lloyd', 'CDKM', 'BCLS', 'FCFC', 'BKNC', 'ASBK-means', 'GPU-Yinyang', 'GPU-CDKM', 'GPU-BCLS', 'GPU-FCFC', 'GPU-BKNC', 'DR-means', 'Toka'}

% 2.datasets
datasets = {'rename this term into your datasets'};

% 3.the iter value size
iters = [1, 10, 20, 30, 40, 50]

% 4.the k value size
k_values = [5, 10, 20, 50, 100]

% 5.different lambda/eta value
start = 0.4
step = 0.0005
end = 1
lambda_values = [start + i * step for i in range(int((end - start) / step) + 1)]
 
% 6.group size
group_sizes = [1, 2, 8, 32, 64]
```
