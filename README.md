# UAM-Net (Pytorch)

## Requirements

* pytorch == 1.3
* torchvision == 0.4.2
* yacs == 0.1.8

## Usage

**To train the model :**

```
python uam_train.py
```

##Cite this project
If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entry.
```
@article{ZHANG2022108594,
title = {Unabridged Adjacent Modulation for Clothing Parsing},
journal = {Pattern Recognition},
pages = {108594},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108594},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322000759},
author = {Dong Zhang and Chengting Zuo and Qianhao Wu and Liyong Fu and Xinguang Xiang},
keywords = {Encoder-decoder network, Clothing parsing, Attention learning, Features modulation, Self-supervised learning},
abstract = {Clothing parsing has made tremendous progress in the domain of computer vision recently. Most state-of-the-art methods are based on the encoder-decoder architecture. However, the existing methods mainly neglect problems of feature uncalibration within blocks and semantics dilution between blocks. In this work, we propose an unabridged adjacent modulation network (UAM-Net) to aggregate multi-level features for clothing parsing. We first build an unabridged channel attention (UCA) mechanism on feature maps within each block for feature recalibration. We further design a top-down adjacent modulation (TAM) for decoder blocks. By deploying TAM, high-level semantic information and visual contexts can be gradually transferred into lower-level layers without loss. The joint implementation of UCA and TAM ensures that the encoder has an enhanced feature representation ability, and the low-level features of the decoders contain abundant semantic contexts. Quantitative and qualitative experimental results on two challenging benchmarks (i.e., colorful fashion parsing and the modified fashion clothing) declare that our proposed UAM-Net can achieve competitive high-accurate performance with the state-of-the-art methods. The source codes are available at: https://github.com/ctzuo/UAM-Net.}
}
```
