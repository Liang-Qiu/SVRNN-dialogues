# SVRNN-Dialogues

This repo contains code for our paper [Structured Attention for Unsupervised Dialogue Structure Induction](https://arxiv.org/pdf/2009.08552.pdf), accepted as a long paper in EMNLP 2020. The codebase is developed on [Unsupervised-Structure-Learning](https://github.com/wyshi/Unsupervised-Structure-Learning) and re-written with PyTorch.

## Dependencies

Dependencies are listed in `requirement.txt`. Install them by running

```bash
make install
```

## Dataset

### SimDial

Simulated dialogs in JSON are generated with [SimDial](https://github.com/snakeztc/SimDial).
Generate samples and interpretion by running

```bash
make simdial
```

### Ubuntu Chat Corpus

First download the Ubuntu Chat Corpus from [here](https://daviduthus.org/UCC/).
Then generate samples from the corpus by running

```bash
make ubuntu
```

### MultiWOZ

Unzip the dataset `data/MultiWOZ_2.1.zip` first, then run

```bash
make mwoz
```

If you want to use GloVe in your experiment, download it [here](https://nlp.stanford.edu/projects/glove/).

## Train  

All configuration is in `params.py`. You should change `use_cuda=True` if you want to use GPU. Try VRNN with Linear CRF Attention on SimDial/MultiWOZ by running

```bash
# SimDial
python train_linear_vrnn.py
```

```bash
# MultiWOZ
python train_multiwoz.py
```

or VRNN with Non-projective Dependency Tree Attention on Ubuntu Chat Corpus

```bash
# Ubuntu Chat Corpus
python train_tree_vrnn.py
```

## Decode

After training, there will be a ckpt_dir under `log/linear_vrnn` or `log/tree_vrnn`, e.g., run1532935232. In the ckpt_dir, there will be saved checkpints in format `*.pt`.

```bash
# SimDial
python train_linear_vrnn.py --decode --ckpt_dir run1532935232 --ckpt_name vrnn_39.pt
# MultiWOZ
python train_multiwoz.py --decode --ckpt_dir run1532935232 --ckpt_name vrnn_39.pt
```

or

```bash
# Ubuntu Chat Corpus
python train_tree_vrnn.py --decode --ckpt_dir run1532935232 --ckpt_name vrnn_39.pt
```

## Citation

If you find the paper and/or the code helpful, please cite

``` bibtex
@inproceedings{qiu-etal-2020-structured,
    title = "Structured Attention for Unsupervised Dialogue Structure Induction",
    author = "Qiu, Liang  and
      Zhao, Yizhou  and
      Shi, Weiyan  and
      Liang, Yuan  and
      Shi, Feng  and
      Yuan, Tao  and
      Yu, Zhou  and
      Zhu, Song-Chun",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.148",
    doi = "10.18653/v1/2020.emnlp-main.148",
    pages = "1889--1899",
    abstract = "Inducing a meaningful structural representation from one or a set of dialogues is a crucial but challenging task in computational linguistics. Advancement made in this area is critical for dialogue system design and discourse analysis. It can also be extended to solve grammatical inference. In this work, we propose to incorporate structured attention layers into a Variational Recurrent Neural Network (VRNN) model with discrete latent states to learn dialogue structure in an unsupervised fashion. Compared to a vanilla VRNN, structured attention enables a model to focus on different parts of the source sentence embeddings while enforcing a structural inductive bias. Experiments show that on two-party dialogue datasets, VRNN with structured attention learns semantic structures that are similar to templates used to generate this dialogue corpus. While on multi-party dialogue datasets, our model learns an interactive structure demonstrating its capability of distinguishing speakers or addresses, automatically disentangling dialogues without explicit human annotation.",
}
```
