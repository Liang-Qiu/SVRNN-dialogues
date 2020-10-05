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
make dataset data_path=path/to/your/ubuntu/corpus
```

If you want to use GloVe in your experiment, download it [here](https://nlp.stanford.edu/projects/glove/).

## Train  

All configuration is in `params.py`. You should change `use_cuda=True` if you want to use GPU. Try VRNN with Linear CRF Attention on SimDial by running

```bash
python train_linear_vrnn.py
```

or VRNN with Non-projective Dependency Tree Attention on Ubuntu Chat Corpus

```bash
python train_tree_vrnn.py
```

## Decode

After training, there will be a ckpt_dir under `log/linear_vrnn` or `log/tree_vrnn`, e.g., run1532935232. In the ckpt_dir, there will be saved checkpints in format `*.pt`.

```bash
python train_linear_vrnn.py --decode --ckpt_dir run1532935232 --ckpt_name vrnn_60.pt
```

or

```bash
python train_tree_vrnn.py --decode --ckpt_dir run1532935232 --ckpt_name vrnn_60.pt
```

## Citation

If you find the paper and/or the code helpful, please cite

``` bibtex
@article{qiu2020structured,
  title={Structured Attention for Unsupervised Dialogue Structure Induction},
  author={Qiu, Liang and Zhao, Yizhou and Shi, Weiyan and Liang, Yuan and Shi, Feng and Yuan, Tao and Yu, Zhou and Zhu, Song-Chun},
  journal={arXiv preprint arXiv:2009.08552},
  year={2020}
}
```
