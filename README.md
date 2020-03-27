# TSAN-Dialogues

All configuration is in `params.py`. You should change `use_cuda=True` if you want to use GPU.

## Install

```bash
make install
```

## Train  

```bash
make train
```

## Decode

```bash
python main.py --forward_only True --ckpt_dir run1585003537 --ckpt_name vrnn_5.pt
```

## Interpret

```bash
python interpretion.py --ckpt_dir run1585003537 --ckpt_name vrnn_5.pt
```
