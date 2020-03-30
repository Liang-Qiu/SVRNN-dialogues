from __future__ import print_function

import random
import torch
import torch.optim as optim
import numpy as np
from beeprint import pp

from data_apis.vocab import Vocab
from data_apis.UbuntuChatCorpus import Batcher
import params


def get_dataset(device):
    # vocabulary
    vocab = Vocab(params.vocab_path, params.max_vocab_cnt, params.use_glove,
                  params.glove_path)

    train_loader = Batcher(params.data_path,
                           vocab,
                           mode="train",
                           device=device)
    valid_loader = Batcher(params.eval_data_path,
                           vocab,
                           mode="eval",
                           device=device)
    test_loader = Batcher(params.test_data_path,
                          vocab,
                          mode="decode",
                          device=device)

    return train_loader, valid_loader, test_loader, vocab


def main():
    pp(params)
    # set random seeds
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # set device
    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, vocab = get_dataset(device)

    x = train_loader._next_batch()

    print("test")


if __name__ == "__main__":
    main()