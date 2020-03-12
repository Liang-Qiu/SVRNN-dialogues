from __future__ import print_function

import random
import os
import time

import pickle as pkl
import torch
from torch import nn, optim
import numpy as np
import nltk

from models.vrnn import VRNN
from data_apis.data_utils import SWDADataLoader
from data_apis.SWDADialogCorpus import SWDADialogCorpus
import params


def get_dataset():
    with open(params.api_dir, "rb") as fh:
        api = pkl.load(fh, encoding='latin1')
    dial_corpus = api.get_dialog_corpus()

    train_dial, labeled_dial, test_dial = dial_corpus.get(
        "train"), dial_corpus.get("labeled"), dial_corpus.get("test")

    # convert to numeric input outputs
    train_feed = SWDADataLoader("Train", train_dial, params.max_utt_len,
                                params.max_dialog_len)
    valid_feed = test_feed = SWDADataLoader("Test", test_dial,
                                            params.max_utt_len,
                                            params.max_dialog_len)
    return train_feed, valid_feed, test_feed


def train(model, train_loader, optimizer):
    model.train()
    while True:
        optimizer.zero_grad()
        batch = train_loader.next_batch()
        if batch is None:
            break
        loss = model(*batch)
        loss.backward()
        optimizer.step()
        print(loss)


def main():
    # set random seeds
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # set device
    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader = get_dataset()

    if params.forward_only or params.resume:
        log_dir = os.path.join(params.work_dir, params.test_path)
    else:
        log_dir = os.path.join(params.work_dir, "run" + str(int(time.time())))

    model = VRNN()

    optimizer = optim.Adam(model.parameters(), lr=params.init_lr)
    if train_loader.num_batch is None or train_loader.ptr >= train_loader.num_batch:
        train_loader.epoch_init(params.batch_size, shuffle=True)
    for epoch in range(1, params.max_epoch + 1):
        train(model, train_loader, optimizer)


if __name__ == "__main__":
    main()
