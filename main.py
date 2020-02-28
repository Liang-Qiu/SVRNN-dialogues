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
from data.data_utils import SWDADataLoader
from data.SWDADialogCorpus import SWDADialogCorpus
import params


def get_dataset():
    with open(params.api_dir, "rb") as fh:
        api = pkl.load(fh, encoding='latin1')
    dial_corpus = api.get_dialog_corpus()
    if params.with_label_loss:
        labeled_dial_labels = api.get_state_corpus(
            params.max_dialog_len)['labeled']

    train_dial, labeled_dial, test_dial = dial_corpus.get(
        "train"), dial_corpus.get("labeled"), dial_corpus.get("test")

    # convert to numeric input outputs
    train_feed = SWDADataLoader("Train", train_dial, params.max_utt_len,
                                params.max_dialog_len)
    test_feed = SWDADataLoader("Test", test_dial, params.max_utt_len,
                               params.max_dialog_len)
    if params.with_label_loss:
        labeled_feed = SWDADataLoader("Labeled",
                                      labeled_dial,
                                      params.max_utt_len,
                                      params.max_dialog_len,
                                      labeled=True)
    valid_feed = test_feed

    if params.forward_only or params.resume:
        log_dir = os.path.join(params.work_dir, params.test_path)
    else:
        log_dir = os.path.join(params.work_dir, "run" + str(int(time.time())))


# def train(model, train_loader, optimizer):
#     model.train()
#     loss = 0

#     for batch_idx, (data, _) in enumerate(train_loader):
#         optimizer.zero_grad()
#         _, _ = model(data)
#         loss = loss_function()
#         loss.backward()

#         train_loss += loss.data[0]
#         optimizer.step()

#         if batch_idx % 100 == 1:
#             temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx),
#                               temp_min)

#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(train_loader.dataset)))

# def eval():
#     pass


def main():
    # set random seeds
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # set device
    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    get_dataset()

    # n_vocab = 2000
    # model = VRNN(n_vocab)
    # optimizer = optim.Adam()
    # for epoch in range(1, params.epochs + 1):
    #     train(model, train_loader, optimizer)


if __name__ == "__main__":
    main()
