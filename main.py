from __future__ import print_function

import random
import os
import time
import argparse

import pickle as pkl
import torch
from torch import nn, optim
import numpy as np
import nltk
from beeprint import pp

from models.vrnn import VRNN
from data_apis.data_utils import SWDADataLoader
from data_apis.SWDADialogCorpus import SWDADialogCorpus
from utils.loss import print_loss
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
    return train_feed, valid_feed, test_feed, np.array(api.word2vec)


def train(model, train_loader, optimizer):
    losses = []
    local_t = 0
    start_time = time.time()
    loss_names = ["loss"]
    model.train()

    while True:
        optimizer.zero_grad()
        batch = train_loader.next_batch()
        if batch is None:
            break
        local_t += 1
        loss = model(*batch)
        losses.append(loss)
        loss.backward()
        optimizer.step()

        if local_t % (train_loader.num_batch // 10) == 0:
            print_loss("%.2f" %
                       (train_loader.ptr / float(train_loader.num_batch)),
                       loss_names, [losses],
                       postfix='')
    # finish epoch!
    epoch_time = time.time() - start_time
    print_loss("Epoch Done", loss_names, [losses],
               "step time %.4f" % (epoch_time / train_loader.num_batch))


def valid(model, valid_loader):
    losses = []
    while True:
        batch = valid_loader.next_batch()
        if batch is None:
            break
        loss = model(*batch)
        losses.append(loss)

    print_loss("ELBO_VALID", ['losses valid'], [losses], "")


def decode(model, data_loader):
    results = []
    while True:
        batch = data_loader.next_batch()
        if batch is None:
            break
        result = model(*batch, interpret=True)
        results.append(result)
    return results


def main(args):
    pp(params)
    # set random seeds
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # set device
    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, word2vec = get_dataset()

    if args.forward_only or args.resume:
        log_dir = os.path.join(params.log_dir, args.test_path)
        checkpoint_path = os.path.join(log_dir, args.checkpoint_path)
    else:
        log_dir = os.path.join(params.log_dir, "run" + str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)

    model = VRNN()
    optimizer = optim.Adam(model.parameters(), lr=params.init_lr)

    if word2vec is not None and not args.forward_only:
        print("Load word2vec")
        model.embedding.from_pretrained(torch.from_numpy(word2vec))

    # Write config to a file for logging
    if not args.forward_only:
        with open(os.path.join(log_dir, "run.log"), "w") as f:
            f.write(pp(params, output=False))

    last_epoch = 0
    if args.resume:
        print("Resuming training from %s" % checkpoint_path)
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        last_epoch = state['epoch']

    # Train and evaluate
    if not args.forward_only:
        for epoch in range(last_epoch + 1, params.max_epoch + 1):
            print(">> Epoch %d with lr %f" % (epoch, params.init_lr))
            if train_loader.num_batch is None or train_loader.ptr >= train_loader.num_batch:
                train_loader.epoch_init(params.batch_size, shuffle=True)
            train(model, train_loader, optimizer)
            valid_loader.epoch_init(params.batch_size, shuffle=False)
            valid(model, valid_loader)

            if args.save_model:
                print("Save the model at the end of each epoch.")
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state,
                           os.path.join(log_dir, "vrnn_" + str(epoch) + ".pt"))
    # Inference only
    else:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        if not args.use_test_batch:
            train_loader.epoch_init(params.batch_size, shuffle=False)
            results = decode(model, train_loader)
        else:
            valid_loader.epoch_init(params.batch_size, shuffle=False)
            results = decode(
                model, valid_loader
            )  # [num_batches(8), 4, max_dialog_len(10), batch_size(16), n_state(10)]
            # TODO: exchange dim 2 and dim3
        with open(os.path.join(log_dir, "result.pkl"), "wb") as fh:
            pkl.dump(results, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward_only',
                        default=False,
                        type=bool,
                        help='Only do decoding')
    parser.add_argument('--resume',
                        default=False,
                        type=bool,
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path',
                        default='',
                        type=str,
                        help='Name of the saved model checkpoint')
    parser.add_argument('--test_path',
                        default='',
                        type=str,
                        help='The dir to load checkpoint for forward only')
    parser.add_argument('--save_model',
                        default=True,
                        type=bool,
                        help='Create checkpoints')
    parser.add_argument(
        '--use_test_batch',
        default=True,
        type=bool,
        help='Whether or not use test dataset for structure interpretion')

    args = parser.parse_args()

    main(args)
