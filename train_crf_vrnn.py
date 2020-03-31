
from __future__ import print_function

import random
import os
import time
import argparse

import pickle as pkl
import torch
from torch import nn, optim
import numpy as np
from beeprint import pp

from models.vrnn import VRNN
from data_apis.data_utils import SWDADataLoader
from data_apis.SWDADialogCorpus import SWDADialogCorpus
from utils.loss import print_loss
import params


def get_dataset(device):
    with open(params.api_dir, "rb") as fh:
        api = pkl.load(fh, encoding='latin1')
    dial_corpus = api.get_dialog_corpus()

    train_dial, labeled_dial, test_dial = dial_corpus.get(
        "train"), dial_corpus.get("labeled"), dial_corpus.get("test")

    # convert to numeric input outputs
    train_loader = SWDADataLoader("Train",
                                  train_dial,
                                  params.max_utt_len,
                                  params.max_dialog_len,
                                  device=device)
    valid_loader = test_loader = SWDADataLoader("Test",
                                                test_dial,
                                                params.max_utt_len,
                                                params.max_dialog_len,
                                                device=device)
    return train_loader, valid_loader, test_loader, np.array(api.word2vec)


def train(model, train_loader, optimizer):
    elbo_t = []
    rc_loss = []
    kl_loss = []
    bow_loss = []
    local_t = 0
    start_time = time.time()
    loss_names = ["elbo_t", "rc_loss", "kl_loss", "bow_loss"]
    model.train()

    while True:
        optimizer.zero_grad()
        batch = train_loader.next_batch()
        if batch is None:
            break
        local_t += 1
        loss = model(*batch)
        # use .data to free the loss Variable
        elbo_t.append(loss[0].data)
        rc_loss.append(loss[1].data)
        kl_loss.append(loss[2].data)
        bow_loss.append(loss[3].data)
        loss[0].backward(
        )  # loss[0] = elbo_t = rc_loss + weight_kl * kl_loss + weight_bow * bow_loss
        optimizer.step()

        if local_t % (train_loader.num_batch // 10) == 0:
            print_loss("%.2f" %
                       (train_loader.ptr / float(train_loader.num_batch)),
                       loss_names, [elbo_t, rc_loss, kl_loss, bow_loss],
                       postfix='')
    # finish epoch!
    epoch_time = time.time() - start_time
    print_loss("Epoch Done", loss_names, [elbo_t, rc_loss, kl_loss, bow_loss],
               "step time %.4f" % (epoch_time / train_loader.num_batch))


def valid(model, valid_loader):
    elbo_t = []
    model.eval()
    while True:
        batch = valid_loader.next_batch()
        if batch is None:
            break
        loss = model(*batch)
        elbo_t.append(loss[0].data)

    print_loss("Valid", ["elbo_t"], [elbo_t], "")
    return torch.mean(torch.stack(elbo_t))


def decode(model, data_loader):
    results = []
    model.eval()
    while True:
        batch = data_loader.next_batch()
        if batch is None:
            break
        result = model(*batch, training=False)
        results.append(result)
    return results


def main(args):
    pp(params)
    # set random seeds
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, word2vec = get_dataset(device)

    if args.forward_only or args.resume:
        log_dir = os.path.join(params.log_dir, args.ckpt_dir)
        checkpoint_path = os.path.join(log_dir, args.ckpt_name)
    else:
        log_dir = os.path.join(params.log_dir, "run" + str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)

    model = VRNN().to(device)
    if params.op == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=params.init_lr,
                               weight_decay=params.lr_decay)
    elif params.op == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=params.init_lr,
                                  weight_decay=params.lr_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=params.init_lr,
                              weight_decay=params.lr_decay)

    if word2vec is not None and not args.forward_only:
        print("Load word2vec")
        model.embedding.from_pretrained(torch.from_numpy(word2vec),
                                        freeze=False)

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
    patience = params.max_epoch
    dev_loss_threshold = np.inf
    best_dev_loss = np.inf
    if not args.forward_only:
        for epoch in range(last_epoch + 1, params.max_epoch + 1):
            print(">> Epoch %d" % (epoch))
            for param_group in optimizer.param_groups:
                print("Learning rate %f" % param_group['lr'])

            if train_loader.num_batch is None or train_loader.ptr >= train_loader.num_batch:
                train_loader.epoch_init(params.batch_size, shuffle=True)
            train(model, train_loader, optimizer)
            print("Best valid loss so far %f" % best_dev_loss)
            valid_loader.epoch_init(params.batch_size, shuffle=False)
            valid_loss = valid(model, valid_loader)
            if valid_loss < best_dev_loss:
                # increase patience when valid_loss is small enough
                if valid_loss <= dev_loss_threshold * params.improve_threshold:
                    patience = max(patience, epoch * params.patient_increase)
                    dev_loss_threshold = valid_loss

                # still save the best train model
                if args.save_model:
                    print("Update the best valid loss and save the model.")
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(
                        state,
                        os.path.join(log_dir, "vrnn_" + str(epoch) + ".pt"))
                    best_dev_loss = valid_loss

            if params.early_stop and patience <= epoch:
                print("Early stop due to run out of patience!!")
                break
    # Inference only
    else:
        state = torch.load(checkpoint_path)
        print("Load model from %s" % checkpoint_path)
        model.load_state_dict(state['state_dict'])
        if not args.use_test_batch:
            train_loader.epoch_init(params.batch_size, shuffle=False)
            results = decode(model, train_loader)
        else:
            test_loader.epoch_init(params.batch_size, shuffle=False)
            results = decode(
                model, test_loader
            )  # [num_batches(8), 4, batch_size(16), max_dialog_len(10), n_state(10)]
        with open(os.path.join(log_dir, "result.pkl"), "wb") as fh:
            pkl.dump(results, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward_only',
                        default=False,
                        type=bool,
                        help='Whether only do decoding')
    parser.add_argument('--resume',
                        default=False,
                        type=bool,
                        help='Resume training from checkpoint')
    parser.add_argument(
        '--ckpt_dir',
        default='',
        type=str,
        help='The directory to load the checkpoint, e.g. run1585003537')
    parser.add_argument(
        '--ckpt_name',
        default='',
        type=str,
        help='Name of the saved model checkpoint, e.g. vrnn_60.pt')
    parser.add_argument('--save_model',
                        default=True,
                        type=bool,
                        help='whether save checkpoints')
    parser.add_argument(
        '--use_test_batch',
        default=False,
        type=bool,
        help='Whether use test dataset for structure interpretion')

    args = parser.parse_args()

    main(args)
