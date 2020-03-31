from __future__ import print_function

import random
import time

import torch
import torch.optim as optim
import numpy as np
from beeprint import pp

from data_apis.vocab import Vocab
from data_apis.UbuntuChatCorpus import Batcher
from models.tree_vrnn import TreeVRNN
import params
from utils.loss import print_loss


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


def train(model, train_loader, optimizer, step):
    optimizer.zero_grad()
    batch = train_loader._next_batch()
    if batch is None:
        return
    loss = model(batch.enc_batch,
                 batch.enc_lens,
                 batch.dec_batch,
                 batch.target_batch,
                 batch.padding_mask,
                 batch.tgt_index,
                 training=True)

    loss[0].backward(
    )  # loss[0] = elbo_t = rc_loss + weight_kl * kl_loss + weight_bow * bow_loss
    optimizer.step()

    # use .data to free the loss Variable
    return loss[0].data, loss[1].data, loss[2].data, loss[3].data


def valid(model, valid_loader):
    elbo_t = []
    while True:
        batch = valid_loader._next_batch()
        if batch is None:
            break
        loss = model(batch.enc_batch,
                     batch.enc_lens,
                     batch.dec_batch,
                     batch.target_batch,
                     batch.padding_mask,
                     batch.tgt_index,
                     training=True)
        elbo_t.append(loss[0].data)

    print_loss("Valid", ["elbo_t"], [elbo_t], "")
    return torch.mean(torch.stack(elbo_t))


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

    model = TreeVRNN().to(device)
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

    last_step = 0
    elbo_t = []
    rc_loss = []
    kl_loss = []
    bow_loss = []
    loss_names = ["elbo_t", "rc_loss", "kl_loss", "bow_loss"]
    for step in range(last_step + 1, params.n_training_steps + 1):
        start_time = time.time()
        model.train()
        losses = train(model, train_loader, optimizer, step)
        elbo_t.append(losses[0])
        rc_loss.append(losses[1])
        kl_loss.append(losses[2])
        bow_loss.append(losses[3])

        if step % params.print_after == 0:
            for param_group in optimizer.param_groups:
                print("Learning rate %f" % param_group['lr'])
            print_loss("%.2f" % (step / float(params.n_training_steps)),
                       loss_names, [elbo_t, rc_loss, kl_loss, bow_loss],
                       postfix='')
            # valid
            model.eval()
            valid_loss = valid(model, valid_loader)

        if step == params.n_training_steps:
            print_loss("Training Done", loss_names,
                       [elbo_t, rc_loss, kl_loss, bow_loss], "")
    training_time = time.time() - start_time
    print("step time %.4f" % (training_time / params.n_training_steps))


if __name__ == "__main__":
    main()