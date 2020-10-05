from __future__ import print_function

import random
import time
import argparse
import os

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

    train_loader = Batcher(params.data_path, vocab, mode="train", device=device)
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

    if args.forward_only or args.resume:
        log_dir = os.path.join(params.log_dir, "tree_vrnn", args.ckpt_dir)
        checkpoint_path = os.path.join(log_dir, "tree_vrnn", args.ckpt_name)
    else:
        log_dir = os.path.join(params.log_dir, "tree_vrnn",
                               "run" + str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)

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

    # write config to a file for logging
    if not args.forward_only:
        with open(os.path.join(log_dir, "run.log"), "w") as f:
            f.write(pp(params, output=False))

    last_step = 0
    if args.resume:
        print("Resuming training from %s" % checkpoint_path)
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        last_step = state['step']

    elbo_t = []
    rc_loss = []
    kl_loss = []
    bow_loss = []
    loss_names = ["elbo_t", "rc_loss", "kl_loss", "bow_loss"]
    patience = params.n_training_steps
    dev_loss_threshold = np.inf
    best_dev_loss = np.inf
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
            print("Best valid loss so far %f" % best_dev_loss)
            model.eval()
            valid_loss = valid(model, valid_loader)
            if valid_loss < best_dev_loss:
                print("Get a smaller valid loss, update the best valid loss")
                best_dev_loss = valid_loss
                # increase patience when valid_loss is small enough
                if valid_loss <= dev_loss_threshold * params.improve_threshold:
                    patience = max(patience, step * params.patient_increase)
                    dev_loss_threshold = valid_loss

                # still save the best train model
                if args.save_model:
                    print("Saving the model.")
                    state = {
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(
                        state, os.path.join(log_dir,
                                            "vrnn_" + str(step) + ".pt"))

            if params.early_stop and patience <= step:
                print("Early stop due to run out of patience!!")
                break

        if step == params.n_training_steps:
            print_loss("Training Done", loss_names,
                       [elbo_t, rc_loss, kl_loss, bow_loss], "")
    training_time = time.time() - start_time
    print("step time %.4f" % (training_time / params.n_training_steps))


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
    main()
