import sys
import argparse
from collections import Counter
import os

import pickle as pkl
import numpy as np
import copy
import torch
from beeprint import pp

import params
from models.vrnn import VRNN
from data_apis.SWDADialogCorpus import SWDADialogCorpus


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def id_to_sent(id_to_vocab, ids):
    sent = []
    for id in ids:
        if id:
            if id_to_vocab[id] != '<s>' and id_to_vocab[id] != '</s>':
                sent.append(id_to_vocab[id])
        else:
            break
    return " ".join(sent)


def id_to_probs(probs, ids, id_to_vocab, SOFTMAX=False):
    if SOFTMAX:
        probs = softmax(probs)
    else:
        pass

    product = 1
    for id in ids:
        if id_to_vocab[id] == '</s>':
            break
        elif id_to_vocab[id] == '<s>':
            pass
        elif id:
            product *= probs[id]
        else:
            print("")
            raise Exception("id is empty!")
    return product


def id_to_log_probs(probs, ids, id_to_vocab, SOFTMAX=False):
    if SOFTMAX:
        probs = softmax(probs)
    else:
        pass

    sum = 0
    for id in ids:
        if id_to_vocab[id] == '</s>':
            break
        elif id_to_vocab[id] == '<s>':
            pass
        elif id:
            sum += np.log(probs[id])
        else:
            print("")
            raise Exception("id is empty!")
    return sum


def get_state_sents(state,
                    converted_sents,
                    converted_labels,
                    last_n=3,
                    sys_side=1):
    state_sents = []
    for i in range(len(converted_sents)):
        for j, label in enumerate(converted_labels[i]):
            if label == state:
                if converted_sents[i][j][sys_side]:
                    last_n_sents = [
                        converted_sents[i][j - i_last_n][sys_side]
                        for i_last_n in range(last_n) if (j - i_last_n) >= 0
                    ]
                    last_n_sents = last_n_sents[::-1]
                    last_n_sents = "\n ".join(last_n_sents)

                    state_sents.append(last_n_sents)
    return state_sents


def main(args):
    with open(params.api_dir, "rb") as fh:
        api2 = pkl.load(fh, encoding='latin1')

    with open(os.path.join(args.log_dir, "result.pkl"), "rb") as fh:
        results = pkl.load(fh)

    state = torch.load(os.path.join(args.log_dir, "vrnn_2.pt"))
    # pp(state['state_dict'])

    converted_labels = []
    converted_sents = []
    conv_probs = []
    for batch_i in range(len(results)):
        usr_sents = results[batch_i][0]
        sys_sents = results[batch_i][1]
        probs = results[batch_i][2]
        trans_probs = results[batch_i][3]
        bow_logits1 = results[batch_i][4]
        bow_logits2 = results[batch_i][5]
        for i in range(params.batch_size):
            this_dialog_labels = []
            this_dialog_sents = []
            prev_label = -1
            this_conv_prob = 1
            for turn_j in range(params.max_dialog_len):
                if not usr_sents[i, turn_j, 0]:
                    break
                label = probs[i, turn_j].argmax()
                usr_tokens = id_to_sent(api2.id_to_vocab, usr_sents[i, turn_j])
                sys_tokens = id_to_sent(api2.id_to_vocab, sys_sents[i, turn_j])
                usr_prob = id_to_log_probs(bow_logits1[i, turn_j],
                                           usr_sents[i, turn_j],
                                           api2.id_to_vocab,
                                           SOFTMAX=True)
                sys_prob = id_to_log_probs(bow_logits2[i, turn_j],
                                           sys_sents[i, turn_j],
                                           api2.id_to_vocab,
                                           SOFTMAX=True)

                this_dialog_labels += [label]
                this_dialog_sents += [[usr_tokens, sys_tokens]]
                this_turn_prob = usr_prob + sys_prob
                this_conv_prob += this_turn_prob
            conv_probs.append(this_conv_prob)
            converted_labels.append(this_dialog_labels)
            converted_sents.append(this_dialog_sents)

    sents_by_state = []
    for i in range(params.n_state):
        sents_by_state.append(
            get_state_sents(i,
                            converted_sents,
                            converted_labels,
                            sys_side=0,
                            last_n=1))
    sents_by_state_sys = []
    for i in range(params.n_state):
        sents_by_state_sys.append(
            get_state_sents(i,
                            converted_sents,
                            converted_labels,
                            sys_side=1,
                            last_n=1))
    WITH_START = True
    if WITH_START:
        sents_by_state = [['START']] + sents_by_state
        sents_by_state_sys = [['START']] + sents_by_state_sys

    transition_count = np.zeros((params.n_state, params.n_state))

    # for labels in converted_labels:
    #     # origin = 0
    #     for i in xrange(len(labels) - 1):
    #         #dest = l
    #         print(i)
    #         transition_count[labels[i], labels[i + 1]] += 1
    #         #origin = dest
    #     #transition_prob[origin, 11] += 1

    # transition_prob = np.eye((params.n_state, params.n_state))
    # for i in xrange(params.n_state):
    #     transition_prob[i] = transition_count[i] / transition_count[i].sum()

    # # direct transition only, for direct transition, the transition probs are from the fetch_results from the model
    # DIRECT_TRANSITION = False
    # if DIRECT_TRANSITION:
    #     label_i_list = np.eye(params.n_state, params.n_state)
    #     for i in range(params.n_state):
    #         label_i_list[i][i] = 1
    #         print(label_i_list)
    #     label_i_list = np.vstack([[1] * params.n_state, label_i_list])

    #     prob_list = []
    #     for i in range(0, params.n_state + 1):
    #         tmp_prob = np.matmul(
    #             np.matmul(
    #                 np.matmul(label_i_list[i], fetch_results[0]) +
    #                 fetch_results[1], fetch_results[2]) + fetch_results[3],
    #             fetch_results[4]) + fetch_results[5]
    #         prob_list.append(softmax(tmp_prob))

    #     transition_prob = prob_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir',
                        default='log/run1584581812',
                        type=str,
                        help='Path to the saved result')

    args = parser.parse_args()

    main(args)
