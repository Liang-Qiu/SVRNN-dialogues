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
            for turn_j in range(probs.shape[1]):
                label = probs[i, turn_j].argmax()
                usr_tokens = id_to_sent(api2.id_to_vocab, usr_sents[i, turn_j])
                sys_tokens = id_to_sent(api2.id_to_vocab, sys_sents[i, turn_j])
                usr_prob = id_to_log_probs(bow_logits1[i, turn_j], usr_sents[i, turn_j], api2.id_to_vocab, SOFTMAX=True)
                sys_prob = id_to_log_probs(bow_logits2[i, turn_j], sys_sents[i, turn_j], api2.id_to_vocab, SOFTMAX=True)
                
                this_dialog_labels += [label]
                this_dialog_sents += [[usr_tokens, sys_tokens]]
                this_turn_prob = usr_prob + sys_prob
                print(this_turn_prob)
                this_conv_prob += this_turn_prob
        conv_probs.append(this_conv_prob)
        converted_labels.append(this_dialog_labels)
        converted_sents.append(this_dialog_sents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir',
                        default='log/run1584581812',
                        type=str,
                        help='Path to the saved result')

    args = parser.parse_args()

    main(args)
