# Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
from __future__ import division

import numpy as np


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, data, max_utt_len, max_dialog_len, labeled=False):
        # assert len(data) == len(meta_data)
        self.name = name
        self.data = data
        # self.meta_data = meta_data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = max_utt_len
        self.max_dialog_size = max_dialog_len
        self.labeled = labeled
        print("Max dialog len %d and min dialog len %d and avg len %f" %
              (np.max(all_lens), np.min(all_lens), float(np.mean(all_lens))))
        # self.indexes = list(np.argsort(all_lens))
        self.indexes = list(range(self.data_size))
        np.random.shuffle(self.indexes)

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:(self.max_utt_size -
                             1)] + [tokens[-1]], [1] * self.max_utt_size
        elif do_pad:
            return tokens + [0] * (self.max_utt_size - len(tokens)), [1] * len(
                tokens) + [0] * (self.max_utt_size - len(tokens))
        else:
            return tokens

    def pad_dialog(self, dialog):
        dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask = [], [], [], []
        if len(dialog) >= self.max_dialog_size:
            for turn in dialog[:self.max_dialog_size]:
                usr_input, usr_mask = self.pad_to(turn[0])
                sys_input, sys_mask = self.pad_to(turn[1])
                dialog_usr_input.append(usr_input)
                dialog_sys_input.append(sys_input)
                dialog_usr_mask.append(usr_mask)
                dialog_sys_mask.append(sys_mask)
        else:
            all_pad_input, all_pad_mask = self.pad_to([])
            for turn in dialog:
                usr_input, usr_mask = self.pad_to(turn[0])
                sys_input, sys_mask = self.pad_to(turn[1])
                dialog_usr_input.append(usr_input)
                dialog_sys_input.append(sys_input)
                dialog_usr_mask.append(usr_mask)
                dialog_sys_mask.append(sys_mask)
            for _ in range(self.max_dialog_size - len(dialog)):
                dialog_usr_input.append(all_pad_input)
                dialog_sys_input.append(all_pad_input)
                dialog_usr_mask.append(all_pad_mask)
                dialog_sys_mask.append(all_pad_mask)
        assert len(dialog_usr_input) == len(dialog_sys_input) == len(
            dialog_usr_mask) == len(dialog_sys_mask) == self.max_dialog_size
        return dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask

    def _prepare_batch(self, cur_index_list):
        # the batch index, the starting point and end point for segment
        # need usr_input_sent, sys_input_sent, dialog_len_mask, usr_full_mask, sys_full_mask = batch

        dialogs = [self.data[idx] for idx in cur_index_list]
        dialog_lens = [self.data_lens[idx] for idx in cur_index_list]

        usr_input_sent, sys_input_sent, usr_full_mask, sys_full_mask = [], [], [], []
        for dialog in dialogs:
            dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask = self.pad_dialog(
                dialog)
            usr_input_sent.append(dialog_usr_input)
            sys_input_sent.append(dialog_sys_input)
            usr_full_mask.append(dialog_usr_mask)
            sys_full_mask.append(dialog_sys_mask)

        # initial_prev_zt = np.ones()

        return np.array(usr_input_sent), np.array(sys_input_sent), np.array(dialog_lens), \
               np.array(usr_full_mask), np.array(sys_full_mask)
