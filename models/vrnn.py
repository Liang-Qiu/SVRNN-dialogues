from __future__ import print_function
from __future__ import division

import os
import re
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append("..")
import params
from .vae_cell import VAECell


class VRNN(nn.Module):
    """
    VRNN with gumbel-softmax
    """
    def __init__(self):
        super(VRNN, self).__init__()

        # neural layers
        self.embedding = nn.Embedding(params.max_vocab_cnt, params.embed_size)

        if params.cell_type == "gru":
            self.sent_rnn = nn.GRU(params.embed_size,
                                   params.encoding_cell_size,
                                   params.num_layer,
                                   batch_first=True)
        else:
            self.sent_rnn = nn.LSTM(params.embed_size,
                                    params.encoding_cell_size,
                                    params.num_layer,
                                    batch_first=True)

        self.vae_cell = VAECell(state_is_tuple=True)

    def forward(self, usr_input_sent, sys_input_sent, dialog_length_mask,
                usr_input_mask, sys_input_mask):
        ########################## sent_embedding  ######################
        usr_input_embedding = self.embedding(
            usr_input_sent)  # (16, 10, 40, 300)
        usr_input_embedding = usr_input_embedding.view(
            [-1, params.max_utt_len, params.embed_size])  # (160, 40, 300)

        sys_input_embedding = self.embedding(
            sys_input_sent)  # (16, 10, 40, 300)
        sys_input_embedding = sys_input_embedding.view(
            [-1, params.max_utt_len, params.embed_size])  # (160, 40, 300)

        # TODO: dynamic RNN
        if params.cell_type == "gru":
            _, usr_sent_embedding = self.sent_rnn(usr_input_embedding)
            _, sys_sent_embedding = self.sent_rnn(sys_input_embedding)
        else:
            _, (_, usr_sent_embedding) = self.sent_rnn(usr_input_embedding)
            _, (_, sys_sent_embedding) = self.sent_rnn(sys_input_embedding)
        usr_sent_embedding = usr_sent_embedding.view(
            -1, params.max_dialog_len,
            params.encoding_cell_size)  # (16, 10, 400)
        sys_sent_embedding = sys_sent_embedding.view(
            -1, params.max_dialog_len,
            params.encoding_cell_size)  # (16, 10, 400)

        if params.dropout > 0:
            usr_sent_embedding = F.dropout(usr_sent_embedding,
                                           p=params.dropout)
            sys_sent_embedding = F.dropout(sys_sent_embedding,
                                           p=params.dropout)

        joint_embedding = torch.cat(
            [usr_sent_embedding, sys_sent_embedding],
            dim=2)  # (batch, dialog_len, encoding_cell_size * 2) (16, 10, 800)

        ########################### state level ############################
        dec_input_embedding_usr = self.embedding(
            usr_input_sent)  # (16, 10, 40, 300)
        dec_input_embedding_sys = self.embedding(
            sys_input_sent)  # (16, 10, 40, 300)
        dec_input_embedding = [
            dec_input_embedding_usr, dec_input_embedding_sys
        ]

        dec_seq_lens_usr = torch.sum(torch.sign(usr_input_mask),
                                     dim=2)  # (16, 10)
        dec_seq_lens_sys = torch.sum(torch.sign(sys_input_mask),
                                     dim=2)  # (16, 10)
        dec_seq_lens = [dec_seq_lens_usr, dec_seq_lens_sys]

        output_tokens = [usr_input_sent, sys_input_sent]

        prev_z = torch.ones(params.batch_size, params.n_state)

        losses = []
        for utt in range(params.max_dialog_len):
            inputs = joint_embedding[:, utt, :]
            if params.cell_type == "gru":
                state = torch.zeros(params.batch_size, params.state_cell_size)
            else:
                c = h = torch.zeros(params.batch_size, params.state_cell_size)
                state = (c, h)
            dec_input_emb = [
                dec_input_embedding[0][:, utt, :, :],
                dec_input_embedding[1][:, utt, :, :]
            ]
            dec_seq_len = [dec_seq_lens[0][:, utt], dec_seq_lens[1][:, utt]]
            output_token = [
                output_tokens[0][:, utt, :], output_tokens[1][:, utt, :]
            ]

            elbo_t, z_samples, state, p_z, bow_logits1, bow_logits2 = self.vae_cell(
                inputs,
                state,
                dec_input_emb,
                dec_seq_len,
                output_token,
                prev_z_t=prev_z)

            max_idx = torch.argmax(z_samples, 1, keepdim=True)
            zts_onehot = torch.FloatTensor(z_samples.shape)
            zts_onehot.zero_()
            zts_onehot.scatter_(1, max_idx, 1)
            # TODO: stop gradient?
            prev_z = zts_onehot
            losses.append(elbo_t)
        loss = torch.cat(losses, dim=0)
        loss_avg = torch.sum(loss) / (torch.sum(usr_input_mask) +
                                      torch.sum(sys_input_mask))

        return loss_avg
