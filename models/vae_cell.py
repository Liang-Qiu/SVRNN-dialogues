"""Torch version of https://github.com/wyshi/Unsupervised-Structure-Learning
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequential import MLP
sys.path.append("..")
from utils.sample import gumbel_softmax
import params


class VAECell(object):
    def __init__(self,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True):

        self._state_is_tuple = state_is_tuple
        self.num_zt = params.n_state
        # temperature of gumbel_softmax
        self.tau = torch.tensor([5.0], requires_grad=True)
        self.vocab_size = params.max_vocab_cnt
        self.max_utt_len = params.max_utt_len
        if params.word_weights:
            self.weights = torch.tensor(params.word_weights,
                                        requires_grad=False)
        else:
            self.weights = params.word_weights
        self.enc_mlp = MLP(params.encoding_cell_size * 2 +
                           params.state_cell_size, [400, 200],
                           dropout_rate=params.dropout)
        self.enc_linear = nn.Linear(200, self.num_zt)
        self.dec_mlp = MLP(self.num_zt, [200, 200],
                           dropout_rate=params.dropout)
        # TODO: dynamic LSTM, dropout and num_layers
        self.dec_rnn_1 = nn.LSTM(params.state_cell_size + 200,
                                 200 + self.num_zt)
        self.dec_linear_1 = nn.Linear(200 + self.num_zt, self.vocab_size)
        self.dec_rnn_2 = nn.LSTM(params.state_cell_size + 200,
                                 2 * (200 + self.num_zt))
        self.dec_linear_2 = nn.Linear(2 * (200 + self.num_zt), self.vocab_size)

        self.linear1 = nn.Linear()

    def encode(self, inputs, h_prev):
        enc_inputs = torch.cat(
            [h_prev, inputs],
            1)  # [batch, encoding_cell_size * 2 + state_cell_size]

        net = self.enc_mlp(enc_inputs)
        logits_z = self.enc_linear(net)
        log_q_z = F.log_softmax(logits_z)

        return logits_z, log_q_z

    def decode(self, z_samples, h_prev, dec_input_embedding, forward=False):
        net = self.dec_mlp(z_samples)
        # decoder for user utterance
        dec_input_1 = torch.cat([h_prev, net],
                                1)  # [batch, state_cell_size + 200]

        if not forward:
            print(type(dec_input_embedding[0]))
            dec_input_embedding[0] = dec_input_embedding[0][:, 0:-1, :]
            dec_input_embedding[1] = dec_input_embedding[1][:, 0:-1, :]
            dec_outs_1, final_state_1 = self.dec_rnn_1(
                dec_input_embedding[0], (dec_input_1, dec_input_1))

            dec_input_2_c = torch.cat([dec_input_1, final_state_1[0]],
                                      dim=1)  # [batch, state_cell_size + 200]
            dec_input_2_h = torch.cat([dec_input_1, final_state_1[1]],
                                      dim=1)  # [batch, state_cell_size + 200]

            dec_outs_2, final_state_2 = self.dec_rnn_2(
                dec_input_embedding[1], (dec_input_2_c, dec_input_2_h))
        return dec_outs_1, dec_outs_2

    def forward(self,
                inputs,
                state,
                dec_input_embedding,
                dec_seq_lens,
                output_tokens,
                forward=False,
                prev_z_t=None):
        if params.with_direct_transition:
            assert prev_z_t is not None
        if self._state_is_tuple:
            (c_prev, h_prev) = state
        # encode
        logits_z, log_q_z = self.encode(inputs, h_prev)

        # sample
        z_samples, logits_z_samples = gumbel_softmax(
            logits_z, self.tau, hard=False)  # [batch, num_zt]
        print("z_samples")
        print(z_samples)

        #decode
        dec_outs_1, dec_outs_2 = self.decode(z_samples, h_prev,
                                             dec_input_embedding, forward)

        # for computing BOW loss
        if params.with_bow_loss:
            bow_fc1 = layers.fully_connected(dec_input_1,
                                             400,
                                             activation_fn=tf.tanh,
                                             scope="bow_fc1")
            if params.dropout > 0:
                bow_fc1 = F.dropout(bow_fc1, p=params.dropout)
            self.bow_logits1 = layers.fully_connected(bow_fc1,
                                                      self.vocab_size,
                                                      activation_fn=None,
                                                      scope="bow_project1")
            print(
                "self.bow_logits[1]")  #(None, vocab_size), None is batch size
            print(self.bow_logits1)
            # sys.exit()

            bow_fc2 = layers.fully_connected(dec_input_2_h,
                                             400,
                                             activation_fn=tf.tanh,
                                             scope="bow_fc2")
            if params.dropout > 0:
                bow_fc2 = F.dropout(bow_fc2, p=params.dropout)
            self.bow_logits2 = layers.fully_connected(bow_fc2,
                                                      self.vocab_size,
                                                      activation_fn=None,
                                                      scope="bow_project2")

        if params.with_direct_transition:
            net3 = slim.stack(prev_z_t, slim.fully_connected, [100, 100])
            if params.dropout > 0:
                net3 = F.dropout(net3, params.dropout)
            p_z = slim.fully_connected(net3,
                                       self.num_zt,
                                       activation_fn=tf.nn.softmax)
            p_z = tf.identity(p_z, name="p_z_transition")
            log_p_z = torch.log(p_z + 1e-20)  # equation 5

        else:
            net3 = slim.stack(h_prev, slim.fully_connected, [100, 100])
            if params.dropout < 1.0:
                net3 = F.dropout(net3, params.dropout)
            p_z = slim.fully_connected(net3,
                                       self.num_zt,
                                       activation_fn=tf.nn.softmax)
            log_p_z = torch.log(p_z + 1e-20)  # equation 5

        recur_input = torch.cat([net2, inputs], 1,
                                'recNet_inputs')  # [batch, 600]
        next_state = self.state_cell(inputs=recur_input, state=state)

        return elbo_t, logits_z_samples, next_state[
            1], p_z, self.bow_logits1, self.bow_logits2
