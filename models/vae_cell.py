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
        self.enc_mlp = MLP(params.encoding_cell_size * 2 + params.n_state,
                           [400, 200],
                           dropout_rate=params.dropout)
        self.enc_linear = nn.Linear(200, self.num_zt)
        self.dec_mlp = MLP(self.num_zt, [200, 200],
                           dropout_rate=params.dropout)
        self.dec_rnn_1 = nn.LSTMCell(200 + self.num_zt)
        self.dec_linear_1 = nn.Linear(200 + self.num_zt, self.vocab_size)
        self.dec_rnn_2 = nn.LSTMCell(2 * (200 + self.num_zt))
        self.dec_linear_2 = nn.Linear(2 * (200 + self.num_zt), self.vocab_size)

    def encode(self,
               inputs,
               state,
               dec_input_embedding,
               dec_seq_lens,
               output_tokens,
               forward=False,
               prev_z_t=None):
        if self._state_is_tuple:
            (c_prev, h_prev) = state
        if params.with_direct_transition:
            assert prev_z_t is not None

        print(type(dec_input_embedding[0]))

        enc_inputs = torch.cat(
            [h_prev, inputs],
            1)  # [batch, sent_encoding_size*2 + state_cell_size]

        net = self.enc_mlp(enc_inputs)
        logits_z = self.enc_linear(net)
        log_q_z = F.log_softmax(logits_z)

        return logits_z, log_q_z

    def decode(self, z_samples):
        net = self.dec_mlp(z_samples)
        # decoder for user utterance
        dec_input_1 = torch.cat([h_prev, net],
                                1)  # [batch, state_cell_size + 200]

        dec_init_state_1 = tf.contrib.rnn.LSTMStateTuple(
            dec_input_1, dec_input_1)

        if not forward:
            dec_input_embedding[0] = dec_input_embedding[0][:, 0:-1, :]
            dec_input_embedding[1] = dec_input_embedding[1][:, 0:-1, :]
            dec_outs_1, final_state_1 = dynamic_rnn_decoder(
                self.decoder_cell_1,
                loop_func_1,
                inputs=dec_input_embedding[0],
                sequence_length=dec_seq_lens[0] - 1,
                scope_name="dynamic_rnn_decoder_1")

            dec_input_2_c = tf.concat(
                [dec_input_1, final_state_1[0]], 1,
                'decNet_inputs_2_c')  # [batch, state_cell_size + 200]
            dec_input_2_h = tf.concat(
                [dec_input_1, final_state_1[1]], 1,
                'decNet_inputs_2_h')  # [batch, state_cell_size + 200]

            dec_init_state_2 = tf.contrib.rnn.LSTMStateTuple(
                dec_input_2_c, dec_input_2_h)
            loop_func_2 = decoder_fn_lib.context_decoder_fn_train(
                dec_init_state_2, selected_attribute_embedding)
            dec_outs_2, final_state_2 = dynamic_rnn_decoder(
                self.decoder_cell_2,
                loop_func_2,
                inputs=dec_input_embedding[1],
                sequence_length=dec_seq_lens[1] - 1,
                scope_name="dynamic_rnn_decoder_2")
        return dec_outs_1, dec_outs_2

    def forward(self,
                inputs,
                state,
                dec_input_embedding,
                dec_seq_lens,
                output_tokens,
                forward=False,
                prev_z_t=None):
        # encode
        logits_z, log_q_z = self.encode(inputs)

        # sample
        z_samples, logits_z_samples = gumbel_softmax(
            logits_z, self.tau, hard=False)  # [batch, num_zt]
        print("z_samples")
        print(z_samples)

        #decode
        dec_outs_1, dec_outs_2 = self.decode(z_samples)

        # for computing BOW loss
        if self.config.with_bow_loss:
            bow_fc1 = layers.fully_connected(dec_input_1,
                                             400,
                                             activation_fn=tf.tanh,
                                             scope="bow_fc1")
            if self.config.keep_prob < 1.0:
                bow_fc1 = tf.nn.dropout(bow_fc1, self.config.keep_prob)
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
            if self.config.keep_prob < 1.0:
                bow_fc2 = tf.nn.dropout(bow_fc2, self.config.keep_prob)
            self.bow_logits2 = layers.fully_connected(bow_fc2,
                                                      self.vocab_size,
                                                      activation_fn=None,
                                                      scope="bow_project2")

        with vs.variable_scope("priorNetwork") as priorScope:
            if self.config.with_direct_transition:
                net3 = slim.stack(prev_z_t, slim.fully_connected, [100, 100])
                if self.config.keep_prob < 1.0:
                    net3 = tf.nn.dropout(net3, self.config.keep_prob)
                p_z = slim.fully_connected(net3,
                                           self.num_zt,
                                           activation_fn=tf.nn.softmax)
                p_z = tf.identity(p_z, name="p_z_transition")
                log_p_z = tf.log(p_z + 1e-20)  # equation 5

            else:
                net3 = slim.stack(h_prev, slim.fully_connected, [100, 100])
                if self.config.keep_prob < 1.0:
                    net3 = tf.nn.dropout(net3, self.config.keep_prob)
                p_z = slim.fully_connected(net3,
                                           self.num_zt,
                                           activation_fn=tf.nn.softmax)
                log_p_z = tf.log(p_z + 1e-20)  # equation 5

        with vs.variable_scope("recurrence") as recScope:
            recur_input = tf.concat([net2, inputs], 1,
                                    'recNet_inputs')  # [batch, 600]
            next_state = self.state_cell(inputs=recur_input, state=state)

        return elbo_t, logits_z_samples, next_state[
            1], p_z, self.bow_logits1, self.bow_logits2
