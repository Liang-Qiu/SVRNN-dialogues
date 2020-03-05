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
# from models.dynamic_VAE import dynamic_vae


class VRNN(nn.Module):
    """
    VRNN with gumbel-softmax
    """
    def __init__(self):
        super(VRNN, self).__init__()

        # neural layers
        self.embedding = nn.Embedding(params.n_vocab, params.embed_size)

        if params.cell_type == "gru":
            self.sent_rnn = nn.GRU(params.embed_size,
                                   params.encoding_cell_size,
                                   params.num_layer,
                                   batch_first=True,
                                   dropout=params.dropout)
        else:
            self.sent_rnn = nn.LSTM(params.embed_size,
                                    params.encoding_cell_size,
                                    params.num_layer,
                                    batch_first=True,
                                    dropout=params.dropout)

        self.vae_cell = VAECell(
            num_units=300,
            # num_zt=self.config.n_state,
            # vocab_size=self.n_vocab,
            # max_utt_len=self.max_utt_len,
            # config=config,
            use_peepholes=False,
            cell_clip=None,
            initializer=None,
            num_proj=None,
            proj_clip=None,
            num_unit_shards=None,
            num_proj_shards=None,
            forget_bias=1.0,
            state_is_tuple=True)

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
        sys_sent_embedding = sys_sent_embedding.view(-1, params.max_dialog_len,
                                                     params.encoding_cell_size)

        if params.dropout > 0:
            usr_sent_embedding = F.dropout(usr_sent_embedding,
                                           p=params.dropout)
            sys_sent_embedding = F.dropout(sys_sent_embedding,
                                           p=params.dropout)

        joint_embedding = torch.cat(
            [usr_input_embedding, sys_input_embedding],
            dim=2)  # (batch, dialog_len, embedding_size*2) (16, 10, 800)

        ########################### state level ############################
        dec_input_embedding_usr = self.embedding(
            usr_input_sent)  # (16, 10, 50, 300)
        dec_input_embedding_sys = self.embedding(
            sys_input_sent)  # (16, 10, 50, 300)
        dec_input_embedding = [
            dec_input_embedding_usr, dec_input_embedding_sys
        ]

        dec_seq_lens_usr = torch.sum(torch.sign(self.usr_full_mask), dim=2)
        dec_seq_lens_sys = torch.sum(torch.sign(self.sys_full_mask), dim=2)
        dec_seq_lens = [dec_seq_lens_usr, dec_seq_lens_sys]

        output_tokens = [usr_input_sent, sys_input_sent]

    #     self.initial_prev_z = tf.placeholder(tf.float32,
    #                                          (None, self.config.n_state),
    #                                          'initial_prev_z')
    #     losses, z_ts, p_ts, bow_logits1, bow_logits2 = dynamic_vae(
    #         self.VAE_cell,
    #         joint_embedding,
    #         dec_input_embedding,
    #         dec_seq_lens,
    #         output_tokens,
    #         z_t_size=self.config.n_state,
    #         sequence_length=self.dialog_length_mask,
    #         initial_state=None,
    #         dtype=tf.float32,
    #         parallel_iterations=None,
    #         swap_memory=False,
    #         time_major=False,
    #         scope=None,
    #         initial_prev_z=self.initial_prev_z)

    #     z_ts = F.softmax(z_ts)  # (16, 10, 12)
    #     z_ts_mask = tf.to_float(
    #         torch.sign(torch.sum(self.usr_full_mask, dim=2)))  # (16, 10)
    #     z_ts_mask = tf.expand_dims(z_ts_mask, 2)  # (16, 10, 1)
    #     self.z_ts = z_ts * z_ts_mask
    #     self.p_ts = p_ts
    #     self.bow_logits1 = bow_logits1
    #     self.bow_logits2 = bow_logits2
    #     loss_avg = torch.sum(losses) / tf.to_float(
    #         torch.sum(self.usr_full_mask) + torch.sum(self.sys_full_mask))

    #     loss_avg = tf.identity(loss_avg, name="loss_average")

    #     self.basic_loss = loss_avg
    #     tf.summary.scalar("basic_loss", self.basic_loss)

    #     self.summary_op = tf.summary.merge_all()

    # self.saver = tf.train.Saver(tf.global_variables(),
    #                             write_version=tf.train.SaverDef.V2)

    # @staticmethod
    # def print_loss(prefix, loss_names, losses, postfix):
    #     template = "%s "
    #     for name in loss_names:
    #         template += "%s " % name
    #         template += " %f "
    #     template += "%s"
    #     template = re.sub(' +', ' ', template)
    #     avg_losses = []
    #     values = [prefix]

    #     for loss in losses:
    #         values.append(np.mean(loss))
    #         avg_losses.append(np.mean(loss))
    #     values.append(postfix)

    #     print(template % tuple(values))
    #     return avg_losses

    # def valid(self,
    #           name,
    #           sess,
    #           valid_feed,
    #           labeled_feed=None,
    #           labeled_labels=None):
    #     elbo_losses = []

    #     while True:
    #         batch = valid_feed.next_batch()
    #         if self.config.with_label_loss:
    #             labeled_batch = labeled_feed.next_batch()
    #         if batch is None:
    #             break
    #         if self.config.with_label_loss:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=labeled_batch,
    #                                           labeled_labels=labeled_labels,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)
    #         else:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=None,
    #                                           labeled_labels=None,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)

    #         elbo_loss = sess.run(self.basic_loss, feed_dict)
    #         elbo_losses.append(elbo_loss)

    #     avg_losses = self.print_loss(name, ['elbo_losses valid'],
    #                                  [elbo_losses], "")

    #     return avg_losses[0]

    # def get_zt(self,
    #            global_t,
    #            sess,
    #            train_feed,
    #            update_limit=5000,
    #            labeled_feed=None,
    #            labeled_labels=None):
    #     local_t = 0
    #     start_time = time.time()
    #     results = []
    #     i_batch = 0
    #     while i_batch < train_feed.num_batch:
    #         # print(train_feed.num_batch)
    #         batch = train_feed.next_batch()
    #         if self.config.with_label_loss:
    #             labeled_batch = labeled_feed.next_batch()
    #         if batch is None:
    #             break
    #         if update_limit is not None and local_t >= update_limit:
    #             break

    #         if self.config.with_label_loss:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=labeled_batch,
    #                                           labeled_labels=labeled_labels,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)
    #         else:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=None,
    #                                           labeled_labels=labeled_labels,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)

    #         fetch_list = [
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/biases:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/biases:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/biases:0'
    #         ]
    #         z_ts, p_ts, bow_logits1, bow_logits2, w1, b1, w2, b2, w3, b3 = sess.run(
    #             [self.z_ts, self.p_ts, self.bow_logits1, self.bow_logits2] +
    #             fetch_list, feed_dict)
    #         # print(w_1)

    #         global_t += 1
    #         local_t += 1
    #         i_batch += 1
    #         result = [batch, z_ts, p_ts, bow_logits1, bow_logits2]
    #         results.append(result)
    #         # if local_t % (train_feed.num_batch // 10) == 0:
    #         #     # kl_w = sess.run(self.kl_w, {self.global_t: global_t}
    #         #     print("%.2f" % (train_feed.ptr / float(train_feed.num_batch)))

    #     epoch_time = time.time() - start_time
    #     return results, [w1, b1, w2, b2, w3, b3]

    # def get_log_prob(self,
    #                  global_t,
    #                  sess,
    #                  train_feed,
    #                  transition_prob,
    #                  update_limit=5000,
    #                  labeled_feed=None,
    #                  labeled_labels=None):
    #     # reconstruct the decoder to get self.bow_logits1 and self.bow_logits2
    #     local_t = 0
    #     start_time = time.time()
    #     results = []
    #     i_batch = 0
    #     while i_batch < train_feed.num_batch:
    #         # print(train_feed.num_batch)
    #         batch = train_feed.next_batch()
    #         if self.config.with_label_loss:
    #             labeled_batch = labeled_feed.next_batch()
    #         if batch is None:
    #             break
    #         if update_limit is not None and local_t >= update_limit:
    #             break

    #         if self.config.with_label_loss:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=labeled_batch,
    #                                           labeled_labels=labeled_labels,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)
    #         else:
    #             feed_dict = self.batch_2_feed(batch=batch,
    #                                           labeled_batch=None,
    #                                           labeled_labels=labeled_labels,
    #                                           global_t=None,
    #                                           use_prior=False,
    #                                           repeat=1)

    #         fetch_list = [
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/biases:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/biases:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/weights:0',
    #             'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/biases:0'
    #         ]
    #         z_ts, p_ts, w1, b1, w2, b2, w3, b3 = sess.run(
    #             [self.z_ts, self.p_ts] + fetch_list, feed_dict)
    #         # print(w_1)

    #         global_t += 1
    #         local_t += 1
    #         i_batch += 1
    #         result = [batch, z_ts, p_ts]
    #         results.append(result)
    #         if local_t % (train_feed.num_batch // 10) == 0:
    #             # kl_w = sess.run(self.kl_w, {self.global_t: global_t}
    #             print("%.2f" % (train_feed.ptr / float(train_feed.num_batch)))

    #     epoch_time = time.time() - start_time
    #     return results, [w1, b1, w2, b2, w3, b3]

    # def print_model_stats(self, tvars):
    #     total_parameters = 0
    #     for variable in tvars:
    #         # shape is an array of tf.Dimension
    #         shape = variable.get_shape()
    #         variable_parametes = 1
    #         for dim in shape:
    #             variable_parametes *= dim.value
    #         print("Trainable %s with %d parameters" %
    #               (variable.name, variable_parametes))
    #         total_parameters += variable_parametes
    #     print("Total number of trainable parameters is %d" % total_parameters)

    # def optimize(self, sess, config, loss, log_dir):
    #     if log_dir is None:
    #         return
    #     # optimization
    #     if self.scope is None:
    #         tvars = tf.trainable_variables()
    #     else:
    #         tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                   scope=self.scope)
    #     grads = tf.gradients(loss, tvars)
    #     if config.grad_clip is not None:
    #         grads, _ = tf.clip_by_global_norm(grads,
    #                                           tf.constant(config.grad_clip))
    #     # add gradient noise
    #     if config.grad_noise > 0:
    #         grad_std = tf.sqrt(config.grad_noise /
    #                            tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
    #         grads = [
    #             g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std)
    #             for g in grads
    #         ]

    #     if config.op == "adam":
    #         print("Use Adam")
    #         optimizer = tf.train.AdamOptimizer(config.init_lr)
    #     elif config.op == "rmsprop":
    #         print("Use RMSProp")
    #         optimizer = tf.train.RMSPropOptimizer(config.init_lr)
    #     else:
    #         print("Use SGD")
    #         optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    #     self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
    #     self.print_model_stats(tvars)
    #     train_log_dir = os.path.join(log_dir, "checkpoints")
    #     print("Save summary to %s" % log_dir)
    #     self.train_summary_writer = tf.summary.FileWriter(
    #         train_log_dir, sess.graph)
