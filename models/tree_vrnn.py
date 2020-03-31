from __future__ import print_function
from __future__ import division

import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append("..")
import params
from .tree_vae_cell import TreeVAECell
from utils.loss import BPR_BOW_loss_single


class TreeVRNN(nn.Module):
    def __init__(self):
        super(TreeVRNN, self).__init__()

        self.embedding = nn.Embedding(params.max_vocab_cnt, params.emb_dim)

        if params.cell_type == "gru":
            self.sent_rnn = nn.GRU(params.embed_size,
                                   params.sen_hidden_dim,
                                   params.num_layer,
                                   batch_first=True)
            self.vae_cell = TreeVAECell(state_is_tuple=False)
        else:
            self.sent_rnn = nn.LSTM(params.embed_size,
                                    params.sen_hidden_dim,
                                    params.num_layer,
                                    batch_first=True)
            self.vae_cell = TreeVAECell(state_is_tuple=True)
        if params.dropout not in (None, 0):
            self.dropout = nn.Dropout(params.dropout)

    def forward(self,
                enc_batch,
                enc_lens,
                dec_batch,
                target_batch,
                padding_mask,
                tgt_index,
                training=True):
        ########################## sentence embedding  ##################
        input_embedding = self.embedding(enc_batch)  # (5, 9, 50, 300)

        input_embedding = input_embedding.view(
            [-1, params.max_enc_steps, params.embed_size])  # (45, 50, 300)

        if params.cell_type == "gru":
            sent_embeddings, _ = self.sent_rnn(input_embedding)
        else:
            sent_embeddings, (_, _) = self.sent_rnn(
                input_embedding)  # (45, 50, 400)

        sent_embedding = torch.zeros(
            params.branch_batch_size * params.sen_batch_size,
            params.sen_hidden_dim)

        if params.use_cuda and torch.cuda.is_available():
            sent_embedding = sent_embedding.cuda()

        for i in range(sent_embedding.shape[0]):
            if enc_lens[i] > 0:
                sent_embedding[i] = sent_embeddings[i, enc_lens[i] - 1, :]

        sent_embedding = sent_embedding.view(
            -1, params.sen_batch_size, params.sen_hidden_dim)  # (5, 9, 400)

        if params.dropout not in (None, 0):
            sent_embedding = self.dropout(sent_embedding)

        ########################### state level ############################
        dec_input_embedding = self.embedding(dec_batch)  # (5, 50, 300)

        prev_z = torch.ones(params.branch_batch_size, params.n_state)

        z_samples_list = []
        z_samples_context_list = []
        h_list = []
        z_onehot_list = []
        p_z_list = []
        q_z_list = []
        log_p_z_list = []
        log_q_z_list = []

        if params.cell_type == "gru":
            state = torch.zeros(params.branch_batch_size,
                                params.state_cell_size)
            if params.use_cuda and torch.cuda.is_available():
                state = state.cuda()
        else:
            h = c = torch.zeros(params.branch_batch_size,
                                params.state_cell_size)
            if params.use_cuda and torch.cuda.is_available():
                h = h.cuda()
                c = c.cuda()
            state = (h, c)

        for utt in range(params.sen_batch_size):
            inputs = sent_embedding[:, utt, :]

            z_samples, z_samples_context, state, p_z, q_z, log_p_z, log_q_z = self.vae_cell(
                inputs,
                state,
                prev_z_t=prev_z,
                prev_embeddings=sent_embedding[:, :utt, :])
            # save the previous state
            z_samples_list.append(z_samples)
            z_samples_context_list.append(z_samples_context)
            if params.cell_type == "gru":
                h_list.append(state)
            else:
                h_list.append(state[0])
            p_z_list.append(p_z)
            q_z_list.append(q_z)
            log_p_z_list.append(log_p_z)
            log_q_z_list.append(log_q_z)

            shape = z_samples.size()
            _, ind = z_samples.max(dim=-1)
            zts_onehot = torch.zeros_like(z_samples).view(-1, shape[-1])
            if params.use_cuda and torch.cuda.is_available():
                zts_onehot = zts_onehot.cuda()
            zts_onehot.scatter_(1, ind.view(-1, 1), 1)
            zts_onehot = zts_onehot.view(*shape)
            # stop gradient
            zts_onehot = (zts_onehot - z_samples).detach() + z_samples
            prev_z = zts_onehot
            # TODO: check whether have converged to local minima
            z_onehot_list.append(zts_onehot)

        # decode
        # pick tgt_idx from encoder
        h_prev = torch.zeros(params.branch_batch_size, params.n_state)
        z_samples_dec = torch.zeros(params.branch_batch_size, params.n_state)
        z_samples_context_dec = torch.zeros(params.branch_batch_size,
                                            params.n_state)
        p_z_dec = torch.zeros(params.branch_batch_size, params.n_state)
        q_z_dec = torch.zeros(params.branch_batch_size, params.n_state)
        log_p_z_dec = torch.zeros(params.branch_batch_size, params.n_state)
        log_q_z_dec = torch.zeros(params.branch_batch_size, params.n_state)
        if params.use_cuda and torch.cuda.is_available():
            h_prev = h_prev.cuda()
            z_samples_dec = z_samples_dec.cuda()
            z_samples_context_dec = z_samples_context_dec.cuda()
            p_z_dec = p_z_dec.cuda()
            q_z_dec = q_z_dec.cuda()
            log_p_z_dec = log_p_z_dec.cuda()
            log_q_z_dec = log_q_z_dec.cuda()
        for i in range(params.branch_batch_size):
            h_prev[i, :] = h_list[tgt_index[i]][i, :]
            z_samples_dec[i, :] = z_samples_list[tgt_index[i]][i, :]
            z_samples_context_dec[i, :] = z_samples_context_list[tgt_index[i]][
                i, :]
            p_z_dec[i, :] = p_z_list[tgt_index[i]][i, :]
            q_z_dec[i, :] = q_z_list[tgt_index[i]][i, :]
            log_p_z_dec[i, :] = p_z_list[tgt_index[i]][i, :]
            log_q_z_dec[i, :] = log_q_z_list[tgt_index[i]][i, :]

        dec_outs, bow_logits = self.vae_cell.decode(
            z_samples_dec,
            h_prev,
            dec_input_embedding,
            z_samples_context=z_samples_context_dec)
        elbo_t, rc_loss, kl_loss, bow_loss = BPR_BOW_loss_single(
            target_batch,
            dec_outs,
            padding_mask,
            log_p_z_dec,
            log_q_z_dec,
            p_z_dec,
            q_z_dec,
            bow_logits=bow_logits)

        mask_len = torch.sum(padding_mask)
        elbo_t_avg = elbo_t / mask_len
        rc_loss_avg = rc_loss / mask_len
        kl_loss_avg = kl_loss / mask_len
        bow_loss_avg = bow_loss / mask_len

        z_ts = torch.stack(z_onehot_list)
        p_ts = torch.stack(p_z_list)
        z_ts = z_ts.permute(1, 0, 2).cpu().detach().numpy()
        p_ts = p_ts.permute(1, 0, 2).cpu().detach().numpy()
        bow_logits = bow_logits.cpu().detach().numpy()

        if training:
            return elbo_t_avg, rc_loss_avg, kl_loss_avg, bow_loss_avg
        else:
            return enc_batch.cpu().detach().numpy(), z_ts, p_ts, bow_logits
