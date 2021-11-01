"""Torch version of https://github.com/wyshi/Unsupervised-Structure-Learning
"""
import sys

import torch
from torch import nn
from loguru import logger
from transformers import AutoModel

sys.path.append("..")
import params
from .linear_vae_cell import LinearVAECell


class LinearVRNN(nn.Module):
    """
    VRNN with gumbel-softmax
    """
    def __init__(self):
        super(LinearVRNN, self).__init__()

        if params.use_bert:
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
        else:

            self.embedding = nn.Embedding(params.max_vocab_cnt,
                                          params.embed_size)  # (1770, 300)
            if params.cell_type == "gru":
                self.sent_rnn = nn.GRU(params.embed_size,
                                       params.encoding_cell_size,
                                       params.num_layer,
                                       batch_first=True)
            else:
                self.sent_rnn = nn.LSTM(
                    params.embed_size,  # 300
                    params.encoding_cell_size,  # 400
                    params.num_layer,  # 1
                    batch_first=True)

        self.vae_cell = LinearVAECell(
            state_is_tuple=True if self.cell_type == "lstm" else False)

        if params.dropout not in (None, 0):
            self.dropout = nn.Dropout(params.dropout)  # 0.5
        if params.use_struct_attention:
            # Input Memory Net: Joint Embedding to Query Matrix
            # [batch, length, params.encoding_cell_size * 2] -> [batch, length, 210 * 2]
            self.input_memory = nn.Linear(
                params.encoding_cell_size * 2,  # 400 * 2
                (200 + params.n_state) * 2)  # (200 + 10) * 2

    def forward(self,
                usr_input_sent,
                sys_input_sent,
                dialog_length_mask,
                usr_input_mask,
                sys_input_mask,
                training=True):
        #------------------------- utterance embedding -------------------------#
        if params.use_bert:
            usr_input_embedding = self.bert(
                usr_input_sent,
                attention_mask=usr_input_mask).pooler_output  # [CLS] token
            sys_input_embedding = self.bert(
                sys_input_sent,
                attention_mask=sys_input_mask).pooler_output  # [CLS] token
        else:
            usr_input_embedding = self.embedding(
                usr_input_sent
            )  # [batch_size, max_dialog_len, max_utt_len, embed_size] (40, 13, 50, 300)
            usr_input_embedding = usr_input_embedding.view(
                [-1, params.max_utt_len, params.embed_size]
            )  # [batch_size * max_dialog_len, max_utt_len, embed_size] (520, 50, 300)

            sys_input_embedding = self.embedding(
                sys_input_sent
            )  # [batch_size, max_dialog_len, max_utt_len, embed_size] (40, 13, 50, 300)
            sys_input_embedding = sys_input_embedding.view(
                [-1, params.max_utt_len, params.embed_size]
            )  # [batch_size * max_dialog_len, max_utt_len, embed_size] (520, 50, 300)

            usr_sent_mask = torch.sign(
                usr_input_mask.view(-1, params.max_utt_len)
            )  # [batch_size * max_dialog_len, max_utt_len] (520, 50)
            sys_sent_mask = torch.sign(
                sys_input_mask.view(-1, params.max_utt_len)
            )  # [batch_size * max_dialog_len, max_utt_len] (520, 50)
            usr_sent_len = torch.sum(usr_sent_mask,
                                    dim=1)  # [batch_size * max_dialog_len] (520)
            sys_sent_len = torch.sum(sys_sent_mask,
                                    dim=1)  # [batch_size * max_dialog_len] (520)
            if params.cell_type == "gru":
                usr_sent_embeddings, _ = self.sent_rnn(usr_input_embedding)
                sys_sent_embeddings, _ = self.sent_rnn(sys_input_embedding)
            else:
                usr_sent_embeddings, (_, _) = self.sent_rnn(
                    usr_input_embedding
                )  # [batch_size * max_dialog_len, max_utt_len, encoding_cell_size] (520, 50, 400)

                sys_sent_embeddings, (_, _) = self.sent_rnn(
                    sys_input_embedding
                )  # [batch_size * max_dialog_len, max_utt_len, encoding_cell_size] (520, 50, 400)

            usr_sent_embedding = torch.zeros(
                params.batch_size * params.max_dialog_len,
                params.encoding_cell_size)  # (520, 400)
            sys_sent_embedding = torch.zeros(
                params.batch_size * params.max_dialog_len,
                params.encoding_cell_size)
            if params.use_cuda and torch.cuda.is_available():
                usr_sent_embedding = usr_sent_embedding.cuda()
                sys_sent_embedding = sys_sent_embedding.cuda()

        for i in range(usr_sent_embedding.shape[0]):
            if usr_sent_len[i] > 0:
                usr_sent_embedding[i] = usr_sent_embeddings[i,
                                                            usr_sent_len[i] -
                                                            1, :]
            if sys_sent_len[i] > 0:
                sys_sent_embedding[i] = sys_sent_embeddings[i,
                                                            sys_sent_len[i] -
                                                            1, :]

        usr_sent_embedding = usr_sent_embedding.view(
            -1, params.max_dialog_len, params.encoding_cell_size
        )  # [batch_size, max_dialog_len, encoding_cell_size] (40, 13, 400)
        sys_sent_embedding = sys_sent_embedding.view(
            -1, params.max_dialog_len, params.encoding_cell_size
        )  # [batch_size, max_dialog_len, encoding_cell_size] (40, 13, 400)

        if params.dropout not in (None, 0):
            usr_sent_embedding = self.dropout(usr_sent_embedding)
            sys_sent_embedding = self.dropout(sys_sent_embedding)

        joint_embedding = torch.cat(
            [usr_sent_embedding, sys_sent_embedding], dim=2
        )  # [batch_size, max_dialog_len, encoding_cell_size * 2] (40, 13, 800)

        # Pytorch-struct
        if params.use_struct_attention:
            input_query = self.input_memory(joint_embedding)
            input_query = input_query.view(
                params.batch_size, -1, 2,
                200 + params.n_state)  # [batch_size, max_dialog_len, 2, 210]
        else:
            input_query = None

        #----------------- dialogue turn (state) recurrance -------------------#
        dec_input_embedding_usr = self.embedding(
            usr_input_sent
        )  # [batch_size, max_dialog_len, max_utt_len, embed_size] (40, 13, 50, 300)
        dec_input_embedding_sys = self.embedding(
            sys_input_sent
        )  # [batch_size, max_dialog_len, max_utt_len, embed_size] (40, 13, 50, 300)
        dec_input_embedding = [
            dec_input_embedding_usr, dec_input_embedding_sys
        ]

        dec_seq_lens_usr = torch.sum(torch.sign(usr_input_mask),
                                     dim=2)  # (40, 13)
        dec_seq_lens_sys = torch.sum(torch.sign(sys_input_mask),
                                     dim=2)  # (40, 13)
        dec_seq_lens = [dec_seq_lens_usr, dec_seq_lens_sys]

        output_tokens = [usr_input_sent, sys_input_sent]

        prev_z = torch.ones(params.batch_size, params.n_state)
        elbo_ts = []
        rc_losses = []
        kl_losses = []
        bow_losses = []
        z_ts = []
        p_ts = []
        bow_logits_1 = []
        bow_logits_2 = []
        if params.cell_type == "gru":
            state = torch.zeros(params.batch_size, params.n_state)  # (40, 10)
            if params.use_cuda and torch.cuda.is_available():
                state = state.cuda()
        else:
            h = c = torch.zeros(params.batch_size, params.n_state)  # (40, 10)
            if params.use_cuda and torch.cuda.is_available():
                h = h.cuda()
                c = c.cuda()
            state = (h, c)
        # TODO: this for loop has not be parallelized
        for utt in range(params.max_dialog_len):
            inputs = joint_embedding[:, utt, :]
            dec_input_emb = [
                dec_input_embedding[0][:, utt, :, :],
                dec_input_embedding[1][:, utt, :, :]
            ]
            dec_seq_len = [dec_seq_lens[0][:, utt], dec_seq_lens[1][:, utt]]
            output_token = [
                output_tokens[0][:, utt, :], output_tokens[1][:, utt, :]
            ]

            losses, z_samples, state, p_z, bow_logits1, bow_logits2 = self.vae_cell(
                inputs,
                state,
                dec_input_emb,
                dec_seq_len,
                output_token,
                prev_z_t=prev_z,
                prev_embeddings=joint_embedding[:, :utt, :],
                input_query=input_query)

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
            # TODO: check whether have converged to local minimas
            elbo_ts.append(losses[0])
            rc_losses.append(losses[1])
            kl_losses.append(losses[2])
            bow_losses.append(losses[3])
            z_ts.append(zts_onehot)
            p_ts.append(p_z)
            bow_logits_1.append(bow_logits1)
            bow_logits_2.append(bow_logits2)

        mask_len = (torch.sum(usr_input_mask) + torch.sum(sys_input_mask))
        elbo_ts = torch.stack(elbo_ts)
        elbo_t_avg = torch.sum(elbo_ts) / mask_len
        rc_losses = torch.stack(rc_losses)
        rc_loss_avg = torch.sum(rc_losses) / mask_len
        kl_losses = torch.stack(kl_losses)
        kl_loss_avg = torch.sum(kl_losses) / mask_len
        bow_losses = torch.stack(bow_losses)
        bow_loss_avg = torch.sum(bow_losses) / mask_len

        z_ts = torch.stack(z_ts)
        p_ts = torch.stack(p_ts)
        bow_logits_1 = torch.stack(bow_logits_1)
        bow_logits_2 = torch.stack(bow_logits_2)

        z_ts = z_ts.permute(1, 0, 2).cpu().detach().numpy()
        p_ts = p_ts.permute(1, 0, 2).cpu().detach().numpy()
        bow_logits_1 = bow_logits_1.permute(1, 0, 2).cpu().detach().numpy()
        bow_logits_2 = bow_logits_2.permute(1, 0, 2).cpu().detach().numpy()

        if training:
            return elbo_t_avg, rc_loss_avg, kl_loss_avg, bow_loss_avg
        else:
            return usr_input_sent.cpu().detach().numpy(), sys_input_sent.cpu(
            ).detach().numpy(), z_ts, p_ts, bow_logits_1, bow_logits_2
