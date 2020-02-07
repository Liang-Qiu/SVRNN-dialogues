import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_matching.models.attention import AttentionDecoder


class GSNModel(nn.Module):
    def __init__(self, config, vocab):
        """Constructor.
        
        Args:
            config ([type]): [description]
            vocab ([type]): [description]
            vsize ([type]): [description]
        """
        super(GSNModel, self).__init__()

        self.config = config
        self.vocab = vocab
        self.dropout = config['graph_structure_net']['dropout']
        self.sen_hidden_dim = config['graph_structure_net']['sen_hidden_dim']
        self.branch_batch_size = config['graph_structure_net'][
            'branch_batch_size']
        self.sen_batch_size = config['graph_structure_net']['sen_batch_size']
        self.max_enc_steps = config['graph_structure_net']['max_enc_steps']
        self.emb_dim = config['graph_structure_net']['emb_dim']

        vsize = vocab.count

        # TODO: original GSN use truncated_norm initializer widely.
        # Let's use Torch default initializer (Xavier) for now.

        # TODO: use GloVe as pre-trained embedding
        self.embeddings = nn.Embedding(vsize, self.emb_dim)
        self.encoder = nn.LSTM(self.emb_dim,
                               self.sen_hidden_dim,
                               batch_first=True,
                               bidirectional=True)
        self.c_reduce = nn.Linear(self.sen_hidden_dim * 2, self.sen_hidden_dim)
        self.h_reduce = nn.Linear(self.sen_hidden_dim * 2, self.sen_hidden_dim)

        # use GRU as a GATE to update the hidden-state
        # the gate for information which is from children to parents
        self.cell_c_p = nn.GRUCell(self.sen_hidden_dim * 2,
                                   self.sen_hidden_dim * 2)
        # the gate for information which is from parents to children
        self.cell_p_c = nn.GRUCell(self.sen_hidden_dim * 2,
                                   self.sen_hidden_dim * 2)
        # use GRU as a GATE to update the same user's utterance (which is called user link)
        self.cell_user_c_p = nn.GRUCell(self.sen_hidden_dim * 2,
                                        self.sen_hidden_dim * 2)
        self.cell_user_p_c = nn.GRUCell(self.sen_hidden_dim * 2,
                                        self.sen_hidden_dim * 2)

        self.reduce = nn.Linear(self.sen_hidden_dim * 2, self.sen_hidden_dim)

        if self.config['graph_structure_net']['positional_enc']:
            self.decoder = AttentionDecoder(
                self.emb_dim, self.sen_hidden_dim, self.sen_hidden_dim * 2 +
                self.config['graph_structure_net']['positional_enc_dim'],
                self.sen_hidden_dim, self.dropout)
        else:
            self.decoder = AttentionDecoder(self.emb_dim, self.sen_hidden_dim,
                                            self.sen_hidden_dim * 2,
                                            self.sen_hidden_dim, self.dropout)
        self.output_reduce = nn.Linear(self.sen_hidden_dim, vsize)
        self.zero_emb = torch.zeros(1, self.sen_hidden_dim * 2)

    def _get_position_encoding(self,
                               length,
                               hidden_size,
                               min_timescale=1.0,
                               max_timescale=1.0e4):
        """Add the positional encoding.

        Args:
            length ([type]): [description]
            hidden_size ([type]): [description]
            min_timescale (float, optional): [description]. Defaults to 1.0.
            max_timescale ([type], optional): [description]. Defaults to 1.0e4.

        Returns:
            [type]: [description]
        """
        position = torch.arange(0, length, dtype=torch.float)
        num_timescales = hidden_size // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            (torch.arange(0, num_timescales, dtype=torch.float)) *
            -log_timescale_increment)

        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
            inv_timescales, 0)
        signal = torch.cat([torch.sin(scaled_time),
                            torch.cos(scaled_time)],
                           dim=1)
        return signal

    def forward(self, batch):
        # embedding for encoder-decoder framework
        # embed the input variable
        emb_enc_inputs = self.embeddings(batch.enc_batch)
        emb_dec_inputs = self.embeddings(batch.dec_batch)

        emb_enc_inputs = emb_enc_inputs.view(
            self.branch_batch_size * self.sen_batch_size, self.max_enc_steps,
            self.emb_dim)

        # encode all sentences in the dialogue session
        # reproduced after https://discuss.pytorch.org/t/bidirectional-gru-layer-dropout-behavior/17683
        (encoder_outputs, (enc_states_h,
                           enc_states_c)) = self.encoder(emb_enc_inputs)

        encoder_outputs = encoder_outputs.view(
            self.branch_batch_size * self.sen_batch_size, self.max_enc_steps,
            2, self.sen_hidden_dim)
        encoder_outputs_fw = F.dropout(encoder_outputs[:, :, 0, :],
                                       p=self.dropout)
        encoder_outputs_bw = F.dropout(encoder_outputs[:, :, 1, :],
                                       p=self.dropout)
        encoder_outputs = torch.cat(
            (encoder_outputs_fw, encoder_outputs_bw),
            dim=2).view(self.branch_batch_size * self.sen_batch_size,
                        self.max_enc_steps, 2 * self.sen_hidden_dim)

        # position embedding, learning from transformer
        if self.config['graph_structure_net']['positional_enc']:
            max_length = encoder_outputs.shape[1]
            # positional_encoding [max_length, config.postiional_enc_dim]
            positional_encoding = self._get_position_encoding(
                max_length,
                self.config['graph_structure_net']['positional_enc_dim'])
            t = torch.unsqueeze(positional_encoding, 0)
            encoder_outputs = torch.cat([
                t.repeat(self.branch_batch_size * self.sen_batch_size, 1, 1),
                encoder_outputs
            ], -1)

        sen_enc_states = encoder_outputs

        enc_states_c = torch.cat(
            (enc_states_c[0, :, :], enc_states_c[1, :, :]), dim=1)
        enc_states_h = torch.cat(
            (enc_states_h[0, :, :], enc_states_h[1, :, :]), dim=1)
        enc_states_c = F.relu(self.c_reduce(enc_states_c))
        enc_states_h = F.relu(self.h_reduce(enc_states_h))

        # concat the c and h of the LSTM
        enc_states = torch.cat((enc_states_c, enc_states_h), 1)

        # concat a zero embedding at the frist dimension in the enc_states
        # because we want use it as the variable of all padding sentences
        # when we use the GSN computing method.
        hidden_state_list = torch.cat([self.zero_emb, enc_states], 0)

        struct_child = torch.matmul(batch.state_matrix,
                                    batch.struct_conv).type(torch.LongTensor)
        struct_parent = torch.matmul(batch.struct_conv,
                                     batch.state_matrix).type(torch.LongTensor)

        relate_user_child = torch.matmul(
            batch.relate_user, batch.struct_conv).type(torch.LongTensor)
        relate_user_parent = torch.matmul(
            batch.struct_conv, batch.relate_user).type(torch.LongTensor)

        dict_norm_sent = []
        # transfer the information from children to parents
        for _ in range(self.config['graph_structure_net']['n_gram']):
            emb_enc_p = F.embedding(struct_parent, hidden_state_list)
            emb_enc_c = F.embedding(struct_child, hidden_state_list)

            emb_enc_p = emb_enc_p.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)
            emb_enc_c = emb_enc_c.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)

            enc_p_change = self.cell_c_p(emb_enc_c, emb_enc_p)
            enc_p_change = F.dropout(enc_p_change, p=self.dropout)

            enc_p_change = enc_p_change.view(self.branch_batch_size,
                                             self.sen_batch_size,
                                             self.sen_batch_size,
                                             self.sen_hidden_dim * 2)

            enc_p_change = enc_p_change * batch.mask_emb
            enc_p_change = torch.sum(enc_p_change, 1)
            enc_p_change = enc_p_change.view(
                self.branch_batch_size * self.sen_batch_size,
                self.sen_hidden_dim * 2)

            enc_p_change = torch.cat([self.zero_emb, enc_p_change], 0)

            # use the norm to control the information fusion
            if self.config['graph_structure_net']['use_norm']:
                vlid_norm_sent = torch.norm(enc_p_change, dim=1)
                vild_norm_sent = torch.pow(vlid_norm_sent, 2)
                dict_norm_sent = torch.div(
                    vlid_norm_sent +
                    self.config['graph_structure_net']['norm_alpha'],
                    vlid_norm_sent + 1)
                dict_norm_sent = dict_norm_sent.view(
                    self.branch_batch_size * self.sen_batch_size + 1, 1)
                dict_norm_sent = torch.detach(dict_norm_sent)

            if not self.config['graph_structure_net'][
                    'user_struct'] and self.config['graph_structure_net'][
                        'use_norm']:
                hidden_state_list += enc_p_change * dict_norm_sent
            elif not self.config['graph_structure_net'][
                    'user_struct'] and not self.config['graph_structure_net'][
                        'use_norm']:
                hidden_state_list += enc_p_change

            # to update the relate info
            if self.config['graph_structure_net']['user_struct']:
                emb_relate_p = F.embedding(relate_user_parent,
                                           hidden_state_list)
                emb_relate_c = F.embedding(relate_user_child,
                                           hidden_state_list)
                emb_relate_p = emb_relate_p.view(
                    self.branch_batch_size * self.sen_batch_size *
                    self.sen_batch_size, self.sen_hidden_dim * 2)
                emb_relate_c = emb_relate_c.view(
                    self.branch_batch_size * self.sen_batch_size *
                    self.sen_batch_size, self.sen_hidden_dim * 2)

                enc_user_p_change = self.cell_user_c_p(emb_relate_c,
                                                       emb_relate_p)
                enc_user_p_change = F.dropout(enc_user_p_change,
                                              p=self.dropout)

                enc_user_p_change = enc_user_p_change.view(
                    self.branch_batch_size, self.sen_batch_size,
                    self.sen_batch_size, self.sen_hidden_dim * 2)
                enc_user_p_change = enc_user_p_change * batch.mask_user

                enc_user_p_change = torch.sum(enc_user_p_change, 1)
                enc_user_p_change = enc_user_p_change.view(
                    self.branch_batch_size * self.sen_batch_size,
                    self.sen_hidden_dim * 2)
                enc_user_p_change = torch.cat(
                    [self.zero_emb, enc_user_p_change], 0)

                if self.config['graph_structure_net']['use_norm']:
                    vlid_norm_user = torch.norm(enc_user_p_change, dim=1)
                    vild_norm_user = torch.pow(vlid_norm_user, 2)
                    dict_norm_user = torch.div(
                        vlid_norm_user +
                        self.config['graph_structure_net']['norm_alpha'],
                        vlid_norm_user + 1)
                    dict_norm_user = dict_norm_user.view(
                        self.branch_batch_size * self.sen_batch_size + 1, 1)
                    dict_norm_user = torch.detach(dict_norm_user)

                    hidden_state_list = hidden_state_list + enc_user_p_change * dict_norm_user + enc_p_change * dict_norm_sent
                else:
                    hidden_state_list = hidden_state_list + enc_user_p_change + enc_p_change

        # transfer the information from parents to children
        for _ in range(self.config['graph_structure_net']['n_gram']):
            emb_enc_p = F.embedding(struct_parent, hidden_state_list)
            emb_enc_c = F.embedding(struct_child, hidden_state_list)

            emb_enc_p = emb_enc_p.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)
            emb_enc_c = emb_enc_c.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)

            enc_c_change = self.cell_p_c(emb_enc_p, emb_enc_c)
            enc_c_change = F.dropout(enc_c_change, p=self.dropout)
            enc_c_change = enc_c_change.view(self.branch_batch_size,
                                             self.sen_batch_size,
                                             self.sen_batch_size,
                                             self.sen_hidden_dim * 2)

            enc_c_change = enc_c_change * batch.mask_emb

            enc_c_change = torch.sum(enc_c_change, 2)
            enc_c_change = enc_c_change.view(
                self.branch_batch_size * self.sen_batch_size,
                self.sen_hidden_dim * 2)
            enc_c_change = torch.cat([self.zero_emb, enc_c_change], 0)

            if self.config['graph_structure_net']['use_norm']:
                vlid_norm_sent = torch.norm(enc_c_change, dim=1)
                vild_norm_sent = torch.pow(vlid_norm_sent, 2)
                dict_norm_sent = torch.div(
                    vlid_norm_sent +
                    self.config['graph_structure_net']['norm_alpha'],
                    vlid_norm_sent + 1)
                dict_norm_sent = dict_norm_sent.view(
                    self.branch_batch_size * self.sen_batch_size + 1, 1)
                dict_norm_sent = torch.detach(dict_norm_sent)

            if not self.config['graph_structure_net'][
                    'user_struct'] and self.config['graph_structure_net'][
                        'use_norm']:
                hidden_state_list += enc_c_change * dict_norm_sent
            elif not self.config['graph_structure_net'][
                    'user_struct'] and not self.config['graph_structure_net'][
                        'use_norm']:
                hidden_state_list += enc_c_change

            # to update the relate info
            emb_relate_p = F.embedding(relate_user_parent, hidden_state_list)
            emb_relate_c = F.embedding(relate_user_child, hidden_state_list)

            emb_relate_p = emb_relate_p.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)
            emb_relate_c = emb_relate_c.view(
                self.branch_batch_size * self.sen_batch_size *
                self.sen_batch_size, self.sen_hidden_dim * 2)

            enc_user_c_change = self.cell_user_p_c(emb_relate_p, emb_relate_c)
            enc_user_c_change = F.dropout(enc_user_c_change, p=self.dropout)

            enc_user_c_change = enc_user_c_change.view(self.branch_batch_size,
                                                       self.sen_batch_size,
                                                       self.sen_batch_size,
                                                       self.sen_hidden_dim * 2)
            enc_user_c_change = enc_user_c_change * batch.mask_user

            enc_user_c_change = torch.sum(enc_user_c_change, 2)
            enc_user_c_change = enc_user_c_change.view(
                self.branch_batch_size * self.sen_batch_size,
                self.sen_hidden_dim * 2)
            enc_user_c_change = torch.cat([self.zero_emb, enc_user_c_change],
                                          0)

            if self.config['graph_structure_net']['use_norm']:
                vlid_norm_user = torch.norm(enc_user_c_change, dim=1)
                vild_norm_user = torch.pow(vlid_norm_user, 2)
                dict_norm_user = torch.div(
                    vlid_norm_user +
                    self.config['graph_structure_net']['norm_alpha'],
                    vlid_norm_user + 1)
                dict_norm_user = dict_norm_user.view(
                    self.branch_batch_size * self.sen_batch_size + 1, 1)
                dict_norm_user = torch.detach(dict_norm_user)
                hidden_state_list = hidden_state_list + enc_user_c_change * dict_norm_user + enc_c_change * dict_norm_sent
            else:
                hidden_state_list = hidden_state_list + enc_user_c_change + enc_c_change

        new_hidden_state = F.relu(self.reduce(hidden_state_list))

        # get variable for decoder
        dec_hidden_state_init = new_hidden_state[batch.tgt_index + 1]
        if not self.config['graph_structure_net']['long_attn']:
            enc_state = sen_enc_states[batch.tgt_index]
            attn_mask = batch.attn_mask.view(-1, self.max_enc_steps)
            attn_mask = attn_mask[batch.tgt_index]
        else:
            enc_state = sen_enc_states.view(
                self.branch_batch_size,
                self.sen_batch_size * self.max_enc_steps, -1)
            attn_mask = batch.attn_mask.view(
                self.branch_batch_size,
                self.sen_batch_size * self.max_enc_steps)

        dec_state = dec_hidden_state_init
        sen_enc_states = enc_state

        dec_out, dec_out_state, attn_dists = self.decoder(
            emb_dec_inputs, dec_hidden_state_init, enc_state, attn_mask)
        # input_size, hidden_size, attn_size, output_size, dropout
        # sen_enc_states, cell, config.mode=="decode")

        vocab_scores = []
        for i, output in enumerate(dec_out):
            vocab_scores.append(self.output_reduce(output))
        vocab_dists = [F.softmax(s) for s in vocab_scores]

        return vocab_dists
