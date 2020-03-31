import sys
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("..")
import params


def BPR_BOW_loss(output_tokens,
                 dec_outs_1,
                 dec_outs_2,
                 log_p_z,
                 log_q_z,
                 p_z,
                 q_z,
                 bow_logits1=None,
                 bow_logits2=None):
    dec_outs_1 = dec_outs_1.view(-1, params.max_vocab_cnt)
    labels_1 = output_tokens[0][:, 1:].reshape(-1)
    label_mask_1 = torch.sign(labels_1)
    dec_outs_2 = dec_outs_2.view(-1, params.max_vocab_cnt)
    labels_2 = output_tokens[1][:, 1:].reshape(-1)
    label_mask_2 = torch.sign(labels_2)

    if params.word_weights is not None:
        weights = torch.tensor(params.word_weights, requires_grad=False)
        if params.use_cuda and torch.cuda.is_available():
            weights = weights.cuda()
        rc_loss1 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
            dec_outs_1, labels_1) * label_mask_1.float()
        rc_loss2 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
            dec_outs_2, labels_2) * label_mask_2.float()
    else:
        rc_loss1 = nn.CrossEntropyLoss(reduction='none')(
            dec_outs_1, labels_1) * label_mask_1.float()
        rc_loss2 = nn.CrossEntropyLoss(reduction='none')(
            dec_outs_2, labels_2) * label_mask_2.float()
    rc_loss_1 = torch.sum(rc_loss1)
    rc_loss_2 = torch.sum(rc_loss2)

    # KL_loss
    kl_loss = (log_q_z - log_p_z) * q_z
    kl_loss = torch.sum(kl_loss)

    if params.with_BPR:
        q_z_prime = torch.mean(q_z, dim=0)
        log_q_z_prime = torch.log(q_z_prime + 1e-20)

        p_z_prime = torch.mean(p_z, dim=0)
        log_p_z_prime = torch.log(p_z_prime + 1e-20)

        kl_loss = (log_q_z_prime - log_p_z_prime) * q_z_prime
        # TODO: BPR?
        kl_loss = params.kl_loss_weight * torch.sum(kl_loss)
        # kl_loss = torch.div(torch.sum(kl_loss), params.batch_size)

    elbo_t = rc_loss_1 + rc_loss_2 + kl_loss

    # BOW_loss
    bow_loss_1 = bow_loss_2 = 0
    if params.with_BOW:
        tile_bow_logits1 = (torch.unsqueeze(
            bow_logits1, 1).repeat(1, params.max_utt_len - 1, 1)).view(
                -1,
                params.max_vocab_cnt)  # [batch * (max_utt - 1), vocab_size]
        tile_bow_logits2 = (torch.unsqueeze(bow_logits2, 1).repeat(
            1, params.max_utt_len - 1, 1)).view(-1, params.max_vocab_cnt)

        if params.word_weights is not None:
            bow_loss1 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
                tile_bow_logits1, labels_1) * label_mask_1.float()
            bow_loss2 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
                tile_bow_logits2, labels_2) * label_mask_2.float()
        else:
            bow_loss1 = nn.CrossEntropyLoss(reduction='none')(
                tile_bow_logits1, labels_1) * label_mask_1.float()
            bow_loss2 = nn.CrossEntropyLoss(reduction='none')(
                tile_bow_logits2, labels_2) * label_mask_2.float()

        bow_loss_1 = params.bow_loss_weight * torch.sum(bow_loss1)
        bow_loss_2 = params.bow_loss_weight * torch.sum(bow_loss2)

        elbo_t = elbo_t + bow_loss_1 + bow_loss_2

    return elbo_t, rc_loss_1 + rc_loss_2, kl_loss, bow_loss_1 + bow_loss_2


def BPR_BOW_loss_single(output_tokens,
                        dec_outs,
                        dec_mask,
                        log_p_z,
                        log_q_z,
                        p_z,
                        q_z,
                        bow_logits=None):
    dec_outs = dec_outs.view(-1, params.max_vocab_cnt)
    labels = output_tokens.long().reshape(-1)
    label_mask = dec_mask.float().reshape(-1)

    if params.word_weights is not None:
        weights = torch.tensor(params.word_weights, requires_grad=False)
        if params.use_cuda and torch.cuda.is_available():
            weights = weights.cuda()
        rc_loss = nn.CrossEntropyLoss(weight=weights, reduction='none')(
            dec_outs, labels) * label_mask

    else:
        rc_loss = nn.CrossEntropyLoss(reduction='none')(dec_outs,
                                                        labels) * label_mask

    rc_loss = torch.sum(rc_loss)

    # KL_loss
    kl_loss = (log_q_z - log_p_z) * q_z
    kl_loss = torch.sum(kl_loss)

    if params.with_BPR:
        q_z_prime = torch.mean(q_z, dim=0)
        log_q_z_prime = torch.log(q_z_prime + 1e-20)

        p_z_prime = torch.mean(p_z, dim=0)
        log_p_z_prime = torch.log(p_z_prime + 1e-20)

        kl_loss = (log_q_z_prime - log_p_z_prime) * q_z_prime
        # TODO: BPR?
        kl_loss = params.kl_loss_weight * torch.sum(kl_loss)
        # kl_loss = torch.div(torch.sum(kl_loss), params.batch_size)

    elbo_t = rc_loss + kl_loss

    # BOW_loss
    bow_loss = 0
    if params.with_BOW:
        tile_bow_logits = (torch.unsqueeze(
            bow_logits, 1).repeat(1, params.max_dec_steps, 1)).view(
                -1,
                params.max_vocab_cnt)  # [batch * (max_utt - 1), vocab_size]

        if params.word_weights is not None:
            bow_loss = nn.CrossEntropyLoss(weight=weights, reduction='none')(
                tile_bow_logits, labels) * label_mask
        else:
            bow_loss = nn.CrossEntropyLoss(reduction='none')(
                tile_bow_logits, labels) * label_mask

        bow_loss = params.bow_loss_weight * torch.sum(bow_loss)

        elbo_t = elbo_t + bow_loss

    return elbo_t, rc_loss, kl_loss, bow_loss


def print_loss(prefix, loss_names, losses, postfix):
    template = "%s "
    for name in loss_names:
        template += "%s " % name
        template += " %f "
    template += "%s"
    template = re.sub(' +', ' ', template)
    values = [prefix]

    for loss in losses:
        values.append(torch.mean(torch.stack(loss)))
    values.append(postfix)

    print(template % tuple(values))
