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

    #print("loss.py",output_tokens[0].shape)
    #exit()

    label_mask_1 = torch.sign(labels_1)
    dec_outs_2 = dec_outs_2.view(-1, params.max_vocab_cnt)
    labels_2 = output_tokens[1][:, 1:].reshape(-1)
    label_mask_2 = torch.sign(labels_2)

    if params.word_weights is not None:
        weights = torch.tensor(params.word_weights, requires_grad=False)
        rc_loss_1 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
            dec_outs_1, labels_1) #* label_mask_1
        rc_loss_2 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
            dec_outs_2, labels_2) #* label_mask_2
    else:
        rc_loss_1 = nn.CrossEntropyLoss(reduction='none')(
            dec_outs_1, labels_1) #* label_mask_1
        rc_loss_1 = rc_loss_1 * label_mask_1.float()
        rc_loss_2 = nn.CrossEntropyLoss(reduction='none')(
            dec_outs_2, labels_2) #* label_mask_2
        rc_loss_2 = rc_loss_2 * label_mask_2.float()
        #print(rc_loss_1.shape)
        #print(label_mask_1.shape)
        #exit()

    # KL_loss
    kl_tmp = (log_q_z - log_p_z) * q_z
    kl_tmp =  1000 * torch.sum(kl_tmp)

    if params.with_BPR:
        q_z_prime = torch.mean(q_z, dim=0)
        log_q_z_prime = torch.log(q_z_prime + 1e-20)

        p_z_prime = torch.mean(p_z, dim=0)
        log_p_z_prime = torch.log(p_z_prime + 1e-20)

        kl_bpr = (log_q_z_prime - log_p_z_prime) * q_z_prime
        kl_bpr = 1000 * torch.div(torch.sum(kl_bpr), params.batch_size)

    if not params.with_BPR:
        elbo_t = torch.sum(rc_loss_1) + torch.sum(rc_loss_2) + kl_tmp
    else:
        elbo_t = torch.sum(rc_loss_1) + torch.sum(rc_loss_2) + kl_bpr

    # BOW_loss
    if params.with_BOW:
        tile_bow_logits1 = (torch.unsqueeze(
            bow_logits1, 1).repeat(1, params.max_utt_len - 1, 1)).view(
                -1,
                params.max_vocab_cnt)  # [batch * (max_utt - 1), vocab_size]
        tile_bow_logits2 = (torch.unsqueeze(bow_logits2, 1).repeat(
            1, params.max_utt_len - 1, 1)).view(-1, params.max_vocab_cnt)

        if params.word_weights is not None:
            weights = torch.tensor(params.word_weights, requires_grad=False)
            bow_loss1 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
                tile_bow_logits1, labels_1) * label_mask_1
            bow_loss2 = nn.CrossEntropyLoss(weight=weights, reduction='none')(
                tile_bow_logits2, labels_2) * label_mask_2
        else:
            bow_loss1 = nn.CrossEntropyLoss(reduction='none')(
                tile_bow_logits1, labels_1) #* label_mask_1
            bow_loss2 = nn.CrossEntropyLoss(reduction='none')(
                tile_bow_logits2, labels_2) #* label_mask_2
        elbo_t = elbo_t + params.bow_loss_weight * (torch.sum(bow_loss1) +
                                                    torch.sum(bow_loss2))
    # elbo_t = torch.unsqueeze(elbo_t, 1)

    return elbo_t


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
