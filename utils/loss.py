import torch
import torch.nn.functional as F


def BPR_bow_loss(output_tokens,
                 dec_outs_1,
                 dec_outs_2,
                 log_p_z,
                 log_q_z,
                 with_BPR=True,
                 with_bow=True):
    labels_1 = output_tokens[0][:, 1:]
    label_mask_1 = tf.to_float(tf.sign(labels_1))
    labels_2 = output_tokens[1][:, 1:]
    label_mask_2 = tf.to_float(tf.sign(labels_2))

    # BPR_loss
    rc_loss_1 = F.nll_loss(dec_outs_1, labels_1)
    if self.weights is not None:
        weights = tf.gather(self.weights, labels_1)
        rc_loss_1 = compute_weighted_loss(rc_loss_1, weights=weights)
    rc_loss_1 = tf.reduce_sum(rc_loss_1 * label_mask_1, reduction_indices=1)

    rc_loss_2 = F.nll_loss(dec_outs_2, labels_2)

    if self.weights is not None:
        rc_loss_2 = compute_weighted_loss(rc_loss_2, weights=weights)
    rc_loss_2 = tf.reduce_sum(rc_loss_2 * label_mask_2, reduction_indices=1)

    # KL_loss
    kl_tmp = (log_q_z - log_p_z) * q_z
    kl_tmp = torch.sum(kl_tmp, dim=1)

    if with_BPR:
        q_z_prime = torch.mean(q_z, dim=0)
        log_q_z_prime = torch.log(q_z_prime + 1e-20)  # equation 9

        p_z_prime = torch.mean(p_z, dim=0)
        log_p_z_prime = torch.log(p_z_prime + 1e-20)

        kl_bpr = (log_q_z_prime - log_p_z_prime) * q_z_prime
        kl_bpr = torch.sum(kl_bpr)
        infered_batch_size = tf.shape(inputs)[0]
        kl_bpr = tf.div(kl_bpr, tf.to_float(infered_batch_size))

    if not with_BPR:
        elbo_t = rc_loss_1 + rc_loss_2 + kl_tmp
    else:
        elbo_t = rc_loss_1 + rc_loss_2 + kl_bpr

    # BOW_loss
    if with_bow:
        tile_bow_logits1 = tf.tile(tf.expand_dims(self.bow_logits1, 1),
                                   (1, self.config.max_utt_len - 1, 1))
        bow_loss1 = torch.nll_loss(tile_bow_logits1, labels_1) * label_mask_1
        if self.weights is not None:
            weights = tf.gather(self.weights, labels_1)
            bow_loss1 = compute_weighted_loss(bow_loss1, weights=weights)

        bow_loss1 = torch.sum(bow_loss1, dim=1)

        tile_bow_logits2 = tf.tile(tf.expand_dims(self.bow_logits2, 1),
                                   (1, self.config.max_utt_len - 1, 1))
        bow_loss2 = torch.nll_loss(tile_bow_logits2, labels_2) * label_mask_2
        if self.weights is not None:
            bow_loss2 = compute_weighted_loss(bow_loss2, weights=weights)
        bow_loss2 = torch.sum(bow_loss2, dim=1)

        elbo_t = elbo_t + self.config.bow_loss_weight * (bow_loss1 + bow_loss2)

    elbo_t = tf.expand_dims(elbo_t, 1)
