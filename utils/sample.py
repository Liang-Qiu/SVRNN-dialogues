from __future__ import division

import torch
import torch.nn.functional as F
import params


# Thanks for the implementation at https://github.com/dev4488/VAE_gumble_softmax/blob/master/vae_gumbel_softmax.py
# Note: PyTorch also has this in their official API now.
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, requires_grad=True).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    device = "cuda" if params.use_cuda and torch.cuda.is_available() else "cpu"
    y = logits + sample_gumbel(logits.size(), 1e-20, device)
    return F.softmax(y / temperature, dim=1), y / temperature


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y, logits = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        if params.use_cuda and torch.cuda.is_available():
            y_hard = y_hard.cuda()
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y, logits
