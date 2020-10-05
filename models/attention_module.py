import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import params
import torch_struct


class Attn(nn.Module):

    def __init__(self, method, query_size, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.query_size = query_size
        self.hidden_size = hidden_size

        print("Use attention type %s" % method)
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.query_size)

        elif self.method == 'concat':
            self.attn_1 = nn.Linear(self.hidden_size, self.query_size)
            self.attn_2 = nn.Linear(self.query_size * 2, self.query_size)
            self.v = nn.Parameter(torch.rand(query_size))

    def forward(self, query, encoder_outputs, tgt_index):
        context = []
        for b in range(encoder_outputs.size(0)):
            this_len = tgt_index[b] + 1
            # Create variable to store attention energies
            attn_energies = torch.zeros(this_len)

            if params.use_cuda and torch.cuda.is_available():
                attn_energies = attn_energies.cuda()

            # Calculate energy for each encoder output
            for i in range(this_len):
                attn_energies[i] = self.score(query[b, :], encoder_outputs[b,
                                                                           i])
            # Normalize energies to weights in range 0 to 1, resize to B x S
            c = F.softmax(attn_energies,
                          dim=0).matmul(encoder_outputs[b, :this_len, :])
            context.append(c)
        return torch.stack(context)

    def score(self, query, encoder_output):

        if self.method == 'dot':
            energy = torch.dot(encoder_output, query)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(energy, query)
            return energy

        elif self.method == 'concat':
            energy = self.attn_2(
                torch.cat((query, self.attn_1(encoder_output)), 0))
            energy = self.v.dot(energy)
            return energy
