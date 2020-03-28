import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import params


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        print("Use attention type %s" % method)
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size,
                                    max_len)  # B x S

        if params.use_cuda and torch.cuda.is_available():
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[b, :],
                                                 encoder_outputs[b, i])

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = torch.dot(encoder_output, hidden)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(energy, hidden)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 0))
            energy = self.v.dot(energy)
            return energy