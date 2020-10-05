import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn import init


class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_sizes,
                 has_bias=True,
                 dropout_rate=0.5,
                 activate_final=False):
        super(MLP, self).__init__()

        dims = [input_size
               ] + [output_sizes[i] for i in range(len(output_sizes))]

        self._has_bias = has_bias
        self._activate_final = activate_final
        self._dropout_rate = dropout_rate
        if dropout_rate not in (None, 0):
            self._dropout = nn.Dropout(dropout_rate)

        self._linear = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=has_bias)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        for _, layer in enumerate(self._linear):
            x = layer(x)
        if self._dropout_rate not in (None, 0):
            x = self._dropout(x)
        if self._activate_final:
            x = nn.ReLU(x)
        return x
