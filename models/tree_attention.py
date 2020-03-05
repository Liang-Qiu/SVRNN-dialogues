import torch
from torch_struct import DependencyCRF, LinearChainCRF
import matplotlib.pyplot as plt


def show(x):
    plt.imshow(x.detach())


# Make some data.
vals = torch.zeros(2, 10, 10) + 1e-5
vals[:, :5, :5] = torch.rand(5)
vals[:, 5:, 5:] = torch.rand(5)
print(vals[:, 5:, 5:])
dist = DependencyCRF(vals.log())
show(dist.log_potentials[0])

# Compute marginals
show(dist.marginals[0])

plt.show()