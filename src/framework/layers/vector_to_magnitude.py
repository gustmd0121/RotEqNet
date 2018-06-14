# Global imports
import torch
import torch.nn as nn


class VectorToMagnitude(nn.Module):
    def __init__(self):
        super(VectorToMagnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.sqrt(v ** 2 + u ** 2)
        return u