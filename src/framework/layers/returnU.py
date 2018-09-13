# Global imports
import torch
import torch.nn as nn


class ReturnU(nn.Module):
    """ Returns u component of vector field) """

    def __init__(self):
        super(ReturnU, self).__init__()

    def forward(self, input):
        u = input[0]

        return u
