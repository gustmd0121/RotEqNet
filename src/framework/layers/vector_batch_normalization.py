# Global imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from ..utils import *


class VectorBatchNorm(nn.Module):
    """
    Custom Batch Normalization for vector fields.
    Normalize magnitude only.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):

        super(VectorBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()

    def forward(self, input):
        """
        Based on https://github.com/lberrada/bn.pytorch
        """
        if self.training:
            u = input[0]
            v = input[1]

            # Vector to magnitude
            p = torch.sqrt(u ** 2 + v ** 2)

            # We want to normalize the vector magnitudes,
            # therefore we ommit the mean (var = (p-p.mean())**2)
            # since we do not want to move the center of the vectors.

            var = (p) ** 2

            # Compute mean over every dimension, except the feature dimension
            # The result is the variance of the magnitude map
            var = torch.mean(var, 0, keepdim=True)
            var = torch.mean(var, 2, keepdim=True)
            var = torch.mean(var, 3, keepdim=True)
            std = torch.sqrt(var)

            alpha = self.weight / (std + self.eps)

            # update running variance
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)

        else:
            alpha = self.weight.data / torch.sqrt(self.running_var + self.eps)

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)
        return u, v
