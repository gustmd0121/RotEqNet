# Global imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class VectorBatchNorm(nn.Module):
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
            # Compute std
            u = input[0]
            v = input[1]
            p = torch.sqrt(u ** 2 + v ** 2)
            std = torch.std(p)

            alpha = self.weight / (std + self.eps)

            # update running variance
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2
            # compute output
            u = input[0] * alpha
            v = input[1] * alpha

        else:
            alpha = self.weight.data / torch.sqrt(self.running_var + self.eps)

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)
        return u, v
