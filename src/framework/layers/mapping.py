# Global imports
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Mapping(nn.Module):
    def __init__(self, input_channels=1, output_channels=21, kernel_size=1):
        super(Mapping, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def forward(self, input):
        u = input[0]
        v = input[1]

        weights_u = torch.randn(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        weights_v = torch.randn(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)


        start = 1/self.output_channels
        end = 2*math.pi
        step = end/self.output_channels

        for i in range(0, self.output_channels):
            weights_u[i, :, :, :] = 0.01*torch.sin(torch.tensor(start + i*step))
            weights_v[i, :, :, :] = 0.01*torch.cos(torch.tensor(start + i*step))

        weights_u = weights_u.cuda()
        weights_v = weights_v.cuda()

        u_out = F.conv2d(u, weights_u, stride=[1], padding=[self.kernel_size // 2], dilation=[1])
        v_out = F.conv2d(v, weights_v, stride=[1], padding=[self.kernel_size // 2], dilation=[1])


        return u_out, v_out
