# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F


class SpatialPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False):
        super(SpatialPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        # Magnitude
        p = torch.sqrt(v ** 2 + u ** 2)
        # Max pool
        _, max_inds = F.max_pool2d(p, self.kernel_size, self.stride,
                                   self.padding, self.dilation, self.ceil_mode,
                                   return_indices=True)
        # Reshape to please pytorch
        s1 = u.size()
        s2 = max_inds.size()

        max_inds = max_inds.view(s1[0], s1[1], s2[2] * s2[3])

        u = u.view(s1[0], s1[1], s1[2] * s1[3])
        v = v.view(s1[0], s1[1], s1[2] * s1[3])

        # Select u/v components according to max pool on magnitude
        u = torch.gather(u, 2, max_inds)
        v = torch.gather(v, 2, max_inds)

        # Reshape back
        u = u.view(s1[0], s1[1], s2[2], s2[3])
        v = v.view(s1[0], s1[1], s2[2], s2[3])

        return u, v
