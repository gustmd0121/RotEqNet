# Global imports
import torch.nn as nn
from torch.nn import functional as F


class VectorUpsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(VectorUpsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        u = F.upsample(u, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        v = F.upsample(v, size=self.size, scale_factor=self.scale_factor, mode=self.mode)

        return u, v
