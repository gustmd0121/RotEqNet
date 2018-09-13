# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class VectorToMagnitude(nn.Module):
    """ Returns magnitude and angle map (polar coordinates) """

    def __init__(self, tresh):
        super(VectorToMagnitude, self).__init__()
        self.tresh = tresh

    def forward(self, input):
        u = input[0]
        v = input[1]
        p = torch.sqrt(v ** 2 + u ** 2)

        angle = torch.atan2(u, v) + math.pi

        return p, angle*torch.clamp(F.threshold(p, self.tresh, 0), 0, 1)
