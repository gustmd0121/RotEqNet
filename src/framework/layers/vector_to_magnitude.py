# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from pprint import pprint
import numpy as np

class VectorToMagnitude(nn.Module):
    def __init__(self, tresh):
        super(VectorToMagnitude, self).__init__()
        self.tresh = tresh

    def forward(self, input):
        u = input[0]
        v = input[1]
        p = torch.sqrt(v ** 2 + u ** 2)
        pmax = p/(torch.max(p)+1e-5)

        angle = torch.atan2(u, v) + math.pi

        return p, angle #*torch.clamp(F.threshold(pmax, self.tresh, 0), 0, 1)
