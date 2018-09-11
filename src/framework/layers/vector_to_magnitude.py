# Global imports
import torch
import torch.nn as nn
import math

class VectorToMagnitude(nn.Module):
    """ Returns magnitude and angle map (polar coordinates) """

    def __init__(self):
        super(VectorToMagnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        # Compute magnitude
        p = torch.sqrt(v ** 2 + u ** 2)

        # Compute angles in range [0, 2*PI]
        angle = torch.atan2(u, v) + math.pi

        return p, angle
