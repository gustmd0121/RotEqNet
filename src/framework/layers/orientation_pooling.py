# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class OrientationPooling(nn.Module):
    """
    Pool rotated filters.
    Select rotated filter with highest magnitude.
    """

    def __init__(self, activation=F.relu):
        super(OrientationPooling, self).__init__()

        self.activation = activation

    def forward(self, input):
        rotconv_outputs = input[0]
        angles = input[1]
        # Get the maximum direction
        strength, max_ind = torch.max(torch.cat(rotconv_outputs, -1), -1)

        # Convert from polar representation
        angle_map = max_ind.float() * (360. / len(angles) / 180. * math.pi)

        if self.activation:
            u = self.activation(strength) * torch.cos(angle_map)
            v = self.activation(strength) * torch.sin(angle_map)
        else:
            u = strength * torch.cos(angle_map)
            v = strength * torch.sin(angle_map)

        return u, v
