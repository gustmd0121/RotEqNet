# Global imports
import torch
import torch.nn as nn
import math

class VectorToMagnitude(nn.Module):
    def __init__(self):
        super(VectorToMagnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        angle = torch.atan(torch.abs(u / (v + 1e-8)))

        # new

        # p = torch.sqrt(v ** 2 + u ** 2)
        # p = torch.sign(torch.sign(u) + torch.sign(v) + 0.1) * p

        # sign_v = torch.sign(torch.sign(v) + 0.5)
        # angle = math.pi - math.pi * sign_v + sign_v * torch.acos(u/(p + 1e-8))
        # print(angle)

        return u+v , angle
