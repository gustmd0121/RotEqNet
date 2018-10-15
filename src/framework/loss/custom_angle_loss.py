# Global imports
import torch
import torch.nn as nn

class Angle_Loss(nn.Module):

    def __init__(self):
        super(Angle_Loss, self).__init__()

    def forward(self, input, target):
        tmp = torch.clamp(target, 0, 1)
        if torch.equal(tmp, torch.zeros_like(tmp)):
            output = torch.sum(input) / torch.numel(input)
        else:
            nonZero = torch.nonzero(target)
            output = torch.sum(torch.abs(target-tmp*input)) / torch.numel(nonZero)
        return output
