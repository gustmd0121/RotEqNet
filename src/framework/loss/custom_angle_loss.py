# Global imports
import torch
import torch.nn as nn


class Angle_Loss(nn.Module):
    """ Computes average L1 Loss only where beetle is in ground truth data. """

    def __init__(self):
        super(Angle_Loss, self).__init__()

    def forward(self, input, target):
        tmp = torch.clamp(target, 0, 1)

        # When beetles orientation in ground truth data is 0 degrees, you can't know where the beetle is located
        if torch.equal(tmp, torch.zeros_like(tmp)):
            output = torch.sum(input) / torch.numel(input)
        else:
            nonZero = torch.nonzero(target)
            output = torch.sum(torch.abs(target - tmp * input)) / torch.numel(nonZero)
        return output
