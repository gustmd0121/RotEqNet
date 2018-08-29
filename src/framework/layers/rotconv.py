# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from pprint import pprint

# Local imports
from ..utils import *


class RotConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, n_angles=8, mode=1):
        super(RotConv, self).__init__()

        kernel_size = ntuple(2)(kernel_size)
        stride = ntuple(2)(stride)
        padding = ntuple(2)(padding)
        dilation = ntuple(2)(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.mode = mode

        # Angles
        self.angles = np.linspace(0, 360, n_angles, endpoint=False)
        self.angle_tensors = []

        # Get interpolation variables
        self.interp_vars = []
        for angle in self.angles:
            out = get_filter_rotation_transforms(list(self.kernel_size), angle)
            self.interp_vars.append(out[:-1])
            self.mask = out[-1]

            self.angle_tensors.append(Variable(torch.FloatTensor(np.array([angle / 180. * np.pi]))))

        if(self.mode==2):
            self.weight1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            # If input is vector field, we have two filters (one for each component)
            # if self.mode == 2:
            self.weight2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        else:
            self.weight1 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight2 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.mode == 2:
            self.weight2.data.uniform_(-stdv, stdv)

    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        self.mask = func(self.mask)
        self.interp_vars = [[[func(el2) for el2 in el1] for el1 in el0] for el0 in self.interp_vars]
        self.angle_tensors = [func(el) for el in self.angle_tensors]

        return super(RotConv, self)._apply(func)

    def forward(self, input):
        outputs = []
        if self.mode == 1:
            input.unsqueeze_(1)
            outputs = []

            # Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                # Apply rotation
                weight = apply_transform(self.weight1, interp_vars, self.kernel_size)

                # Do convolution
                out = F.conv2d(input, weight, None, self.stride, self.padding, self.dilation)
                outputs.append(out.unsqueeze(-1))

        elif self.mode == 2:
            u = input[0]
            v = input[1]
            # Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                angle = self.angle_tensors[ind]
                # Apply rotation
                wu = apply_transform(self.weight1, interp_vars, self.kernel_size)
                wv = apply_transform(self.weight2, interp_vars, self.kernel_size)

                # Do convolution for u
                from pprint import pprint
                #pprint(wu.shape)
                #pprint(angle)
                #pprint(torch.cos(angle))
                #pprint(torch.cos(angle))
                #pprint(angle)
                wru = torch.cos(angle) * wu - torch.sin(angle) * wv
                u_out = F.conv2d(u, wru, None, self.stride, self.padding, self.dilation)

                # Do convolution for v
                wrv = torch.sin(angle) * wu + torch.cos(angle) * wv
                v_out = F.conv2d(v, wrv, None, self.stride, self.padding, self.dilation)

                p = u_out + v_out
                p = F.relu(p)
                outputs.append((p).unsqueeze(-1))

        # print("rotconv:", len(outputs), len(outputs[0]), len(outputs[0][0]), len(outputs[0][0][0]), len(outputs[0][0][0][0]))
        return outputs, self.angles
