# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        weights_u = torch.randn(1,1,5,5)
        weights_u[:][:] = Variable(torch.tensor([[torch.tensor(0), torch.tensor(0), torch.sin(torch.tensor(1*360/21)), torch.sin(torch.tensor(2*360/21)), torch.sin(torch.tensor(3*360/21))],
                                        [torch.sin(torch.tensor(4*360/21)), torch.sin(torch.tensor(5*360/21)), torch.sin(torch.tensor(6*360/21)), torch.sin(torch.tensor(7*360/21)), torch.sin(torch.tensor(8*360/21))],
                                        [torch.sin(torch.tensor(9*360/21)), torch.sin(torch.tensor(10*360/21)), torch.sin(torch.tensor(11*360/21)), torch.sin(torch.tensor(12*360/21)), torch.sin(torch.tensor(13*360/21))],
                                        [torch.sin(torch.tensor(14*360/21)), torch.sin(torch.tensor(15*360/21)), torch.sin(torch.tensor(16*360/21)), torch.sin(torch.tensor(17*360/21)), torch.sin(torch.tensor(18*360/21))],
                                        [torch.sin(torch.tensor(19*360/21)), torch.sin(torch.tensor(20*360/21)), torch.sin(torch.tensor(21*360/21)), torch.tensor(0), torch.tensor(0)]]))

        weights_v = torch.randn(1,1,5,5)
        weights_v[:][:] = Variable(torch.tensor([[torch.tensor(0), torch.tensor(0), torch.cos(torch.tensor(1*360/21)), torch.cos(torch.tensor(2*360/21)), torch.cos(torch.tensor(3*360/21))],
                                        [torch.cos(torch.tensor(4*360/21)), torch.cos(torch.tensor(5*360/21)), torch.cos(torch.tensor(6*360/21)), torch.cos(torch.tensor(7*360/21)), torch.cos(torch.tensor(8*360/21))],
                                        [torch.cos(torch.tensor(9*360/21)), torch.cos(torch.tensor(10*360/21)), torch.cos(torch.tensor(11*360/21)), torch.cos(torch.tensor(12*360/21)), torch.cos(torch.tensor(13*360/21))],
                                        [torch.cos(torch.tensor(14*360/21)), torch.cos(torch.tensor(15*360/21)), torch.cos(torch.tensor(16*360/21)), torch.cos(torch.tensor(17*360/21)), torch.cos(torch.tensor(18*360/21))],
                                        [torch.cos(torch.tensor(19*360/21)), torch.cos(torch.tensor(20*360/21)), torch.cos(torch.tensor(21*360/21)), torch.tensor(0), torch.tensor(0)]]))

        weights_u = weights_u.cuda()
        weights_v = weights_v.cuda()

        u_out = F.conv2d(u, weights_u, None, 1, 5 // 2)
        v_out = F.conv2d(v, weights_v, None, 1, 5 // 2)

        return u_out, v_out
