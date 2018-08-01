# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
from pprint import pprint
import numpy as np

class VectorToMagnitude(nn.Module):
    def __init__(self):
        super(VectorToMagnitude, self).__init__()

    def forward(self, input):
        ps = torch.cat(input[0], -1).squeeze()
        angle_tensors = list()
        for angle in input[1]:
            angle_tensors.append(Variable(torch.FloatTensor(np.array([angle / 180. * np.pi]))))
        angles = torch.cat(angle_tensors, -1).squeeze()
        angles = angles.cuda()
        u = F.relu(ps) * torch.cos(angles)
        v = F.relu(ps) * torch.sin(angles)
        ps = torch.sqrt(u ** 2 + v ** 2)
        # for i in range(len(input[0])):
        #     angle_map = i * (360. / len(input[1]) / 180. * math.pi)
        #     u = ps * torch.cos(angle_map)
        #     v = ps * torch.sin(angle_map)
        # ps = torch.sqrt(ps[0] ** 2 + ps[1] ** 2)
        # print(ps.shape)
        # u = input[0]
        # v = input[1]

        #angle = torch.atan(torch.abs(u / (v + 1e-8)))
        # angle = torch.atan2(u, v) + math.pi
        #pprint(torch.max(angle))


        # new

        # p = torch.sqrt(v ** 2 + u ** 2)
        # p = torch.sign(torch.sign(u) + torch.sign(v) + 0.1) * p


        # In Code Blame, before push: Christofs fault!


        #print(u.shape)
        #print(v.shape)

        # angle = torch.zeros(u.shape).cuda()

        # for i in range(0, u.shape[0]):
        #     for x in range(0, u.shape[2]):
        #         for y in range(0, u.shape[3]):
        #
        #             u_val = u[i, 0, x, y]
        #             v_val = v[i, 0, x, y]
        #
        #             if u_val > 0:
        #                 if v_val >= 0:
        #                     angle[i, 0, x, y] = torch.atan(v_val/u_val)
        #                 else:
        #                     angle[i, 0, x, y] = torch.atan(v_val/u_val) + 2 * math.pi
        #             elif u_val < 0:
        #                 angle[i, 0, x, y] = torch.atan(v_val/u_val) + math.pi
        #             else:
        #                 if v_val > 0:
        #                     angle[i, 0, x, y] = 0.5 * math.pi
        #                 else:
        #                     angle[i, 0, x, y] = 1.5 * math.pi

        #sign_v = torch.sign(torch.sign(v) + 0.5)
        #angle = math.pi - math.pi * sign_v + sign_v * torch.atan(u/(p + 1e-8))

        #pprint(torch.max(angle))

        #pprint(angle.shape)
        #pprint(u.shape)

        # some test
        # p = torch.sqrt(v ** 2 + u ** 2)
        #
        # p = torch.sign(u + v) * p

        return ps
