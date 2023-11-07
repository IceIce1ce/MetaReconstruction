import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F

def hard_shrink_relu(input, lambd=0.0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim)) # M
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight) # d(z, m_i)
        att_weight = F.softmax(att_weight, dim=1) # w_i
        if self.shrink_thres > 0: # hard shrinkage for non-negative value
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres) # \hat{w}_i
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1) # Eq 7
        mem_trans = self.weight.permute(1, 0)  # M^T
        output = F.linear(att_weight, mem_trans)  # latten representation \hat{z} = \hat{w} * M
        return {'output': output, 'att': att_weight} # [3072, 256] -> [6, 2, 16, 16, 256], [3072, 2000] -> [6, 2000, 2, 16, 16]

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(self.mem_dim, self.fea_dim is not None)

class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape # [6, 256, 2, 16, 16]
        l = len(s) # 5
        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1) # [6, 2, 16, 16, 256]
        else:
            x = []
            print('Wrong feature map size')
        x = x.contiguous().view(-1, s[1]) # [3072, 256]
        y = self.memory(x)['output'] # [3072, 256]
        att = self.memory(x)['att'] # [3072, 2000]
        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1]) # [6, 2, 16, 16, 256]
            y = y.permute(0, 4, 1, 2, 3) # [6, 256, 2, 16, 16]
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim) # [6, 2, 16, 16, 2000]
            att = att.permute(0, 4, 1, 2, 3) # [6, 2000, 2, 16, 16]
        else:
            y = x # [3072, 256]
            att = att # [3072, 2000]
            print('Wrong feature map size')
        return {'output': y, 'att': att}