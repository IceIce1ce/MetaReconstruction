import torch
from torch import nn

def feature_map_permute(input):
    s = input.data.shape
    l = len(s)
    if l == 2:
        x = input # N x C
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1) # [6, 2, 16, 16, 2000]
    else:
        x = []
        print('Wrong feature map size')
    x = x.contiguous().view(-1, s[1]) # [3072, 2000]
    return x

class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1) # Eq 9
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        return self.entropy_loss(score)