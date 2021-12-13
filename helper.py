import torch
import torch.nn as nn
import torch.functional as F

class CSPCNN(nn.Module):
    def __init__(self, x , in_channel, out_channel, device):
        self.x = x
        self.split  = x.shape[1] // 2
    def __call__():
        csp_x = self.x[:,:split]
        skip_x = self.x[:,split:]
        conv2d = nn.Conv2d(in_channels//2, out_channels//2, stride=1, padding=1)
        csp_x = conv2d(x)
        out = torch.cat([csp_x, skip_x], 1)
        csp_x = F.relu(out)
        return csp_x



