import torch
import numpy as np
import torch.nn as nn

class QuantLoss(nn.Module):
    """
    the quantization errrp
    """
    def __init__(self):
        super(QuantLoss,self).__init__()

    def forward(self, features):
        out = torch.abs(features) - 1.
        out = torch.mean(torch.square(out))
        return out