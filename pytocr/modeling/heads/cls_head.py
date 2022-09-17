import torch
from torch import nn
from torch.nn import functional as F

class ClsHead(nn.Module):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, class_dim)

    def forward(self, x, **kwargs):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x
