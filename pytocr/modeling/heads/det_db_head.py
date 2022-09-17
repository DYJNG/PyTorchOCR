import torch
from torch import nn


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50, **kwargs):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4,  kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2, padding=0, bias=True),
            nn.Sigmoid())

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4,  kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2, padding=0, bias=True),
            nn.Sigmoid())
        
        """
        自定义模型参数初始化
        逐级递归遍历所有的nn模块，对所有的卷积层、BN层等进行初始化
        对于未选择的层，会自动采用pytorch默认的初始化(conv默认kaiming_normal, bn默认[1, 0])
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x, **kwargs):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {"maps": shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return {"maps": y}

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
