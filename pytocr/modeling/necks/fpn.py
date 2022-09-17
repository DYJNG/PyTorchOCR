import torch
from torch import nn
import torch.nn.functional as F

from .asf import ScaleFeatureSelection


class FPN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels=256, 
        mode=None, 
        use_asf=False, 
        attention_type="scale_spatial", 
        **kwargs):
        """
        :param in_channels: 基础网络输出的维度(1/4 1/8 1/16 1/32 四个特征图的通道列表)
        :param kwargs:
        """
        super(FPN, self).__init__()
        self.mode = mode
        self.use_asf = use_asf
        assert attention_type in [
            "scale_channel", "scale_spatial", "scale_channel_spatial"
            ], "attention_type only support [scale_channel, scale_spatial, scale_channel_spatial]"

        self.in5 = nn.Sequential(
            nn.Conv2d(
                in_channels[-1], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.in4 = nn.Sequential(
            nn.Conv2d(
                in_channels[-2], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.in3 = nn.Sequential(
            nn.Conv2d(
                in_channels[-3], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.in2 = nn.Sequential(
            nn.Conv2d(
                in_channels[-4], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

        if self.mode == "DB":
            smooth_channels = out_channels // 4
            self.out_channels = out_channels
        else:
            smooth_channels = out_channels
            self.out_channels = out_channels * 4
        self.out5 = nn.Sequential(
            nn.Conv2d(
                out_channels, smooth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(smooth_channels),
            nn.ReLU(inplace=True)
            )
        self.out4 = nn.Sequential(
            nn.Conv2d(
                out_channels, smooth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(smooth_channels),
            nn.ReLU(inplace=True)
            )
        self.out3 = nn.Sequential(
            nn.Conv2d(
                out_channels, smooth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(smooth_channels),
            nn.ReLU(inplace=True)
            )
        self.out2 = nn.Sequential(
            nn.Conv2d(
                out_channels, smooth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(smooth_channels),
            nn.ReLU(inplace=True)
            )

        if self.use_asf:  # DB++ ASF模块 也可用于PSE、PA等方法的Neck层
            self.concat_attention = ScaleFeatureSelection(
                self.out_channels, smooth_channels, attention_type=attention_type)

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
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        
        out4 = self._upsample_add(in5, in4)
        out3 = self._upsample_add(out4, in3)
        out2 = self._upsample_add(out3, in2)

        p5 = self.out5(in5)      # 1/32
        p4 = self.out4(out4)     # 1/16
        p3 = self.out3(out3)     # 1/8
        p2 = self.out2(out2)     # 1/4

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        if self.mode == "DB":
            fuse = torch.cat((p5, p4, p3, p2), dim=1)
            if self.use_asf:
                fuse = self.concat_attention(fuse, [p5, p4, p3, p2])
        else:
            fuse = torch.cat((p2, p3, p4, p5), dim=1)
            if self.use_asf:
                fuse = self.concat_attention(fuse, [p2, p3, p4, p5])
        return fuse
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="nearest") + y
