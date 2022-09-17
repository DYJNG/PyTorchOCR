import torch
from torch import nn
import torch.nn.functional as F

from .asf import ScaleFeatureSelection


class FPEM(nn.Module):
    def __init__(self, planes, mode="v2"):
        super(FPEM, self).__init__()
        self.mode = mode
        self.dwconv3_1 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=1, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer3_1 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

        self.dwconv2_1 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=1, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer2_1 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

        self.dwconv1_1 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=1, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer1_1 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

        self.dwconv2_2 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=2, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer2_2 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

        self.dwconv3_2 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=2, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer3_2 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

        self.dwconv4_2 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=2, padding=1, 
            groups=planes, 
            bias=False)
        self.smooth_layer4_2 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="nearest") + y

    def forward(self, x):
        f1, f2, f3, f4 = x
        # dwconv stride = 1
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))    # 1/16
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))   # 1/8
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))   # 1/4

        # dwconv stride = 2
        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_, f1_)))  # 1/8 -> 1/4 -> 1/8
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_, f2_)))  # 1/16 -> 1/8 -> 1/16
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))   # 1/32 -> 1/16 -> 1/32

        if self.mode == "v2":
            f1 = f1 + f1_
            f2 = f2 + f2_
            f3 = f3 + f3_
            f4 = f4 + f4_
        else:
            f1, f2, f3, f4 = f1_, f2_, f3_, f4_
        return [f1, f2, f3, f4]

class FFM(nn.Module):
    def __init__(self, mode="v2", concat_attention=None):
        super(FFM, self).__init__()
        self.mode = mode
        self.concat_attention = concat_attention
    
    def forward(self, x):
        if self.mode == "v2":
            f1, f2, f3, f4 = x[-1]
        else:
            f1, f2, f3, f4 = x[0]
            for i in range(1, len(x)):
                f1 += x[i][0]
                f2 += x[i][1]
                f3 += x[i][2]
                f4 += x[i][3]

        f2 = F.interpolate(f2, scale_factor=2, mode="nearest")
        f3 = F.interpolate(f3, scale_factor=4, mode="nearest")
        f4 = F.interpolate(f4, scale_factor=8, mode="nearest")

        fuse = torch.cat((f1, f2, f3, f4), dim=1)
        if self.concat_attention is not None:  # asf
            fuse = self.concat_attention(fuse, [f1, f2, f3, f4])
        return fuse


class FPEM_FFM(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels=128, 
        mode="v2", 
        fpem_num=2, 
        use_asf=False, 
        attention_type="scale_spatial", 
        **kwargs):
        super(FPEM_FFM, self).__init__()
        assert mode in [
            "v1", "v2"], "FPEM_FFM mode only support [v1, v2]"
        assert attention_type in [
            "scale_channel", "scale_spatial", "scale_channel_spatial"
            ], "attention_type only support [scale_channel, scale_spatial, scale_channel_spatial]"
        if use_asf:  # DB++ ASF模块 也可用于PSE、PA等方法的Neck层
            concat_attention = ScaleFeatureSelection(
                out_channels*4, out_channels, attention_type=attention_type)
        else:
            concat_attention = None

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
        
        for i in range(fpem_num):
            setattr(self, "fpem_{}".format(i + 1), FPEM(out_channels, mode))
        self.ffm = FFM(mode, concat_attention)
        self.fpem_num = fpem_num
        self.out_channels = out_channels * 4
        
    def forward(self, x):
        c2, c3, c4, c5 = x
        ins = [self.in2(c2), self.in3(c3), self.in4(c4), self.in5(c5)]
        fpems = []
        for i in range(self.fpem_num):
            ins = getattr(self, "fpem_{}".format(i + 1))(ins)
            fpems.append(ins)
        fuse = self.ffm(fpems)

        return fuse
