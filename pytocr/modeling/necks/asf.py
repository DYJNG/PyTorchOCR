"""
Adaptive Scale Fusion module (feature attention)
"""
import torch
from torch import nn
import torch.nn.functional as F


class ScaleChannelAttention(nn.Module):
    """
    通道注意力，类似于SE
    输出尺寸为1*1，输出通道数等于FPN的特征层数
    """
    def __init__(self, in_channels, mid_channels, num_features):
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.fc2 = nn.Conv2d(
            mid_channels, num_features, kernel_size=1, bias=False)

    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        channel_atten = F.softmax(global_x, dim=1)
        return channel_atten


class ScaleChannelSpatialAttention(nn.Module):
    """
    通道空间混合注意力，类似于CBAM，先通道注意力后空间注意力
    输出尺寸为H*W，输出通道数等于FPN的特征层数
    """
    def __init__(self, in_channels, mid_channels, num_features):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels, mid_channels , kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_wise = nn.Sequential(
            nn.Conv2d(
                1, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(
                1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(
                in_channels, num_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # N*C*1*1 通道注意力
        channel_atten = self.channel_wise(x)
        # N*C*H*W python broadcast
        global_x = channel_atten + x  # 为什么是+呢，注意力加权应该是*吧
        # N*1*H*W 空间注意力
        x = torch.mean(global_x, dim=1, keepdim=True)
        spatial_atten = self.spatial_wise(x)
        # N*C*H*W python broadcast
        global_x = spatial_atten + global_x
        # N*num_features*H*W
        mix_atten = self.attention_wise(global_x)
        return mix_atten


class ScaleSpatialAttention(nn.Module):
    """
    空间注意力
    输出尺寸为H*W，输出通道数等于FPN的特征层数
    """
    def __init__(self, in_channels, num_features):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            nn.Conv2d(
                1, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(
                1, 1, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(
                in_channels, num_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # N*1*H*W
        global_x = torch.mean(x, dim=1, keepdim=True)
        spatial_atten = self.spatial_wise(global_x)
        # N*C*H*W python broadcast
        global_x = spatial_atten + x
        # N*num_features*H*W
        spatial_atten = self.attention_wise(global_x)
        return spatial_atten


class ScaleFeatureSelection(nn.Module):
    def __init__(
        self, 
        in_channels, 
        inter_channels, 
        out_features_num=4, 
        attention_type="scale_spatial"):
        
        super(ScaleFeatureSelection, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, inter_channels, kernel_size=3, padding=1)

        self.type = attention_type
        if self.type == "scale_spatial":
            self.enhanced_attention = ScaleSpatialAttention(
                inter_channels, out_features_num)
        elif self.type == "scale_channel_spatial":
            self.enhanced_attention = ScaleChannelSpatialAttention(
                inter_channels, inter_channels // 4, out_features_num)
        elif self.type == "scale_channel":
            self.enhanced_attention = ScaleChannelAttention(
                inter_channels, inter_channels//2, out_features_num)
        
        self.out_features_num = out_features_num

        # --- 自定义模型参数初始化 --- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
    
    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ["scale_channel_spatial", "scale_spatial"]:
            # N*num_features*1*1 -> N*num_features*H*W
            shape = features_list[0].shape[2:]
            # 双线性插值，转onnx时需要注意
            score = F.interpolate(score, size=shape, mode="bilinear")
        x = []
        for i in range(self.out_features_num):
            # 对不同特征层进行注意力加权
            # 每个特征层中，注意力张量进行广播，每个特征层内共享同一种注意力权重
            # score: N*1*H*W -> N*C_cur_feature*H*W
            # 本质上最终是对不同的特征层进行注意力加权
            x.append(score[:, i:i+1] * features_list[i])
        return torch.cat(x, dim=1)
