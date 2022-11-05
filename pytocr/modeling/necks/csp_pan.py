import torch
from torch import nn
import torch.nn.functional as F

from .asf import ScaleFeatureSelection
from pytocr.modeling.utils import CNA, DPModule

__all__ = ["CSPPAN"]


class Channel_T(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        act_cfg=dict(type="LeakyReLU", inplace=True)
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(CNA(
                in_channels[i], 
                out_channels, 
                kernel_size=1, 
                padding=0, 
                act_cfg=act_cfg))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class DarknetBottleneck(nn.Module):
    """
    The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        expansion=0.5, 
        add_identity=True, 
        use_depthwise=False, 
        act_cfg=dict(type="LeakyReLU", inplace=True)
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else CNA
        self.conv1 = CNA(
            in_channels, 
            hidden_channels, 
            kernel_size=1, 
            padding=0, 
            act_cfg=act_cfg)
        self.conv2 = conv_func(
            hidden_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=1, 
            padding=(kernel_size-1)//2, 
            act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPModule(nn.Module):
    """
    Cross Stage Partial Layer.
    """
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size=3, 
        expand_ratio=0.5, 
        num_blocks=1, 
        add_identity=True, 
        use_depthwise=False, 
        act_cfg=dict(type="LeakyReLU", inplace=True)
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = CNA(
            in_channels, 
            mid_channels, 
            kernel_size=1, 
            padding=0, 
            act_cfg=act_cfg)
        self.short_conv = CNA(
            in_channels, 
            mid_channels, 
            kernel_size=1, 
            padding=0, 
            act_cfg=act_cfg)
        self.final_conv = CNA(
            mid_channels * 2, 
            out_channels, 
            kernel_size=1, 
            padding=0, 
            act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                expansion=1.0,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class CSPPAN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=5, 
        num_csp_blocks=1, 
        use_depthwise=True, 
        act_cfg=dict(type="LeakyReLU", inplace=True), 
        mode="det", 
        use_asf=False, 
        attention_type="scale_spatial", 
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.use_asf = use_asf
        assert attention_type in [
            "scale_channel", "scale_spatial", "scale_channel_spatial"
            ], "attention_type only support [scale_channel, scale_spatial, scale_channel_spatial]"
        if use_asf:  # ASF模块 
            self.concat_attention = ScaleFeatureSelection(
                out_channels*4, out_channels, attention_type=attention_type)
        if self.mode == "table":
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels * 4
        conv_func = DPModule if use_depthwise else CNA
        self.conv_t = Channel_T(
            in_channels, out_channels, act_cfg=act_cfg)
        # build top-down blocks
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPModule(
                    out_channels * 2, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    num_blocks=num_csp_blocks, 
                    add_identity=False, 
                    use_depthwise=use_depthwise, 
                    act_cfg=act_cfg)
            )
        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.downsamples.append(
                conv_func(
                    out_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=(kernel_size-1)//2, 
                    act_cfg=act_cfg)
            )
            self.bottom_up_blocks.append(
                CSPModule(
                    out_channels * 2, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    num_blocks=num_csp_blocks, 
                    add_identity=False, 
                    use_depthwise=use_depthwise, 
                    act_cfg=act_cfg)
            )
        
    
    def forward(self, x):
        # 统一各feature map的通道
        x = self.conv_t(x)

        # top-down path
        inner_outs = [x[-1]]
        for idx in range(len(x) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = x[idx - 1]
            upsample_feat = F.interpolate(
                feat_high, scale_factor=2, mode="nearest")
            inner_out = self.top_down_blocks[len(x) - 1 - idx](
                torch.cat((upsample_feat, feat_low), dim=1)
            )
            inner_outs.insert(0, inner_out) # P2, P3, P4, P5
        
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(x) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat((downsample_feat, feat_high), dim=1)
            )
            outs.append(out)  # N2 N3, N4, N5
        
        if self.mode == "table":
            return outs[-1]   # N5
        else:
            outs[-1] = F.interpolate(outs[-1], scale_factor=8, mode="nearest")
            outs[-2] = F.interpolate(outs[-2], scale_factor=4, mode="nearest")
            outs[-3] = F.interpolate(outs[-3], scale_factor=2, mode="nearest")
            fuse = torch.cat(outs, dim=1)
            if self.use_asf:
                fuse = self.concat_attention(fuse, outs)
            return fuse



