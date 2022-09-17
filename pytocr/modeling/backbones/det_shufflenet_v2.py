import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Callable
import logging
import os

__all__ = ["ShuffleNetV2"]

model_urls = {
    "shufflenetv2_x0.5": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenetv2_x1.0": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "shufflenetv2_x1.5": None,
    "shufflenetv2_x2.0": None,
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        scale: float = 0.5,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
        pretrained: bool = False,
        ckpt_path: str = None,
        **kwargs
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        supported_scale = [0.1, 0.5, 1.0, 1.5, 2.0]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)

        if scale == 0.1:
            stages_repeats = [2, 4, 2]
            stages_out_channels = [16, 24, 48, 96, 512]
        elif scale == 0.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 244, 488, 976, 2048]
        
        self.out_channels = []

        input_channels = in_channels
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )                                                                 # 1/2 downsample
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 1/4 downsample

        self.out_channels.append(output_channels)
        
        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, stages_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]  # downsample
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            if name != "stage4":
                self.out_channels.append(output_channels)

        output_channels = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels.append(output_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrained:
            logger = logging.getLogger("root")
            if ckpt_path and os.path.exists(ckpt_path):
                logger.info("load imagenet weights from {}".format(ckpt_path))
                state_dict = torch.load(ckpt_path)
            else:
                logger.info("load imagenet weights from url")
                state_dict = load_state_dict_from_url(model_urls["shufflenetv2_x"+str(scale)])
            self.load_state_dict(state_dict, strict=False)
            logger.info("imagenet weights load success")

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        out = []
        x = self.conv1(x)      # 1/2
        x = self.maxpool(x)    # 1/4
        out.append(x)
        x = self.stage2(x)     # 1/8
        out.append(x)
        x = self.stage3(x)     # 1/16
        out.append(x)
        x = self.stage4(x)     # 1/32
        x = self.conv5(x)      # 1/32
        out.append(x)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
