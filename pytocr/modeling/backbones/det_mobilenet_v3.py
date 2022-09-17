import os
import torch
import logging
from collections import OrderedDict

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, List, Optional, Sequence

from torchvision.models.utils import load_state_dict_from_url


__all__ = ["MobileNetV3"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand   # 与其他版本实现不同的地方
        if cnf.expanded_channels != cnf.input_channels:
            self.conv1 = ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                          norm_layer=norm_layer, activation_layer=activation_layer)
        else:
            self.conv1 = None

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        self.conv2 = ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                      stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                      norm_layer=norm_layer, activation_layer=activation_layer)
        if cnf.use_se:
            self.se = se_layer(cnf.expanded_channels)
        else:
            self.se = None

        # project
        self.conv3 = ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                      activation_layer=nn.Identity)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.conv1 is not None:
            out = self.conv1(x)
        else:
            out = x
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        if self.use_res_connect:
            out += identity
        return out


class MobileNetV3(nn.Module):

    def __init__(
            self,
            in_channels: int = 3, 
            model_name: str = "large",
            width_mult: float = 1.0,
            use_se: bool = True,
            dilation: bool = False,
            reduced_tail: bool = False,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            pretrained: bool = False,
            ckpt_path: str = None,
            **kwargs
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super(MobileNetV3, self).__init__()

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert width_mult in supported_scale, \
            "supported scale are {} but input width_mult is {}".format(supported_scale, width_mult)
        
        inverted_residual_setting = _mobilenet_v3_conf(
                                                    model_name,
                                                    width_mult=width_mult,
                                                    use_se=use_se,
                                                    _reduced_tail=reduced_tail, 
                                                    _dilation=dilation)
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.conv1 = ConvBNActivation(in_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish)
        self.stages = nn.ModuleList()   # ModuleList ModuleDict
        self.out_channels = []
        layers: List[nn.Module] = []
        i = 0
        start_idx = 2 if model_name == "large" else 0
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            if cnf.stride == 2 and i > start_idx:
                self.stages.append(nn.Sequential(*layers))
                self.out_channels.append(cnf.input_channels)
                layers: List[nn.Module] = []
            layers.append(block(cnf, norm_layer))
            i += 1

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.stages.append(nn.Sequential(*layers))
        self.out_channels.append(lastconv_output_channels)

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
                state_dict = load_state_dict_from_url(model_urls["mobilenet_v3_"+str(layers)])
            # 过滤se
            filtered_dict = OrderedDict()
            for k, v in state_dict.items():
                flag = k.find("se") != -1
                if not use_se and flag:
                    continue
                filtered_dict[k] = v
            model_state_dict = self.state_dict()
            model_keys = list(model_state_dict.keys())
            for i, (k, v) in enumerate(filtered_dict.items()):
                if i >= len(model_keys):
                    break
                if k in model_state_dict:
                    name = k
                else:
                    name = model_keys[i]
                model_state_dict[name] = v
            self.load_state_dict(model_state_dict, strict=False)
            logger.info("imagenet weights load success")

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(arch: str, width_mult=1.0, use_se=True, _reduced_tail=False, _dilation=False):
    # non-public config parameters
    reduce_divider = 2 if _reduced_tail else 1
    dilation = 2 if _dilation else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)

    if arch == "large":
        inverted_residual_setting = [
                    #  in  k  exp out  se     act  s dila
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True and use_se, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True and use_se, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True and use_se, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True and use_se, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True and use_se, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
    elif arch == "small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True and use_se, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True and use_se, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True and use_se, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True and use_se, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True and use_se, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True and use_se, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting

