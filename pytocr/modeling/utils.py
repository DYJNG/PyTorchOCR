from torch import nn


# Conv + Norm + Activate
class CNA(nn.Module):
    def __init__(
        self, 
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        dilation=1, 
        num_groups=1, 
        bias=False, 
        norm_cfg=dict(type="BatchNorm2d"), 
        act_cfg=dict(type="ReLU", inplace=True)
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=num_groups, 
            bias=bias)
        if norm_cfg is not None:
            _norm_cfg = norm_cfg.copy()
            norm_type = _norm_cfg.pop("type")
            self.norm = getattr(nn, norm_type)(out_planes, **_norm_cfg)
        else:
            self.norm = None
        if act_cfg is not None:
            _act_cfg = act_cfg.copy()
            act_type = _act_cfg.pop("type")
            self.activate = getattr(nn, act_type)(**_act_cfg)
        else:
            self.activate = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)   
        return x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    def __init__(self, in_planes, squeeze_factor=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeeze_channels = _make_divisible(in_planes // squeeze_factor)
        self.fc1 = nn.Conv2d(in_planes, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_planes, 1)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hardsigmoid(x)
        x = x * identity
        return x


class DPModule(nn.Module):
    """
    Depth-wise and point-wise module.
    """
    def __init__(
        self, 
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        dilation=1, 
        bias=False, 
        norm_cfg=dict(type="BatchNorm2d"), 
        act_cfg=dict(type="ReLU", inplace=True), 
        use_se=False
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = CNA(
            in_planes, 
            in_planes, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            num_groups=in_planes, 
            bias=bias, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg)
        if use_se:
            self.se = SEModule(in_planes)
        self.pw_conv = CNA(
            in_planes, 
            out_planes, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias, 
            act_cfg=act_cfg)
    
    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)

        return x