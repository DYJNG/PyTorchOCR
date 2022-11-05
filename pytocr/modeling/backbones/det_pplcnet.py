import torch
from torch import nn
import os
import logging

from pytocr.modeling.utils import CNA, _make_divisible, DPModule

__all__ = ["PPLCNet"]


NET_CONFIG = {
    # k, in_c, out_c, s, use_se
    "blocks2": [[3, 16, 32, 1, False]], 
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]], 
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]], 
    "blocks5": [
        [3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], 
        [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]], 
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


class PPLCNet(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        scale=1.0, 
        pretrained=False, 
        ckpt_path=None, 
        **kwargs
    ):
        super().__init__()
        self.scale = scale
        self.out_channels = [
            int(NET_CONFIG["blocks3"][-1][2] * scale), 
            int(NET_CONFIG["blocks4"][-1][2] * scale), 
            int(NET_CONFIG["blocks5"][-1][2] * scale), 
            int(NET_CONFIG["blocks6"][-1][2] * scale)
        ]

        self.conv1 = CNA(
            in_channels, 
            _make_divisible(16 * scale), 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            act_cfg=dict(type="Hardswish", inplace=True))

        for key, val in NET_CONFIG.items():
            setattr(self, key, nn.Sequential(*[
                DPModule(
                    _make_divisible(in_c * scale), 
                    _make_divisible(out_c * scale), 
                    kernel_size=k, 
                    stride=s, 
                    padding=(k-1)//2, 
                    act_cfg=dict(type="Hardswish", inplace=True), 
                    use_se=se
                    ) for (k, in_c, out_c, s, se) in val
                ])
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
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
                self.load_state_dict(state_dict, strict=False)
                logger.info("imagenet weights load success")
            else:
                logger.info("imagenet ckpt_path not exists")
        
    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = getattr(self, "blocks2")(x)
        for i in range(4):
            x = getattr(self, "blocks{}".format(i + 3))(x)
            out.append(x)
        
        return out