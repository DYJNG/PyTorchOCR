import torch
from torch import nn
import logging
import os

__all__ = ["VGG"]

class VGG(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_name: str = "v1",
        scale: float = 1.0,
        leaky_relu: bool = False,
        pretrained: bool = False,
        ckpt_path: str = None,
        **kwargs
    ) -> None:
        super(VGG, self).__init__()
        
        supported_model = ["v1", "v2"]
        assert model_name in supported_model, \
            "supported vgg model are {} but input model_name is {}".format(supported_model, model_name)
        
        supported_scale = [0.5, 1.0]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)

        if model_name == "v1":
            ks = [3, 3, 3, 3, 3, 3, 2]
            ps = [1, 1, 1, 1, 1, 1, 0]
            ss = [1, 1, 1, 1, 1, 1, 1]
            if scale == 0.5:
                nm = [32, 64, 128, 128, 256, 256, 512]
            elif scale == 1.0:
                nm = [64, 128, 256, 256, 512, 512, 512]
        elif model_name == "v2":
            ks = [5, 3, 3, 3, 3, 3, 2]
            ps = [2, 1, 1, 1, 1, 1, 0]
            ss = [2, 1, 1, 1, 1, 1, 1]
            if scale == 0.5:
                nm = [32, 64, 128, 128, 256, 256, 256]
            elif scale == 1.0:
                nm = [24, 128, 256, 256, 512, 512, 512]
        
        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            n_in = in_channels if i == 0 else nm[i - 1]
            n_out = nm[i]
            if model_name == "v1":
                cnn.add_module("conv{0}".format(i),
                            nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
                if batch_normalization:
                    cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(n_out))
                if leaky_relu:
                    cnn.add_module("relu{0}".format(i),
                                nn.LeakyReLU(0.2, inplace=True))
                else:
                    cnn.add_module("relu{0}".format(i), nn.ReLU(True))
            elif model_name == "v2":
                if i == 0:
                    cnn.add_module("conv_{0}".format(i),
                                nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
                    cnn.add_module("relu_{0}".format(i), nn.ReLU(True))
                else:
                    cnn.add_module("conv{0}".format(i),
                                nn.Conv2d(n_in, n_in, ks[i], ss[i], ps[i], groups=n_in))
                    if batch_normalization:
                        cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(n_in))
                    cnn.add_module("relu{0}".format(i), nn.ReLU(True))
                    cnn.add_module("convproject{0}".format(i),
                                nn.Conv2d(n_in, n_out, 1, 1, 0))
                    if batch_normalization:
                        cnn.add_module("batchnormproject{0}".format(i), nn.BatchNorm2d(n_out))
                    cnn.add_module("reluproject{0}".format(i), nn.ReLU(True))
        
        conv_relu(0)
        if model_name == "v1":
            cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module("pooling{0}".format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module("pooling{0}".format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.out_channels = nm[-1]
        self.cnn = cnn

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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cnn(x)
        return out
