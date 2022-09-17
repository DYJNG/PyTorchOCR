from torch import nn
from pytocr.modeling.transforms import build_transform
from pytocr.modeling.backbones import build_backbone
from pytocr.modeling.necks import build_neck
from pytocr.modeling.heads import build_head

# 设置可被其他文件import的变量或函数
__all__ = ["BaseModel"]


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]
        # build transform,
        # for rec, transfrom can be TPS,None
        # for det and cls, transform shoule to be None,
        # if you make model differently, you can use transform in det and cls
        if "Transform" not in config or config["Transform"] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        config["Backbone"]["in_channels"] = in_channels
        self.backbone = build_backbone(config["Backbone"], model_type)
        in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if "Neck" not in config or config["Neck"] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_neck(config["Neck"])
            in_channels = self.neck.out_channels

        # build head, head is need for det, rec and cls
        config["Head"]["in_channels"] = in_channels
        self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        x = self.backbone(x)
        y["backbone_out"] = x
        if self.use_neck:
            x = self.neck(x)
        y["neck_out"] = x
        x = self.head(x, targets=data)  # target用于识别注意力机制Head的输入，参考ppocr
        if isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            return y
        else:
            return x

# if __name__ == "__main__":
#     import yaml
#     import torch

#     file_path = "F:/Repos/OCR/PyTorchOCR/configs/det/det_r50_db.yml"
#     config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
#     arch = BaseModel(config["Architecture"])
#     # print(arch)
#     x = torch.randint(0, 255, (1, 3, 224, 224)).float()
#     arch.eval()
#     with torch.no_grad():
#         res = arch(x)
#     print(res["maps"].shape)