from torch import nn
from pytocr.modeling.transforms import build_transform
from pytocr.modeling.backbones import build_backbone
from pytocr.modeling.necks import build_neck
from pytocr.modeling.heads import build_head
from .base_model import BaseModel
from pytocr.utils.save_load import load_pretrained_params
import copy

__all__ = ['DistillationModel']


class DistillationModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super(DistillationModel, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_name_list = []
        for key in config["Models"]:
            model_config = copy.deepcopy(config["Models"][key])
            freeze_params = False
            pretrained = None
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            model = BaseModel(model_config)
            if pretrained is not None:
                model = load_pretrained_params(model, pretrained)
            if freeze_params:
                for param in model.parameters():
                    param.requires_grad = False
                model.training = False
            self.model_dict[key] = model
            self.model_name_list.append(key)

    def forward(self, x):
        result_dict = dict()
        for model_name in self.model_name_list:
            result_dict[model_name] = self.model_dict[model_name](x)
        return result_dict
