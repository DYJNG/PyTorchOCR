__all__ = ["build_backbone"]

def build_backbone(config, model_type):
    if model_type == "det":
        from .det_resnet import ResNet
        from .det_mobilenet_v3 import MobileNetV3
        from .det_shufflenet_v2 import ShuffleNetV2
        from .det_repvgg import RepVGG
        from .det_convnext import ConvNeXt
        from .det_swin import SwinTransformer
        support_dict = [
            "ResNet", "MobileNetV3", "ShuffleNetV2", 
            "RepVGG", "ConvNeXt", "SwinTransformer"
        ]
    elif model_type == "rec" or model_type == "cls":
        from .rec_vgg import VGG
        from .rec_resnet import ResNet
        from .rec_mobilenet_v3 import MobileNetV3
        support_dict = ["VGG", "ResNet", "MobileNetV3"]
    else:
        raise NotImplementedError

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type, support_dict))
    module_class = eval(module_name)(**config)
    return module_class