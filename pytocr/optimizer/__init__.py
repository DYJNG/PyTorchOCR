import copy
import torch

__all__ = ["build_optimizer"]

from .lr_scheduler import WarmupMultiStepLR, WarmupPolyLR, WarmupCosineLR

def build_optimizer(config, parameters, epochs, step_each_epoch):
    config = copy.deepcopy(config)
    base_lr = config.pop("base_lr")
    optim_name = config["optim"].pop("name")
    optim = getattr(torch.optim, optim_name)(
        parameters, 
        lr=base_lr, 
        **config["optim"]) # torch optim has not "**kwargs", so params shuold correct
    
    support_dict = ["WarmupMultiStepLR", "WarmupPolyLR", "WarmupCosineLR"]

    if "lr_decay" in config and "name" in config["lr_decay"]:
        lr_decay_name = config["lr_decay"].pop("name")
        assert lr_decay_name in support_dict, Exception(
            "lr scheduler only support {}".format(support_dict))
        warmup_epoch = config["lr_decay"].pop("warmup_epoch")
        warmup_iters = warmup_epoch * step_each_epoch
        max_iters = epochs * step_each_epoch
        if "T_max_epoch" in config["lr_decay"]:
            T_max_epoch = config["lr_decay"].pop("T_max_epoch")
            T_max_iters = T_max_epoch * step_each_epoch
        else: # cosine schedule 默认周期设置
            T_max_iters = 50 * step_each_epoch
        lr_decay = eval(lr_decay_name)(
            optim, 
            warmup_iters=warmup_iters, 
            max_iters=max_iters, 
            T_max_iters=T_max_iters, 
            **config["lr_decay"])
    else:
        lr_decay = None
    return optim, lr_decay


