import errno
import os

import torch

from pytocr.utils.logging import get_logger

__all__ = ["load_model", "save_model"]


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(
                        path))
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, 
               model, 
               optimizer=None, 
               device=torch.device("cuda:0")):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    global_state = {}
    if checkpoints:
        ckpt = torch.load(checkpoints, map_location=device)
        model_state_dict = model.state_dict()
        for k, v in ckpt["state_dict"].items():
            if k in model_state_dict:
                name = k
            elif "module." + k in model_state_dict:
                name = "module." + k
            else:
                name = k.replace("module.", "")  # remove `module.`
            model_state_dict[name] = v
        model.load_state_dict(model_state_dict, strict=True)
        if optimizer is not None:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            else:
                logger.warning(
                    "{}.pth is not exists, params of optimizer is not loaded".
                    format(checkpoints))
        if "global_state" in ckpt:
            global_state = ckpt["global_state"]
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        pretrained_state_dict = torch.load(pretrained_model, map_location=device)
        if "state_dict" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["state_dict"]
        model_state_dict = model.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict:
                name = k
            elif "module." + k in model_state_dict:
                name = "module." + k
            else:
                name = k.replace("module.", "")  # remove `module.`
            model_state_dict[name] = v
        model.load_state_dict(model_state_dict, strict=True)
        logger.info("load pretrain successful from {}".format(pretrained_model))
    else:
        logger.info("train from scratch")
    return model, optimizer, global_state


def load_pretrained_params(model, path):
    logger = get_logger()
    assert os.path.exists(path), \
        "The {} does not exists!".format(path)

    pretrained_state_dict = torch.load(path)
    if "state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["state_dict"]
    model_state_dict = model.state_dict()
    for k, v in pretrained_state_dict.items():
        if k in model_state_dict:
            name = k
        elif "module." + k in model_state_dict:
            name = "module." + k
        else:
            name = k.replace("module.", "")  # remove `module.`
        model_state_dict[name] = v
    model.load_state_dict(model_state_dict, strict=True)
    # torch.save(model_state_dict, "./model_tmp.pth")  # 保存纯模型权重
    logger.info("load pretrain successful from {}".format(path))
    return model


def save_model(model, 
               optimizer, 
               cfg, 
               model_dir, 
               logger, 
               is_best=False, 
               prefix="pytocr", 
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_dir, logger)
    model_path = os.path.join(model_dir, prefix+".pth")
    if torch.cuda.device_count() > 1:
        mode_state_dict = model.module.state_dict()
    else:
        mode_state_dict = model.state_dict()
    state = {"state_dict": mode_state_dict,
             "optimizer": optimizer.state_dict(),
             "cfg": cfg}
    state.update(kwargs)
    torch.save(state, model_path)
    if is_best:
        logger.info("save best model is to {}".format(model_path))
    else:
        logger.info("save model in {}".format(model_path))
