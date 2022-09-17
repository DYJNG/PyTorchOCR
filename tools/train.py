import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

import torch

from pytocr.data import build_dataloader
from pytocr.modeling.architectures import build_model
from pytocr.losses import build_loss
from pytocr.optimizer import build_optimizer
from pytocr.postprocess import build_post_process
from pytocr.metrics import build_metric
from pytocr.utils.save_load import load_model
import tools.program as program


def main(config, device, local_rank, logger, tsb_writer):
        
    global_config = config["Global"]

    # build dataloader
    train_dataloader, train_sampler = build_dataloader(config, "Train", logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config["Eval"]:
        valid_dataloader, valid_sampler = build_dataloader(config, "Eval", logger)
    else:
        valid_dataloader = None
        valid_sampler = None

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation"]:  # distillation model
            for key in config["Architecture"]["Models"]:
                config["Architecture"]["Models"][key]["Head"][
                    "out_channels"] = char_num
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    model = model.to(device)
    # amp
    use_amp = config["Global"].get("use_amp", False)
    if not use_amp:
        # parallel  DP and DDP
        if not config["Global"]["distributed"]:
            model = torch.nn.DataParallel(model)
        else:   # distribute
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ## 同步bn
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank)
    model.train()

    # build loss
    loss_class = build_loss(config["Loss"]) 
    # loss_class.to(device)  # 损失函数通常不需要再迁移到特定gpu，
                             # 因为损失函数接收一个或多个输入tensor，
                             # 如果输入tensor本身就是在gpu上，则输出tensor自然就在gpu上。

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # amp
    if use_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer)
        # parallel  DP and DDP
        if not config["Global"]["distributed"]:
            model = torch.nn.DataParallel(model)
        else:
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ## 同步bn
            # model = torch.nn.parallel.DistributedDataParallel(
            #     model, 
            #     device_ids=[local_rank], 
            #     output_device=local_rank)
            from apex.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)    # BUG
        
    # load pretrain model
    model, optimizer, global_state = load_model(config, model, optimizer, device)
    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(
            len(valid_dataloader)))

    # build metric
    eval_class = build_metric(config["Metric"])

    # start train
    program.train(config, device, local_rank, train_dataloader, train_sampler, valid_dataloader, valid_sampler, 
                  model, loss_class, optimizer, lr_scheduler, global_state, post_process_class, eval_class, 
                  logger, tsb_writer, use_amp)



if __name__ == "__main__":
    config, device, local_rank, logger, tsb_writer = program.preprocess(is_train=True)
    main(config, device, local_rank, logger, tsb_writer)
