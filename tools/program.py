import os
import sys
import platform
import yaml
import time
import random
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from pytocr.utils.stats import TrainingStats
from pytocr.utils.save_load import save_model
from pytocr.utils.utility import print_dict
from pytocr.utils.logging import get_logger


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", type=str, help="configuration file to use")
        self.add_argument("-o", "--opt", type=str, nargs="+", help="set configuration options")
        # 这个参数是torch.distributed.launch传递过来的，我们设置位置参数来接受，local_rank代表当前程序进程使用的GPU标号
        self.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

global_config = AttrDict()

default_config = {"Global": {"debug": False, }}

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, "rb"), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split(".")
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in pytorch
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using pytorch cpu version ! \nPlease try: \n" \
          "\t1. Install pytorch-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not torch.cuda.is_available():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    world_size = dist.get_world_size()
    if world_size == 1:
        return False
    dist.barrier()
    return True


def set_random_seed(seed, use_cuda=True, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        use_cuda: whether depend on cuda
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def train(config, device, local_rank, 
          train_dataloader, train_sampler, 
          valid_dataloader, valid_sampler, 
          model, loss_class,
          optimizer, lr_scheduler,
          global_state, 
          post_process_class,
          eval_class,
          logger, tsb_writer=None,
          use_amp=False):
    # train_options
    cal_metric_during_train = config["Global"].get(
        "cal_metric_during_train", False)
    log_smooth_window = config["Global"]["log_smooth_window"]
    epoch_num = config["Global"]["epoch_num"]
    print_batch_step = config["Global"]["print_batch_step"]
    # eval_batch_step = config["Global"]["eval_batch_step"]
    eval_epoch_step = config["Global"]["eval_epoch_step"]
    # resume_options
    main_indicator = eval_class.main_indicator
    if len(global_state) > 0:
        best_model_dict = global_state["best_model"]
        start_epoch = global_state["start_epoch"]
        global_step = global_state["global_step"]
    else:
        best_model_dict = {main_indicator: 0}
        start_epoch = 0
        global_step = 0
    # eval_options
    # start_eval_step = 0
    # if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
    #     start_eval_step = eval_batch_step[0]
    #     eval_batch_step = eval_batch_step[1]
    #     if len(valid_dataloader) == 0:
    #         logger.info(
    #             "No Images in eval dataset, evaluation during training will be disabled"
    #         )
    #         start_eval_step = 1e111
    #     logger.info(
    #         "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
    #         format(start_eval_step, eval_batch_step))
    # eval_options
    start_eval_step = 0
    if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
        start_eval_step = eval_epoch_step[0]
        eval_epoch_step = eval_epoch_step[1]
        if local_rank == 0:
            if len(valid_dataloader) == 0:
                logger.info(
                    "No Images in eval dataset, evaluation during training will be disabled"
                )
                start_eval_step = 1e111
            logger.info(
                "During the training process, after the {}th epoch, an evaluation is run every {} epochs".
                format(start_eval_step, eval_epoch_step))
    # save_options
    ckpt_save_type = config["Global"]["ckpt_save_type"]
    save_epoch_step = config["Global"]["save_epoch_step"]
    save_model_dir = config["Global"]["save_model_dir"]
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    train_stats = TrainingStats(log_smooth_window, ["lr"])
    model_average = False   # SRN NEED

    use_srn = config["Architecture"]["algorithm"] == "SRN"
    extra_input = config["Architecture"][
        "algorithm"] in ["SRN", "NRTR", "SAR", "SEED"]
    try:
        model_type = config["Architecture"]["model_type"]
    except:
        model_type = None
    # train_process
    for epoch in range(start_epoch, epoch_num):
        model.train()
        if config["Global"]["distributed"] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        max_iter = len(train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            # transfer data to gpu (batch is a list, not dict)
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            # load data cost time
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            # get cur learning rate
            lr = optimizer.param_groups[0]["lr"]
            # lr = optimizer.state_dict()["param_groups"][0]["lr"]
            images = batch[0]
            if use_srn:
                model_average = True
            train_start = time.time()
            # zero the params grad
            optimizer.zero_grad()
            # forward
            if model_type == "table" or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type == "kie":
                preds = model(batch)
            else:
                preds = model(images)
            # cal loss
            loss = loss_class(preds, batch)
            avg_loss = loss["loss"]
            # backward and step
            if use_amp:
                from apex import amp
                with amp.scale_loss(avg_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                avg_loss.backward()
            optimizer.step()
            # model train cost time
            train_run_cost += time.time() - train_start
            total_samples += len(images)
            # lr decay
            if lr_scheduler is not None:
                # lr_scheduler.step(global_step) # is ok?
                if len(global_state) > 0:   # resume
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()
            
            # acc iou
            # score_shrink_map = cal_text_score(preds[:, 0, :, :], batch["shrink_map"], batch["shrink_mask"], running_metric_text,
            #                                   thred=self.config["post_processing"]["args"]["thresh"])
                
            # logger and tensorboard
            stats = {k: v.detach().cpu().numpy().mean() for k, v in loss.items()}
            stats["lr"] = lr
            train_stats.update(stats)
            # cal trainning metric (only rec and cls need)
            if cal_metric_during_train and model_type is not "det":
                batch = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item.numpy() for item in batch]
                if model_type in ["table", "kie"]:
                    eval_class(preds, batch)
                else:
                    post_result = post_process_class(preds, batch[1])
                    eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)
            # tensorboard write
            if local_rank == 0 and tsb_writer is not None:
                for k, v in train_stats.get().items():
                    tsb_writer.add_scalar("TRAIN/{}".format(k), v, global_step)
                tsb_writer.add_scalar("TRAIN/lr", lr, global_step)
            # logger print
            if local_rank == 0 and (
                (global_step > 0 and global_step % print_batch_step == 0) or
                (idx == max_iter - 1)):
                logs = train_stats.log()
                strs = "epoch: [{}/{}], iter: {}, {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ips: {:.5f}".format(
                    epoch + 1, epoch_num, global_step, logs, train_reader_cost /
                    print_batch_step, (train_reader_cost + train_run_cost) /
                    print_batch_step, total_samples,
                    total_samples / (train_reader_cost + train_run_cost))
                logger.info(strs)
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            global_step += 1
            reader_start = time.time()

        # eval
        if local_rank == 0 and epoch + 1 > start_eval_step and \
                (epoch - start_eval_step + 1) % eval_epoch_step == 0:
            if model_average:
                pass
                # Model_Average = paddle.incubate.optimizer.ModelAverage(
                #     0.15,
                #     parameters=model.parameters(),
                #     min_average_window=10000,
                #     max_average_window=15625)
                # Model_Average.apply()
            if config["Global"]["distributed"] and valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            cur_metric = eval(
                model,
                device, 
                valid_dataloader,
                post_process_class,
                eval_class,
                model_type,
                extra_input=extra_input)
            cur_metric_str = "cur metric, {}".format(", ".join(
                ["{}: {}".format(k, v) for k, v in cur_metric.items()]))
            logger.info(cur_metric_str)

            # logger metric
            if tsb_writer is not None:
                for k, v in cur_metric.items():
                    if isinstance(v, (float, int)):
                        tsb_writer.add_scalar("EVAL/{}".format(k),
                                                cur_metric[k], global_step)
            if cur_metric[main_indicator] >= best_model_dict[
                    main_indicator]:
                best_model_dict.update(cur_metric)
                best_model_dict["best_model_epoch"] = epoch + 1
                global_state["start_epoch"] = epoch
                global_state["best_model"] = best_model_dict
                global_state["global_step"] = global_step
                save_model(
                    model,
                    optimizer,
                    config, 
                    save_model_dir,
                    logger,
                    is_best=True,
                    prefix="best_accuracy",
                    global_state=global_state)
            best_str = "best metric, {}".format(", ".join([
                "{}: {}".format(k, v) for k, v in best_model_dict.items()
            ]))
            logger.info(best_str)
            # logger best metric
            if tsb_writer is not None:
                tsb_writer.add_scalar("EVAL/best_{}".format(main_indicator),
                                        best_model_dict[main_indicator],
                                        global_step)
            
        if local_rank == 0:
            global_state["start_epoch"] = epoch
            global_state["best_model"] = best_model_dict
            global_state["global_step"] = global_step
            save_model(
                model,
                optimizer,
                config, 
                save_model_dir,
                logger,
                is_best=False,
                prefix="latest",
                global_state=global_state)
            if ckpt_save_type == "FixedEpochStep" and epoch + 1 > 0 and \
                                (epoch + 1) % save_epoch_step == 0:
                save_model(
                    model,
                    optimizer,
                    config, 
                    save_model_dir,
                    logger,
                    is_best=False,
                    prefix="epoch_{}".format(epoch),
                    global_state=global_state)
    
    if local_rank == 0:
        best_str = "best metric, {}".format(", ".join(
            ["{}: {}".format(k, v) for k, v in best_model_dict.items()]))
        logger.info(best_str)
        if tsb_writer is not None:
            tsb_writer.close()
    return


def eval(model,
         device, 
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc="eval model:",
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            # transfer data to gpu (batch is a list, not dict)
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()
            if model_type == "table" or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type == "kie":
                preds = model(batch)
            else:
                preds = model(images)
            batch = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item.numpy() for item in batch]
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ["table", "kie"]:
                eval_class(preds, batch)
            else:
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric["fps"] = total_frame / total_time
    return metric


def preprocess(is_train=False):
    args = ArgsParser().parse_args()
    config = load_config(args.config)
    merge_config(args.opt)   ### config 会随着 global_config 改变吗   -> YES

    if is_train:
        # save_config
        save_model_dir = config["Global"]["save_model_dir"]
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = "{}/train.log".format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(name="root", log_file=log_file)

    # check if set use_gpu=True
    use_gpu = config["Global"]["use_gpu"]
    check_gpu(use_gpu)

    alg = config["Architecture"]["algorithm"]
    assert alg in [
        "DB", "PSE", "PAN", "CRNN", "CLS", "Distillation"
    ] # TODO: add new alg

    device = torch.device("cuda:{}".format(args.local_rank) if use_gpu else "cpu")

    # init dist environment
    if use_gpu and torch.cuda.device_count() > 1 and config["Global"]["distributed"]:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        config["Global"]["distributed"] = synchronize()
    else:
        config["Global"]["distributed"] = False

    if config["Global"]["use_tensorboard"] and args.local_rank == 0:
        from tensorboardX import SummaryWriter                 # logdir
        # from torch.utils.tensorboard import SummaryWriter    # log_dir
        save_model_dir = config["Global"]["save_model_dir"]
        tsb_writer_path = "{}/tensorboard/".format(save_model_dir)
        os.makedirs(tsb_writer_path, exist_ok=True)
        tsb_writer = SummaryWriter(logdir=tsb_writer_path)
    else:
        tsb_writer = None
    
    # 初始化随机种子，保证输入一致的情况下，输出一致
    set_random_seed(config["Global"]["seed"], use_gpu, deterministic=True)
    # if args.local_rank == 0:
    print_dict(config, logger)
    logger.info("train with torch {} and device {}".format(torch.__version__,
                                                                device))
    return config, device, args.local_rank, logger, tsb_writer
