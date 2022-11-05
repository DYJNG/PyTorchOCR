import copy
from torch.utils.data import DataLoader, DistributedSampler

from pytocr.data.imaug import transform, create_operators
from pytocr.data.simple_dataset import SimpleDataSet
from pytocr.data.pubtab_dataset import PubTabDataSet

__all__ = ["build_dataloader", "transform", "create_operators"]


def build_dataloader(config, mode, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = ["SimpleDataSet", "PubTabDataSet"]
    module_name = config[mode]["dataset"]["name"]
    assert module_name in support_dict, Exception(
        "DataSet only support {}".format(support_dict))
    assert mode in ["Train", "Eval", "Test"
                    ], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]
    if "pin_memory" in loader_config.keys():
        pin_memory = loader_config["pin_memory"]
    else:
        pin_memory = True

    if not config["Global"]["distributed"]:
        sampler = None
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last, 
            pin_memory=pin_memory)
    else:
        if mode == "Train":
            # Distribute data to multiple cards
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=shuffle, 
                drop_last=drop_last)
        else:
            sampler = None
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory)

    return data_loader, sampler