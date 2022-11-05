import copy

__all__ = ["build_post_process"]

from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .pse_postprocess import PSEPostProcess
from .pan_postprocess import PANPostProcess
from .rec_postprocess import CTCLabelDecode, DistillationCTCLabelDecode
from .cls_postprocess import ClsPostProcess
from .table_postprocess import TableLabelDecode


def build_post_process(config, global_config=None):
    support_dict = [
        "DBPostProcess", "PSEPostProcess", "PANPostProcess",
        "CTCLabelDecode", "ClsPostProcess", 
        "DistillationDBPostProcess", "DistillationCTCLabelDecode", 
        "TableLabelDecode"
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
