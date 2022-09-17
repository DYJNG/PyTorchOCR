import copy

__all__ = ["build_loss"]

# det loss
from .det_db_loss import DBLoss
from .det_pse_loss import PSELoss
from .det_pan_loss import PANLoss

# rec loss
from .rec_ctc_loss import CTCLoss

# cls loss
from .cls_loss import ClsLoss

# # e2e loss
# from .e2e_pg_loss import PGLoss
# from .kie_sdmgr_loss import SDMGRLoss

# basic loss function
from .basic_loss import DistanceLoss

# combined loss function
from .combined_loss import CombinedLoss

# # table loss
# from .table_att_loss import TableAttentionLoss


def build_loss(config):
    support_dict = [
        "DBLoss", "PSELoss", "PANLoss", "CTCLoss", "ClsLoss", "CombinedLoss"
    ]
    config = copy.deepcopy(config)    # 因为后面pop会改变config
    module_name = config.pop("name")
    assert module_name in support_dict, Exception("loss only support {}".format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class