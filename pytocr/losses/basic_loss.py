import torch
import torch.nn as nn
import torch.nn.functional as F


# KL散度 又称为 相对熵，互熵 不对称 <=> H(p, q) - H(p) H:交叉熵
# JS散度 对称
# 都是为了衡量两个概率分布之间的差异性
class KLJSLoss(object):
    def __init__(self, mode="kl", reduction="mean", **kargs):
        assert mode in ["kl", "js", "KL", "JS"
                        ], "mode can only be one of ['kl', 'js', 'KL', 'JS']"
        self.mode = mode
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def __call__(self, p1, p2):

        loss = torch.mul(p2, torch.log((p2 + 1e-5) / (p1 + 1e-5) + 1e-5))

        if self.mode.lower() == "js":
            loss += torch.mul(
                p1, torch.log((p1 + 1e-5) / (p2 + 1e-5) + 1e-5))
            loss *= 0.5
        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        return loss


class DMLLoss(nn.Module):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False, **kargs):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode="js")


    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1)
            out2 = self.act(out2)
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (F.kl_div(
                log_out1, out2, reduction="batchmean") + F.kl_div(
                    log_out2, out1, reduction="batchmean")) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Module):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)
