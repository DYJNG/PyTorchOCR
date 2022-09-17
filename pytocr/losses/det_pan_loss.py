import torch
from torch import nn
from torch.nn import functional as F

from .det_basic_loss import DiceLoss, EmbLoss, IoU, OHEM_BATCH


class PANLoss(nn.Module):
    def __init__(self,
                 alpha=1.0,
                 beta=0.5,
                 gamma=0.25,
                 feature_dim=4,
                 mode="v2",
                 ohem_ratio=3,
                 kernel_sample_mask="pred",
                 reduction="mean",
                 **kwargs):
        """Implement PAN Loss.
        """
        super(PANLoss, self).__init__()
        assert reduction in ["sum", "mean", "none"]
        assert mode in [
            "v1", "v2"], "EmbLoss mode only support [v1, v2]"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction

        self.dice_loss = DiceLoss(eps=0.001, act=True, reduce=False)
        self.emb_loss = EmbLoss(feature_dim=feature_dim, mode=mode, reduce=False)
        self.iou = IoU(eps=1e-6, reduce=False)
        self.ohem_batch = OHEM_BATCH(ohem_ratio=ohem_ratio)

    def forward(self, outputs, labels):
        predicts = outputs["maps"]
        # 上采样到原图尺寸计算损失（更准），也可以在1/4下采样尺寸计算（更快，TODO）
        predicts = F.interpolate(
            predicts, scale_factor=4, mode="nearest")  

        texts = predicts[:, 0, :, :]         # text region   bacth * H * W
        kernels = predicts[:, 1, :, :]       # kernel    bacth * H * W  (不是 batch * 1 * H * W)
        embs = predicts[:, 2:, :, :]         # instance vector   bacth * feature_dim * H * W
        gt_texts, gt_kernels, gt_instance, training_masks = labels[1:]

        # text loss
        selected_masks = self.ohem_batch.select_mask(
            texts, gt_texts, training_masks)
        loss_text = self.dice_loss(
            texts, gt_texts, selected_masks)
        iou_text = self.iou.cal_iou(
            (texts > 0).long(), gt_texts, training_masks)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        if self.kernel_sample_mask == "gt":
            selected_masks = gt_texts * training_masks
        elif self.kernel_sample_mask == "pred":
            selected_masks = (
                F.sigmoid(texts) > 0.5).float() * training_masks
        loss_kernels = self.dice_loss(
            kernels, gt_kernels, selected_masks)
        iou_kernel = self.iou.cal_iou(
            (kernels > 0).long(), gt_kernels, training_masks * gt_texts)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        # emb loss
        loss_emb = self.emb_loss(
            embs, gt_instance, gt_kernels, training_masks)
        losses.update(dict(loss_emb=loss_emb))

        loss = self.alpha * loss_text + self.beta * loss_kernels + self.gamma * loss_emb
        losses["loss"] = loss
        if self.reduction == "sum":
            losses = {x: v.sum() for x, v in losses.items()}
        elif self.reduction == "mean":
            losses = {x: v.mean() for x, v in losses.items()}
        return losses
