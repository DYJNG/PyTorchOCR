import torch
from torch import nn
import torch.nn.functional as F

from .det_basic_loss import DiceLoss, IoU, OHEM_BATCH


class PSELoss(nn.Module):
    def __init__(self,
                 alpha=0.7,
                 ohem_ratio=3,
                 kernel_sample_mask="pred",
                 reduction="mean",
                 **kwargs):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ["sum", "mean", "none"]
        self.alpha = alpha
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction

        self.dice_loss = DiceLoss(eps=0.001, act=True, reduce=False)
        self.iou = IoU(eps=1e-6, reduce=False)
        self.ohem_batch = OHEM_BATCH(ohem_ratio=ohem_ratio)

    def forward(self, outputs, labels):
        predicts = outputs["maps"]
        # 上采样到原图尺寸计算损失（更准），也可以在1/4下采样尺寸计算（更快，TODO）
        predicts = F.interpolate(
            predicts, scale_factor=4, mode="nearest")  

        texts = predicts[:, 0, :, :]         # 原始文本size的kernel
        kernels = predicts[:, 1:, :, :]      # 不同比例shrink的kernel
        gt_texts, gt_kernels, training_masks = labels[1:]

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
        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(
                kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = self.iou.cal_iou(
            (kernels[:, -1, :, :] > 0).long(), 
            gt_kernels[:, -1, :, :],
            training_masks * gt_texts)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernels
        losses["loss"] = loss
        if self.reduction == "sum":
            losses = {x: v.sum() for x, v in losses.items()}
        elif self.reduction == "mean":
            losses = {x: v.mean() for x, v in losses.items()}
        return losses
