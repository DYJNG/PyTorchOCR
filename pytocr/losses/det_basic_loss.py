import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BalanceLoss(nn.Module):
    def __init__(self,
                 balance_loss=True,
                 main_loss_type="BCELoss",
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            balance_loss (bool): whether balance loss or not, default is True
            main_loss_type (str): can only be one of ["CrossEntropy","DiceLoss",
                "Euclidean","BCELoss", "MaskL1Loss"], default is  "BCELoss".
            negative_ratio (int|float): float, default is 3.
            return_origin (bool): whether return unbalanced loss or not, default is False.
            eps (float): default is 1e-6.
        """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss(reduction="none") # reduction必须为none,否则无法进行难例采样
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss(reduction="none")
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
            self.balance_loss = False  # 如果使用dice-loss则返回是常量，没法进行loss排序和难负例采样
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction="none")
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps, reduction="none")
        else:
            loss_type = [
                "CrossEntropy", "DiceLoss", "Euclidean", "BCELoss", "MaskL1Loss"
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type))

    def forward(self, 
                pred: torch.Tensor, 
                gt: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        if self.main_loss_type in ["DiceLoss", "MaskL1Loss"]:
            loss = self.loss(pred, gt, mask=mask)
        else:
            loss = self.loss(pred, gt)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        # negative_loss = negative_loss.view(-1).contiguous()     # view()可能会导致内存不连续，所以需要.contiguous()；reshape则不存在该问题
        negative_loss = negative_loss.reshape(-1)                 # 计算topk需要先reshape为一维
        if negative_count > 0:
            # sort_loss = negative_loss.sort(descending=True)
            # negative_loss = sort_loss[:negative_count]
            negative_loss, _ = negative_loss.topk(negative_count)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    # 这里reduce是batch层面，因为dice是对map之间计算，返回常数
    def __init__(self, eps=1e-6, act=False, reduce=True):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.act = act
        self.reduce = reduce

    def forward(self, pred, gt, mask, weights=None, **kwargs):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        if self.act:
            pred = F.sigmoid(pred)  # logit -> [0-1]
        if self.reduce:  # DB 里的实现
            intersection = (pred * gt * mask).sum()

            union = (pred * mask).sum() + (gt * mask).sum() + self.eps
            loss = 1 - 2.0 * intersection / union
            assert loss <= 1
        else:  # PSE PAN PAN++ 里的实现
            batch_size = pred.shape[0]
            pred = pred.reshape([batch_size, -1])
            gt = gt.reshape([batch_size, -1]).float()
            mask = mask.reshape([batch_size, -1]).float()

            pred = pred * mask
            gt = gt * mask

            a = torch.sum(pred * gt, dim=1)
            b = torch.sum(pred * pred, dim=1) + self.eps
            c = torch.sum(gt * gt, dim=1) + self.eps
            d = (2 * a) / (b + c)
            loss = 1 - d

        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super(MaskL1Loss, self).__init__()
        self.eps = eps
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def forward(self, pred, gt, mask, **kwargs):
        """
        Mask L1 Loss
        """
        if self.reduction == "sum":
            loss = (torch.abs(pred - gt) * mask).sum()
        elif self.reduction == "mean":
            # loss = (torch.abs(pred - gt) * mask).mean() # 若矩阵为空，返回nan
            loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        else:
            loss = torch.abs(pred - gt) * mask
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def forward(self, input, label, **kwargs):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class IoU(object):
    # 这里reduce是batch层面，因为iou是对map之间计算，返回常数
    def __init__(self, eps=1e-6, reduce=True): 
        super(IoU, self).__init__()
        self.eps = eps
        self.reduce = reduce
    
    def iou_single(self, a, b, mask, n_class):
        valid = mask == 1
        a = a[valid]
        b = b[valid]
        miou = []
        for i in range(n_class):
            inter = ((a == i) & (b == i)).float()
            union = ((a == i) | (b == i)).float()

            miou.append(torch.sum(inter) / (torch.sum(union) + self.eps))
        miou = sum(miou) / len(miou)
        return miou

    def cal_iou(self, a, b, mask, n_class=2):
        batch_size = a.shape[0]

        a = a.view(batch_size, -1)
        b = b.view(batch_size, -1)
        mask = mask.view(batch_size, -1)

        iou = a.new_zeros((batch_size, ), dtype=torch.float32)
        for i in range(batch_size):
            iou[i] = self.iou_single(a[i], b[i], mask[i], n_class)

        if self.reduce:
            iou = torch.mean(iou)

        return iou


class OHEM_BATCH(object):
    def __init__(self, ohem_ratio=3):
        super(OHEM_BATCH, self).__init__()
        self.ohem_ratio = ohem_ratio

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = int(torch.sum(gt_text > 0.5)) - int(
            torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))  # TP数

        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], 
                                                  selected_mask.shape[1]).float()
            return selected_mask

        neg_num = int(torch.sum(gt_text <= 0.5))
        neg_num = int(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0],
                                                  selected_mask.shape[1]).float()
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted, _ = torch.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        selected_mask = ((score >= threshold) |
                         (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0],
                                              selected_mask.shape[1]).float()
        return selected_mask

    def select_mask(self, scores, gt_texts, training_masks):
        """OHEM sampling for a batch of imgs.

        Args:
            scores (Tensor): The text scores of size NxHxW.
            gt_texts (Tensor): The gt text masks of size NxHxW.
            training_masks (Tensor): The gt effective mask of size NxHxW.  
                                     ignore the ignore text region
        Returns:
            selected_masks (Tensor): The sampled mask of size NxHxW.
        """
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(
                self.ohem_single(scores[i, :, :], gt_texts[i, :, :],
                                 training_masks[i, :, :]))

        selected_masks = torch.cat(selected_masks, dim=0).float()
        return selected_masks


class EmbLoss(nn.Module):
    def __init__(self, feature_dim=4, mode="v2", reduce=True):
        super(EmbLoss, self).__init__()
        self.feature_dim = feature_dim
        assert mode in [
            "v1", "v2"], "EmbLoss mode only support [v1, v2]"
        self.mode = mode
        self.reduce = reduce
        self.delta_v = 0.5    # 类内距离阈值
        self.delta_d = 1.5    # 聚类中心和背景，不同聚类中心间的类间距离阈值 2*delta_d=3，经验值
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).reshape(-1)  # (H*W)
        instance = instance.reshape(-1)  # (H*W)
        emb = emb.reshape(self.feature_dim, -1)  #  dim * (H*W)

        # 每一个文本行都是一个聚类中心
        unique_labels, unique_ids = torch.unique(instance_kernel,
                                                 sorted=True,
                                                 return_inverse=True)
        num_instance = unique_labels.shape[0]
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance),
                                 dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1) # 聚类中心

        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0) # 类内距离
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(
                -1, self.feature_dim)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).reshape(
                -1, 1).repeat(1, self.feature_dim)
            mask = mask.reshape(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.reshape(num_instance * num_instance, -1)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].reshape(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_d - dist) ** 2
            if self.mode == "v1":    # PAN  不同聚类中心间的距离要大
                l_dis = torch.mean(torch.log(dist + 1.0))
            else:  # "v2" PAN++   不同聚类中心间的距离 、 聚类中心和背景的距离 都要大
                l_dis = [torch.log(dist + 1.0)]
                emb_bg = emb[:, instance == 0].reshape(self.feature_dim, -1)
                if emb_bg.shape[1] > 100:
                    rand_ind = np.random.permutation(emb_bg.shape[1])[:100]
                    emb_bg = emb_bg[:, rand_ind]
                if emb_bg.shape[1] > 0:
                    for i, lb in enumerate(unique_labels):
                        if lb == 0:
                            continue
                        dist = (emb_bg - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
                        dist = F.relu(2 * self.delta_d - dist) ** 2
                        l_dis_bg = torch.mean(
                            torch.log(dist + 1.0), 0, keepdim=True)
                        l_dis.append(l_dis_bg)
                l_dis = torch.mean(torch.cat(l_dis))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self,
                emb,
                instance,
                kernel,
                training_mask):
        loss_batch = emb.new_zeros((emb.shape[0]), dtype=torch.float32)

        for i in range(loss_batch.shape[0]):
            loss_batch[i] = self.forward_single(
                emb[i], instance[i], kernel[i], training_mask[i])

        if self.reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch
