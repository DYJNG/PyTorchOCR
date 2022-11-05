import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SLALoss(nn.Module):
    def __init__(
        self, 
        structure_weight, 
        loc_weight, 
        loc_loss_type="mse", 
        **kwargs
    ):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        supported_name = ["mse", "smooth_l1"]
        assert loc_loss_type in supported_name, \
            "supported loc_loss_type are {} but input loc_loss_type is {}".format(
                supported_name, loc_loss_type)
        self.loc_loss_type = loc_loss_type
        self.eps = 1e-12
    
    def forward(self, predicts, batch):
        structure_probs = predicts["structure_probs"]
        _, _, C = structure_probs.shape
        structure_probs = structure_probs.reshape(-1, C)  # 为了计算交叉熵损失
        structure_targets = batch[1].long()
        structure_targets = structure_targets[:, 1:]
        structure_targets = structure_targets.reshape(-1) # 为了计算交叉熵损失

        structure_loss = self.loss_func(structure_probs, structure_targets)
        structure_loss = torch.mean(structure_loss) * self.structure_weight

        loc_preds = predicts["loc_preds"]
        loc_targets = batch[2].float()
        loc_targets_mask = batch[3].float()
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]

        if self.loc_loss_type == "smooth_l1":
            loc_loss = F.smooth_l1_loss(
                loc_preds * loc_targets_mask, 
                loc_targets * loc_targets_mask, 
                reduction="sum") * self.loc_weight
        else:
            loc_loss = F.mse_loss(
                loc_preds * loc_targets_mask, 
                loc_targets * loc_targets_mask, 
                reduction="sum") * self.loc_weight
        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
        
        total_loss = structure_loss + loc_loss
        return {
            "loss": total_loss, 
            "structure_loss": structure_loss, 
            "loc_loss": loc_loss
        }
        