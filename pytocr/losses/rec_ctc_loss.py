import torch
from torch import nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    def __init__(self, zero_infinity=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=zero_infinity)

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        T, b, _ = predicts.shape  # T * batch_size * n_class
        preds_lengths = torch.tensor([T] * b, dtype=torch.long) # warp-ctc:dtype=torch.long
        labels = batch[1]
        label_lengths = batch[2]
        predicts = F.log_softmax(predicts, dim=2)   # log_softmax
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        return {"loss": loss}
