
from torch import nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.return_feats = return_feats

    def forward(self, x, **kwargs):
        T, b, h = x.shape
        t_rec = x.reshape(T * b, h)
        predicts = self.fc(t_rec)
        # 计算CTC-loss需要(T, batch_size, n_class)的形状
        predicts = predicts.reshape(T, b, -1)

        # 后处理转化为(batch_size, T, n_class)的形状 # 已在后处理中进行
        # predicts = predicts.permute(1, 0, 2) # batch_size * T * n_class 

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:  # when infer
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result