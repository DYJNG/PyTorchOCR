import torch
from torch import nn
import torch.nn.functional as F


class AttentionCell(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_embeddings, 
        use_gru=False
    ):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size+num_embeddings, 
                hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size+num_embeddings, 
                hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), axis=1)
        concat_context = torch.cat((context, char_onehots), dim=1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class SLAHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_size, 
        out_channels=30, 
        max_text_length=500, 
        loc_reg_num=4, 
        use_gru=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels

        # structure
        self.structure_attention_cell = AttentionCell(
            in_channels, 
            hidden_size, 
            num_embeddings=self.num_embeddings, 
            use_gru=use_gru
        )
        self.structure_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.Linear(hidden_size, out_channels)
        )

        # loc
        self.loc_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.Linear(hidden_size, loc_reg_num), 
            nn.Sigmoid()
        )

    def forward(self, x, targets=None):
        batch_size = x.shape[0]
        # reshape
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)  # N * T * C  (batch, width, channels)
        hidden = torch.zeros((batch_size, self.hidden_size), device=x.device)

        structure_preds = []
        loc_preds = []
        if self.training and targets is not None:
            structure = targets[0]
            for i in range(self.max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(
                    structure[:, i], x, hidden)
                structure_preds.append(structure_step)
                loc_preds.append(loc_step)
        else:
            pre_chars = torch.zeros(
                (batch_size), dtype=torch.int64, device=x.device)  # int64
            # for export
            loc_step, structure_step = None, None
            for i in range(self.max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(
                    pre_chars, x, hidden)
                pre_chars = torch.argmax(structure_step, dim=1)  # int64
                structure_preds.append(structure_step)
                loc_preds.append(loc_step)
        structure_preds = torch.stack(structure_preds, dim=1)
        loc_preds = torch.stack(loc_preds, dim=1)
        if not self.training:
            structure_preds = F.softmax(structure_preds, dim=-1)
        return {"structure_probs": structure_preds, "loc_preds": loc_preds}

    def _char_to_onehot(self, input_char):
        input_one_hot = F.one_hot(
            input_char, num_classes=self.num_embeddings)
        return input_one_hot
    
    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        # output shape is b * self.hidden_size
        output, alpha = self.structure_attention_cell(
            hidden, features, emb_feature)
        hidden = output

        # structure
        structure_step = self.structure_generator(output)
        # loc
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step
