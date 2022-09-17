from torch import nn


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 1, "the height of backbone output featuremap must be 1"
        x = x.squeeze(dim=2)  # N C W(T)
        # 计算CTC-loss需要(T, batch_size, n_class)的形状
        x = x.permute(2, 0, 1) # W(T) N C (width, batch, channels)  # TODO: 可改为 N T C
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out=None):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.out_channels = n_hidden * 2
        if n_out is not None:
            self.embedding = nn.Linear(n_hidden * 2, n_out)
            self.out_channels = n_out
        self.n_out = n_out

    def forward(self, input):
        output, _ = self.rnn(input)
        if self.n_out is not None:
            T, b, h = output.shape
            t_rec = output.reshape(T * b, h)
            output = self.embedding(t_rec)  # [T * b, nOut]
            output = output.reshape(T, b, -1)
        return output

class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn = nn.Sequential(
            BidirectionalLSTM(in_channels, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size))

    def forward(self, x):
        x = self.rnn(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        T, b, h = x.shape
        t_rec = x.reshape(T * b, h)
        x = self.fc(t_rec)
        x = x.reshape(T, b, -1)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=256, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, "{} must in {}".format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x
