from torch import nn


class PANHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=6, **kwargs):
        super(PANHead, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        return {"maps": out}

