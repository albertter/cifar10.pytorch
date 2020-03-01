import torch
from torch import nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction = 16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channel, gate_channel // reduction, kernel_size = 1),
            nn.BatchNorm2d(gate_channel // reduction),
            nn.ReLU(inplace = True),
            nn.Conv2d(gate_channel // reduction, gate_channel, kernel_size = 1),
        )
        self.bn = nn.BatchNorm2d(gate_channel)

    def forward(self, x):
        out = self.mlp(self.avg_pool(x))
        return self.bn(out)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction = 16, dilation_val = 4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(gate_channel, gate_channel // reduction, kernel_size = 1),
            nn.BatchNorm2d(gate_channel // reduction),
            nn.ReLU(),

            nn.Conv2d(gate_channel // reduction, gate_channel // reduction, kernel_size = 3, padding = dilation_val,
                      dilation = dilation_val),
            nn.BatchNorm2d(gate_channel // reduction),
            nn.ReLU(),

            nn.Conv2d(gate_channel // reduction, gate_channel // reduction, kernel_size = 3, padding = dilation_val,
                      dilation = dilation_val),
            nn.BatchNorm2d(gate_channel // reduction),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(gate_channel // reduction, 1, kernel_size = 1)

    def forward(self, x):
        return self.final(self.gate_s(x))


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, x):
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x


if __name__ == '__main__':
    model = BAM(32)
    print(model)
    x = torch.randn(2, 32, 32, 32)
    y = model(x)
    print(y.size())
