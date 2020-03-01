import torch
from torch import nn


class SAModule(nn.Module):
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, padding = 1)

    def forward(self, x):
        f_avg = torch.mean(x, 1, keepdim = True)
        f_max, _ = torch.max(x, 1, keepdim = True)
        f = torch.cat([f_max, f_avg], dim = 1)
        f = torch.sigmoid(self.conv1(f))
        return x * f


class CAModule(nn.Module):
    def __init__(self, in_planes, reduction = 16):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size = 1),
        )

    def forward(self, x):
        f_avg = self.avg_pool(x)
        f_max = self.max_pool(x)
        fc = self.mlp(f_avg) + self.mlp(f_max)
        fc = torch.sigmoid(fc)
        return x * fc


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction = 16, spatial = True):
        super(CBAM, self).__init__()
        self.ca = CAModule(in_planes, reduction)
        self.spatial = spatial
        if spatial:
            self.sa = SAModule()

    def forward(self, x):
        x_out = self.ca(x)
        if self.spatial:
            x_out = self.sa(x_out)
        return x_out


# model = CBAM(32)
# print(model)
# x = torch.randn(1, 32, 32, 32)
# y = model(x)
# print(y.size())
