import torch
import torch.nn.functional as F
from torch import nn


class SEModule(nn.Module):
    def __init__(self, in_planes, reduction = 16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size = 1)
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size = 1)

    def forward(self, x):
        w = self.avg_pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid((self.fc2(w)))
        return x * w


def test():
    model = SEModule(32)
    print(model)
    x = torch.randn(1, 32, 32, 32)
    y = model(x)
    print(y.size())


if __name__ == '__main__':
    test()
