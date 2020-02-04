import torch
from torch import nn
import torch.nn.functional as F


class hswish(nn.Module):
    def __init__(self):
        super(hswish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class SeModule(nn.Module):
    def __init__(self, in_size, reduction = 4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, groups = planes,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2)
        self.hs1 = hswish
        self.setting = [
            # kernal_size,exp_size, out_planes, se,nl,stride
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'HS', 2],
            [3, 200, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 480, 112, True, 'HS', 1],
            [3, 672, 112, True, 'HS', 1],
            [5, 672, 160, True, 'HS', 2],
            [5, 960, 160, True, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],

        ]
        self.conv2 = nn.Conv2d(160, 960, kernel_size = 1, padding = 0)
        self.hs2 = hswish
        self.maxpool = nn.MaxPool2d(kernel_size = 7)
        self.conv3 = nn.Conv2d(960, 1280, kernel_size = 1, padding = 0)
        self.conv3 = nn.Conv2d(1280, num_classes, kernel_size = 1, padding = 0)

    def forward(self, x):
        out = self.hs1(self.conv1(x))
        return out


x = torch.randn(1, 3, 32, 32)
model = MobileNetV3Large()
y=model(x)
# model = BottleNeck(32, 64, 2, 6)
print(y.shape)

