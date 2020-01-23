import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, groups = planes,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_class = 10):
        super(MobileNetV2, self).__init__()
        self.inverted_residual_setting = [
            # t, c, n, s
            # (expansion, out_planes, num_blocks, stride)
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # s=1
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace = True)

        self.layer = self._make_layers(32)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_class)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.inverted_residual_setting:
            strides = [stride] + [1] * (num_blocks - 1)
            # print(strides)
            for stride in strides:
                # print(stride)
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
            # print(in_planes)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.layer(out)
        print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.reshape(-1, 1280)
        out = self.fc(out)
        return out


x = torch.randn(1, 3, 32, 32)
model = MobileNetV2()
# model = BottleNeck(32, 64, 2, 6)
print(model(x).shape)
