import torch
from torch import nn
import torch.nn.functional as F


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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 16, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.se = SEModule(out_planes)
        if in_planes != out_planes:
            self.match = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.match = lambda x: x

    def forward(self, x):
        residual = self.match(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_class = 10):
        super(SENet, self).__init__()
        self.inplane = 64
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def senet18():
    return SENet(BasicBlock, [2, 2, 2, 2])


def senet34():
    return SENet(BasicBlock, [3, 4, 6, 3])


def test():
    model = senet34()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())


if __name__ == '__main__':
    test()
