import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, stride = 1, cardinality = 32, bottleneck_width = 4):
        super(Bottleneck, self).__init__()
        width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size = 3, groups = cardinality, stride = stride, padding = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, self.expansion * width, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * width, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnext(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, cardinality = 32, bottleneck_width = 4):
        super(Resnext, self).__init__()
        self.in_planes = 64
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, num_blocks[3], stride = 2)
        self.fc = nn.Linear(cardinality * bottleneck_width * block.expansion * 8, num_classes)

    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, stride, self.cardinality, self.bottleneck_width))
            self.in_planes = self.cardinality * self.bottleneck_width * block.expansion
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        print(out.shape)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnext_32_4d():
    return Resnext(Bottleneck, [3, 4, 6, 3], cardinality = 32, bottleneck_width = 4)


if __name__ == '__main__':
    model = resnext_32_4d()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
