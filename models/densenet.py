import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 16, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
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
        out += residual
        out = F.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace = False),
            nn.Conv2d(in_planes, growth_rate * bn_size, kernel_size = 1, stride = 1, bias = False),
        )
        if drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(growth_rate * bn_size),
            nn.ReLU(inplace = False),
            nn.Conv2d(growth_rate * bn_size, in_planes, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([out, x], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = self.maxpool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_blocks, num_classes = 10):
        super(DenseNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.denselayer1 = self._make_layer(DenseBlock, 64, num_blocks[0], stride = 1)
        self.trasitionlayer1 = TransitionLayer()
        self.denselayer2 = self._make_layer(DenseBlock, 128, num_blocks[1], stride = 2)
        self.denselayer3 = self._make_layer(DenseBlock, 256, num_blocks[2], stride = 2)
        self.denselayer4 = self._make_layer(DenseBlock, 512, num_blocks[3], stride = 2)
        self.fc = nn.Linear(512 * DenseBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(out.shape)
        out = F.avg_pool2d(out, 4)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return out


def densenet121():
    return DenseNet([6, 12, 24, 16])


def test():
    model = densenet121()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())


if __name__ == '__main__':
    test()
