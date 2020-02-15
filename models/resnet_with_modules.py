from .CBAM import *
from .SENet import *
from .GCNet import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1, add_module = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)

        if add_module == 'se':
            self.se = SEModule(self.expansion * planes, 16)
        else:
            self.se = None

        if add_module == 'cbam':
            self.cbam = CBAM(self.expansion * planes, 16)
        else:
            self.cbam = None

        if add_module == 'gc':
            self.gc = GCBlock(self.expansion * planes, 4)
        else:
            self.gc = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se:
            out = self.se(out)
        if self.cbam:
            out = self.cbam(out)
        if self.gc:
            out = self.gc(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride = 1, add_module = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if add_module == 'se':
            self.se = SEModule(self.expansion * planes, 16)
        else:
            self.se = None

        if add_module == 'cbam':
            self.cbam = CBAM(self.expansion * planes, 16)
        else:
            self.cbam = None

        if add_module == 'gc':
            self.gc = GCBlock(self.expansion * planes, 4)
        else:
            self.gc = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se:
            out = self.se(out)
        if self.cbam:
            out = self.cbam(out)
        if self.gc:
            out = self.gc(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, add_module = None):
        super(Resnet, self).__init__()
        assert add_module in ['se', 'cbam', 'gc', None]

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1, add_module = add_module)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2, add_module = add_module)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2, add_module = add_module)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2, add_module = add_module)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, add_module):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, add_module = add_module))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18(add_module):
    return Resnet(BasicBlock, [2, 2, 2, 2], add_module = add_module)


def resnet34(add_module):
    return Resnet(BasicBlock, [3, 4, 6, 3], add_module = add_module)


def resnet50(add_module):
    return Resnet(Bottleneck, [3, 4, 6, 3], add_module = add_module)


def resnet101(add_module):
    return Resnet(Bottleneck, [3, 4, 23, 3], add_module = add_module)


def resnet152(add_module):
    return Resnet(Bottleneck, [3, 8, 36, 3], add_module = add_module)


def test():
    model = resnet101('gc')
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())


if __name__ == '__main__':
    test()
