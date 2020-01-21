import torch
from torch import nn
import torch.nn.functional as F


class GCBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super(GCBasicBlock, self).__init__()
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


class GCBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ratio = 4, stride = 1):
        super(GCBottleneck, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.conv1x1 = nn.Conv2d(self.planes, 1, kernel_size = 1)
        self.softmax = nn.Softmax(dim = 2)
        self.fc1 = nn.Conv2d(self.planes, self.planes // ratio, kernel_size = 1)
        self.LN = nn.LayerNorm([self.planes // ratio, 1, 1])
        self.fc2 = nn.Conv2d(self.planes // ratio, self.planes, kernel_size = 1)

    def spatial_pool(self, x):
        N, C, H, W = x.size()
        input_x = x
        input_x = input_x.view(N, C, H * W)
        input_x = input_x.unsqueeze(1)
        w = self.conv1x1(x)
        w = w.view(N, 1, H * W)
        w = self.softmax(w)
        w = w.unsqueeze(3)

        context = torch.matmul(input_x, w)
        context = context.permute(0, 2, 1, 3)
        # context = context.reshape(N, C, 1, 1)
        return context

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # print(out.shape)
        w = self.spatial_pool(out)
        # print(w.shape)
        w = self.fc1(w)
        # print(w.shape)

        w = self.fc2(F.relu(self.LN(w)))
        # print(w.shape)

        out = out + w
        # print(out.shape)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GCNet(nn.Module):
    def __init__(self, block, num_blocks, num_class = 10):
        super(GCNet, self).__init__()
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


# def senet18():
#     return GCNet(GCBasicBlock, [2, 2, 2, 2])
#
#
# def senet34():
#     return GCNet(GCBasicBlock, [3, 4, 6, 3])

def gc_resnet_50():
    return GCNet(GCBottleneck, [3, 4, 6, 3])


def test():
    model = gc_resnet_50()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())


if __name__ == '__main__':
    test()
