import torch
from torch import nn
import torch.nn.functional as F


class SKConv(nn.Module):
    def __init__(self, features, M = 2, G = 32, r = 16, stride = 1, L = 32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size = 3 + i * 2, stride = stride, padding = 1 + i, groups = G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace = False)
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim = 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim = 1)
        fea_U = torch.sum(feas, dim = 1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim = 1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim = 1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim = 1)
        return fea_v


class SKunit(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride = 1, L = 32):
        super(SKunit, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SKConv(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

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
        print(out.shape)

        out = self.layer3(out)
        print(out.shape)

        out = self.layer4(out)
        print(out.shape)
        out = F.avg_pool2d(out, 4)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return out


def resnext_32_4d():
    return Resnext(SKunit, [3, 4, 6, 3], cardinality = 32, bottleneck_width = 4)


if __name__ == '__main__':
    model = resnext_32_4d()
    model2 = SKConv()
    x = torch.randn(1, 3, 32, 32)
    # y = model(x)
    y = model2(x)
    print(y.shape)
