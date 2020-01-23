import torch
from torch import nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size = 3, stride = stride, padding = 1, groups = in_planes,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    def __init__(self, num_class = 10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace = True)

        self.layer = nn.Sequential(
            Layer(32, 64, 1),
            Layer(64, 128, 2),
            Layer(128, 128, 1),
            Layer(128, 256, 2),
            Layer(256, 256, 1),
            Layer(256, 512, 2),

            Layer(512, 512, 1),
            Layer(512, 512, 1),
            Layer(512, 512, 1),
            Layer(512, 512, 1),
            Layer(512, 512, 1),
            #
            Layer(512, 1024, 2),
            Layer(1024, 1024, 1),

        )
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape(-1, 1024)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = MobileNet()
    print(model(x).shape)
