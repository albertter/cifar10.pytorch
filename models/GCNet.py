import torch
from torch import nn
import torch.nn.functional as F


class GCBlock(nn.Module):
    def __init__(self, channels, ratio = 4):
        super(GCBlock, self).__init__()
        self.planes = channels
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
        w = self.spatial_pool(x)
        w = self.fc1(w)
        w = self.fc2(F.relu(self.LN(w)))
        return x + w
