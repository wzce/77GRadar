from torch import nn
import torch

class BP_Net1(nn.Module):
    def __init__(self):
        super(BP_Net1, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(32, 32),
        #                             nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(32, 64))
        # self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        return x
        # return self.out(x)