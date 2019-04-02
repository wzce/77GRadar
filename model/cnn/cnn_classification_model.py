import torch
from torch import nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=6,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=4)
        )

        # self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        self.fc = nn.Linear(11, 2)

    def forward(self, input_data):
        x = self.conv1d(input_data)
        x = x.view(x.size(0), -1)
        return self.fc(x)
