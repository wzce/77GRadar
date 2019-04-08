import torch
from torch import nn


class RadarCnn1(nn.Module):
    '''
        三个卷积层加一个softmax，不使用池化和全连接
        训练结果，total: 3114：
        | correct_num: 1310
        | complete_correct_rate: 0.4206807964033398
        | relative_correct_num :  1547
        | relative_correct_rate:  0.4967886962106615
    '''

    def __init__(self):
        super(RadarCnn1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )

        self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        # self.fc = nn.Linear(960, 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class RadarCnn1_1(nn.Module):
    '''
        三个卷积层加一个全连接层，不使用池化和全连接
        训练结果，total: 3114：
        | correct_num: 1310
        | complete_correct_rate: 0.4206807964033398
        | relative_correct_num :  1547
        | relative_correct_rate:  0.4967886962106615
    '''

    def __init__(self):
        super(RadarCnn1_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )

        self.out = nn.Linear(64, 64)  # 分类器，预测位置最大的一个
        # self.fc = nn.Linear(960, 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


'''
   三个卷积层， 添加一个全连接层--->softmax
   卷积核大小为3
'''


class RadarCnn2(nn.Module):
    def __init__(self):
        super(RadarCnn2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=3)
        )
        self.fc1 = nn.Linear(20, 64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=6,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc2 = nn.Linear(30, 64)
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.out = nn.Sequential(nn.Linear(31, 64), nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class Radar_Cnn_2_1(nn.Module):
    def __init__(self):
        super(Radar_Cnn_2_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=3)
        )
        self.fc1 = nn.Linear(20, 64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=6,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc2 = nn.Linear(30, 64)
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.out = nn.Sequential(nn.Linear(31, 64))  # 分类器，预测位置最大的一个

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class RadarCnn3(nn.Module):
    '''
       三个卷积层，每层都使用池化， 添加一个全连接层--->softmax
       卷积核大小为3
    '''

    def __init__(self):
        super(RadarCnn3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.out = nn.Sequential(nn.Linear(16, 32),
                                 nn.ReLU(True),
                                 nn.Linear(32, 64),
                                 nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class RadarCnn4(nn.Module):
    def __init__(self):
        super(RadarCnn4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=3)
        )
        self.fc1 = nn.Linear(31, 64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=6,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc2 = nn.Linear(61, 64)
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2),

        )
        self.fc3 = nn.Linear(62, 64)
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc4 = nn.Linear(62, 64)
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=5,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2),
        )
        self.out = nn.Sequential(nn.Linear(61, 64), nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        x = self.conv4(x)
        x = self.fc4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
