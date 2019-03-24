from torch import nn
import torch
import numpy as np
from data_process import feature_extractor
import math


class Net(nn.Module):
    def __init__(self, INPUT_SIZEINPUT_SIZE):
        super(Net, self).__init__()
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

        # self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        self.fc = nn.Linear(64, 64)

    def forward(self, input_data):
        x = self.conv1(input_data)

        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = torch.load('D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_cnn1_pg_avg_sig\cnn3_pg440_nn.pkl')

input_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_avg_50_sig.npy')
label_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_avg_50_sig.npy')

# data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
# input_data, label_data = data_extractor.load_test_data()
L = len(input_data)

correct_num = 0
for step in range(len(input_data)):
    print('\n<----------------------------------------------------------', step)
    x = input_data[step:(step + 1)]
    y = label_data[step:(step + 1)]
    x = np.array(x).reshape(1, 1, 64)
    y = np.array(y).reshape(1, 1, 64)
    x = torch.FloatTensor(x).cuda(0)
    y = torch.ByteTensor(y).cuda(0)
    prediction = model(x)
    # print('prediction: ', prediction[0][0:6])
    _, max = torch.max(prediction.data, 1)
    index = 0
    # mm = prediction[0][0]
    # for i in range(1, len(prediction[0])):
    #     if prediction[0][i] > mm:
    #         index = i
    #         mm = prediction[0][i]
    #     # print('val: ', prediction[0][i])
    # predict = [j - j for j in range(0, 64)]
    # predict[index] = 1
    # predict = [predict]
    #
    # predict = prediction
    # predict = torch.ByteTensor(predict).cuda(0)
    predict = torch.sigmoid(prediction) > 0.5

    print('y:     ', y)
    print('predict:', predict)
    print(torch.eq(y, predict))
    result = torch.eq(y, predict)
    accuracy = torch.sum(result) / torch.sum(torch.ones(y.shape))
    accuracy = accuracy.data.cpu().numpy()
    correct = torch.eq(torch.sum(~result), 0)
    # if math.fabs(0.98437 - accuracy) > 0.0001:
    print('accuracy: ', accuracy)
    print('correct: {}'.format(correct))
    if correct == 1:
        correct_num = correct_num + 1

    print('-------------------------------------------------------------->\n')

print('total:', (step + 1), ' | correct_num:', correct_num, '| correct_percentage:', correct_num / (len(input_data)))

# print('prediction: ', prediction.data.cpu().numpy().flatten())
