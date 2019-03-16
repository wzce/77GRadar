from torch import nn
import torch
import numpy as np
from data_process import feature_extractor
import math


class BP_Net(nn.Module):
    def __init__(self):
        super(BP_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(32, 64),
                                    nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(16, 64))
        self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return self.out(x)


model = torch.load('D:\home\zeewei\projects\\77GRadar\model\\bp\model_dir\\bp1\BP_test_loss_8_new.pkl')
# 定义优化器和损失函数
data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
h_state = None  # 第一次的时候，暂存为0
input_data, label_data = data_extractor.load_test_data()

L = len(input_data)
step = 0
correct_num = 0
for step in range(len(input_data)):
    print('\n\n<----------------------------------------------------------', step)
    x = input_data[step:(step + 1)]
    y = label_data[step:(step + 1)]
    x = np.array(x).reshape(1, 64)
    y = np.array(y).reshape(1, 64)
    x = torch.FloatTensor(x).cuda(0)
    y = torch.ByteTensor(y).cuda(0)
    prediction = model(x)
    # print('prediction: ', prediction[0][0:6])
    _, max = torch.max(prediction.data, 1)
    index = 0
    mm = prediction[0][0]
    for i in range(1, len(prediction[0])):
        if prediction[0][i] > mm:
            index = i
            mm = prediction[0][i]
        # print('val: ', prediction[0][i])
    predict = [j - j for j in range(0, 64)]
    predict[index] = 1
    predict = [predict]
    predict = torch.ByteTensor(predict).cuda(0)
    # predict = torch.sigmoid(prediction) > 0.5

    # predict = prediction
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
