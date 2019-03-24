# !/usr/bin/python
# coding: utf8
from torch import nn
import torch
import numpy as np
from data_process import feature_extractor
from data_process import extractor

import random

bce_loss = nn.BCEWithLogitsLoss()
MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\\bp\model_dir\\bp3\\'


def loss_fn(predict, target):
    loss_f = bce_loss(predict.view(-1), target.view(-1))
    return loss_f


class BP_Net(nn.Module):
    def __init__(self):
        super(BP_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(32, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 32),
                                    nn.ReLU(),
                                    # nn.Linear(32, 64),
                                    # nn.ReLU(True),

                                    nn.Linear(32, 32))
        self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        return self.out(x)


def decrease_range_resolution(list, n=2):
    res = []
    for i in range(0, len(list), n):
        v = 0
        for index in range(i, i + n):
            v = v + list[index]
        res.append(v)
    return res


def train():
    model = BP_Net().cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)

    # data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    data_list = data_extractor.load_data()
    # data_list = load_all_data()  # 道路操场数据混合
    print('the length of data_list : ', len(data_list))

    random.shuffle(data_list)
    batch_num = int(7 * len(data_list) / 10)
    train_data = data_list[0:batch_num]
    test_data = data_list[batch_num:]

    train_data_input = []
    train_data_label = []
    for item in train_data:
        train_data_input.append(item[0])
        train_data_label.append(item[1])

    test_data_input = []
    test_data_label = []
    for item in test_data:
        test_data_input.append(item[0])
        test_data_label.append(item[1])

    '''将距离分辨率改为6m'''
    train_data_input_n = []
    train_data_label_n = []
    for i in range(len(train_data_label)):
        input = decrease_range_resolution(train_data_input[i])
        label = decrease_range_resolution(train_data_label[i])
        train_data_input_n.append(input)
        train_data_label_n.append(label)

    test_data_input_n = []
    test_data_label_n = []
    for i in range(len(test_data_label)):
        input = decrease_range_resolution(test_data_input[i])
        label = decrease_range_resolution(test_data_label[i])
        test_data_input_n.append(input)
        test_data_label_n.append(label)

    input_data_tensor = torch.FloatTensor(train_data_input_n).cuda(0)
    label_data_tensor = torch.FloatTensor(train_data_label_n).cuda(0)

    # test_data, test_label_data = data_extractor.load_test_data()
    test_data_tensor = torch.FloatTensor(test_data_input_n).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label_n).cuda(0)

    np.save('D:\home\zeewei\projects\\77GRadar\model\\bp\\test_data\\bp3.npy', test_data_input_n)
    np.save('D:\home\zeewei\projects\\77GRadar\model\\bp\\test_data\\bp3.npy', test_data_label_n)

    min_loss = 200
    # L = len(input_data)
    for i in range(300000):
        # loss_sum = 0

        prediction = model(input_data_tensor)
        loss = loss_fn(prediction, label_data_tensor)
        loss_sum = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if i % 50 == 0:

            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < 0.711:
                torch.save(model, MODEL_SAVE_DIR + 'bp2_' + str(
                    i) + '.pkl')
            print(i, ' train_mean_loss: ', loss_sum, ' test_loss: ', test_loss, ' min_test_loss: ', min_loss)

    print('test_min_loss: ', min_loss)

    torch.save(model, 'BP2_data_with_empty_loss.pkl')


if __name__ == '__main__':
    train()
