# !/usr/bin/python
# coding: utf8
from torch import nn
import torch
import numpy as np
from data_process import feature_extractor

bce_loss = nn.BCEWithLogitsLoss()
MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\\bp\model_dir\\bp1\\'


def loss_fn(predict, target):
    loss_f = bce_loss(predict.view(-1), target.view(-1))
    return loss_f


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


def train():
    model = BP_Net().cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    input_data, label_data = data_extractor.load_train_data()

    input_data_tensor = torch.FloatTensor(input_data).cuda(0)
    label_data_tensor = torch.FloatTensor(label_data).cuda(0)

    test_data, test_label_data = data_extractor.load_test_data()
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.FloatTensor(test_label_data).cuda(0)
    min_loss = 2
    L = len(input_data)
    for i in range(300):
        # loss_sum = 0

        prediction = model(input_data_tensor)
        loss = loss_fn(prediction, label_data_tensor)
        loss_sum = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        # for step in range(0, L):
        #     optimizer.zero_grad()
        #     x = input_data[step]
        #     y = label_data[step]
        #     x = np.array(x).reshape(1, 64)
        #     y = np.array(y).reshape(1, 64)
        #     x = torch.FloatTensor(x).cuda(0)
        #     y = torch.FloatTensor(y).cuda(0)
        #     prediction = model(x)
        #     loss = loss_fn(prediction, y)
        #     loss_sum = loss_sum + loss.data.cpu().numpy()
        #     loss.backward()
        #     optimizer.step()
        # print('end --------------------->: ', step)

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()
        # if loss_sum / L < 0.077:
        #     torch.save(model, MODEL_SAVE_DIR + 'BP_' + str(
        #         i) + '_new.pkl')
        if test_loss < 0.703:
            if test_loss < min_loss:
                min_loss = test_loss
            torch.save(model, MODEL_SAVE_DIR + 'BP_test_loss_' + str(
                i) + '_new.pkl')
        print(i, ' train_mean_loss: ', loss_sum / L, ' test_loss: ', test_loss)

    print('test_min_loss: ', min_loss)

    torch.save(model, 'BP1_data_with_empty_loss.pkl')


if __name__ == '__main__':
    train()
