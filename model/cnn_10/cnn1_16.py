import torch
from torch import nn
import numpy as np
import random
from data_process import feature_extractor

from data_process import extractor


def decrease_range_resolution(list, n=2):
    res = []
    for i in range(0, len(list), n):
        v = 0
        for index in range(i, i + n):
            if list[index] == 1:
                v = 1
                break
        res.append(v)
    return res


class Net1(nn.Module):
    def __init__(self, INPUT_SIZEINPUT_SIZE):
        super(Net1, self).__init__()
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
            nn.MaxPool1d(kernel_size=4)
        )

        self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        # self.fc = nn.Linear(960, 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


bce_loss = nn.BCEWithLogitsLoss()


def loss_fn(predict, target):
    loss = bce_loss(predict.view(-1), target.view(-1))
    return loss


INPUT_SIZE = 1
LR = 1e-3
# batch_num = 100
batch_size = 7

MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn_10\model\cnn1\\'


def load_all_data():
    pg_data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    data_list = pg_data_extractor.load_data()
    data_list = data_list.tolist()
    random.shuffle(data_list)
    data_list = data_list[0:3000]

    road_data_extractor = feature_extractor.FeatureExtractor()
    input_data, label_data = road_data_extractor.load_train_data()
    input_data = input_data.tolist()
    label_data = label_data.tolist()
    for i in range(len(input_data)):
        a_group_data = []
        a_group_data.append(input_data[i])
        a_group_data.append(label_data[i])
        data_list.append(a_group_data)

    test_data, test_label_data = road_data_extractor.load_test_data()
    for i in range(len(test_data)):
        a_group_data = []
        a_group_data.append(test_data[i])
        a_group_data.append(test_label_data[i])
        data_list.append(a_group_data)
    return data_list


def train_playground():
    model = Net1(INPUT_SIZE).cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
    train_label_n = []
    for item in train_data_label:
        v = decrease_range_resolution(item, 4)
        train_label_n.append(v)

    test_label_n = []
    for item in test_data_label:
        v = decrease_range_resolution(item, 4)
        test_label_n.append(v)

    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn_10\\test_data\\input_data_cnn1_all_16.npy', test_data_input)
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn_10\\test_data\\label_data_cnn1_all_16.npy', test_label_n)

    test_batch_num = len(test_data_input)
    test_data_input = np.array(test_data_input).reshape(test_batch_num, 1, 64)
    test_data_label = np.array(test_label_n).reshape(test_batch_num, 1, 16)
    test_data_tensor = torch.FloatTensor(test_data_input).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label).cuda(0)

    train_data_input = np.array(train_data_input).reshape(batch_num, 1, 64)
    train_data_label = np.array(train_label_n).reshape(batch_num, 1, 16)
    train_data_tensor = torch.FloatTensor(train_data_input).cuda(0)
    train_label_tensor = torch.FloatTensor(train_data_label).cuda(0)

    min_loss = 2
    for i in range(3000):
        # loss_sum = 0
        # for step in range(int(data_len / 10)):
        optimizer.zero_grad()
        # x = input_data[step * 10:(step + 1) * 10]
        # y = label_data[step * 10:(step + 1) * 10]
        # x = np.array(x).reshape(10, 1, 64)
        # y = np.array(y).reshape(10, 1, 64)
        # x = torch.FloatTensor(x).cuda(0)
        # y = torch.FloatTensor(y).cuda(0)
        prediction = model(train_data_tensor)
        loss = loss_fn(prediction, train_label_tensor)
        loss_val = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()
        # if loss_sum / data_len < 0.077:
        #     torch.save(model, MODEL_SAVE_DIR + 'cnn3_' + str(i) + '_new.pkl')

        if i % 30 == 0:
            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < 0.701:
                torch.save(model, MODEL_SAVE_DIR + 'cnn1_1' + str(i) + '_70_16.pkl')
            print(i, ' train_mean_loss: ', loss_val,
                  ' test_loss: ', test_loss, 'min_loss: ', min_loss)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    train_playground()
