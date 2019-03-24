import torch
from torch import nn
import numpy as np
import random
from data_process import extractor
from data_process import empty_extractor

from data_process import classification_data_extractor

MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_classification_pg\\'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                padding=1
            )
        )

        # self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        self.fc = nn.Linear(64, 2)

    def forward(self, input_data):
        x = self.conv1d(input_data)
        x = x.view(x.size(0), -1)
        return self.fc(x)


loss_func = torch.nn.CrossEntropyLoss()
LR = 1e-3


def load_data(type=1):
    empty_extc = empty_extractor.PGEmptyFeatureExtractor()
    empty_data = empty_extc.load_data()
    empty_data = empty_data.tolist()
    # print('empty_data: ', empty_data)

    # 有目标的类别
    data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
    car_data_list = data_extractor.load_data()
    # car_data_list = random.shuffle(car_data_list)
    input_data = []

    # car_label = []
    for item in car_data_list:
        input_data.append(item[0])  # 强度信息
    random.shuffle(input_data)
    car_data = input_data[0:5490]

    # 存储起来用来做验证的
    pg_car_data = input_data[5490:]
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_classification_pg_car_data.npy',
            pg_car_data)
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_classification_pg_empty_data.npy',
            empty_data[300:])

    if type != 1:
        random.shuffle(car_data)
        car_data = car_data[0:300]
        road_extractor = classification_data_extractor.ClassificationExtractor()
        road_car_data = road_extractor.load_car_data()
        random.shuffle(road_car_data)
        road_car_data = road_car_data[0:300]
        for item in road_car_data:
            car_data.append(item)
        road_empty_data = road_extractor.load_empty_data()
        print('road_empty_data: ', len(road_empty_data))
        road_empty_data = road_empty_data
        # random.shuffle(road_empty_data)

        # empty_data = np.concatenate(empty_data, road_empty_data)

        empty_data = empty_data[0:300]
        for item in road_empty_data:
            empty_data.append(item)

    return car_data, empty_data


def train():
    model = Net().cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # extractor = classification_data_extractor.ClassificationExtractor()  # 此处全使用默认的文件路径配置,获取有目标数据和无目标数据

    # car_data = extractor.load_car_data()
    # random.shuffle(car_data)
    car_data, empty_data = load_data(2)
    random.shuffle(empty_data)
    random.shuffle(car_data)

    car_train_data_len = int(2 * len(car_data) / 3)
    empty_train_data_len = int(2 * len(empty_data) / 3)
    train_data = car_data[0:car_train_data_len]
    empty_train_data = empty_data[0:empty_train_data_len]
    train_label = [1 for i in range(len(train_data))]
    for item in empty_train_data:
        train_data.append(item)
        train_label.append(0)

    test_data = car_data[car_train_data_len:len(car_data)]
    test_label = [1 for i in range(len(test_data))]
    empty_test_data = empty_data[empty_train_data_len:len(empty_data)]
    for item in empty_test_data:
        test_data.append(item)
        test_label.append(0)

    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_classification_all.npy', test_data)
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_classification_all.npy', test_label)

    train_len = len(train_data)
    test_len = len(test_data)
    test_batch_num = test_len
    test_data = np.array(test_data).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.LongTensor(test_label).cuda(0)

    min_loss = 2
    for i in range(3000):
        optimizer.zero_grad()
        x = np.array(train_data).reshape(train_len, 1, 64)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.LongTensor(train_label).cuda(0)
        prediction = model(x)
        loss = loss_func(prediction, y)
        loss.backward()
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_func(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()
        if i % 30 == 0:
            if test_loss < min_loss:
                min_loss = test_loss

            if test_loss < 0.08:
                torch.save(model, MODEL_SAVE_DIR + 'cnn_classification_3_' + str(i) + '_all.pkl')
            print(i, ' train_mean_loss: ', loss.data.cpu().numpy(),
                  ' test_loss: ', test_loss, 'min_loss: ', min_loss)
    print('test_min_loss: ', min_loss)


def train_with_pg_test_with_road():
    model = Net().cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # extractor = classification_data_extractor.ClassificationExtractor()  # 此处全使用默认的文件路径配置,获取有目标数据和无目标数据

    # car_data = extractor.load_car_data()
    # random.shuffle(car_data)
    car_data, empty_data = load_data(1)  # 获取操场数据作为训练
    random.shuffle(empty_data)
    random.shuffle(car_data)

    train_data = car_data[0:5490]
    empty_train_data = empty_data
    train_label = [1 for i in range(len(train_data))]
    for item in empty_train_data:
        train_data.append(item)
        train_label.append(0)

    ''' 下面获取道路数据作为测试集数据'''
    road_extractor = classification_data_extractor.ClassificationExtractor()
    road_car_data = road_extractor.load_car_data()
    random.shuffle(road_car_data)
    road_car_data = road_car_data[0:300]
    # for item in road_car_data:
    #     car_data.append(item)
    road_empty_data = road_extractor.load_empty_data()
    print('road_empty_data: ', len(road_empty_data))

    test_data = road_car_data
    test_label = [1 for i in range(len(test_data))]
    # empty_test_data = empty_data[empty_train_data_len:len(empty_data)]
    for item in road_empty_data:
        test_data.append(item)
        test_label.append(0)

    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_classification_road.npy', test_data)
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_classification_road.npy', test_label)

    train_len = len(train_data)
    test_len = len(test_data)
    test_batch_num = test_len
    test_data = np.array(test_data).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.LongTensor(test_label).cuda(0)

    min_loss = 100
    for i in range(130000):
        optimizer.zero_grad()
        x = np.array(train_data).reshape(train_len, 1, 64)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.LongTensor(train_label).cuda(0)
        prediction = model(x)
        loss = loss_func(prediction, y)
        loss.backward()
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_func(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()
        if i % 50 == 0:
            if test_loss < min_loss:
                min_loss = test_loss

            if test_loss < 5:
                torch.save(model, MODEL_SAVE_DIR + 'cnn_classification_3_' + str(i) + '_all_300.pkl')
            print(i, ' train_mean_loss: ', loss.data.cpu().numpy(),
                  ' test_loss: ', test_loss, 'min_loss: ', min_loss)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    train()
