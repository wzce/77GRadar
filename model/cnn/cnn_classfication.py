import torch
from torch import nn
import numpy as np
from data_process import classification_data_extractor

MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_classification\\'


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
            # nn.MaxPool1d(kernel_size=4)
        )

        # self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个
        self.fc = nn.Linear(64, 2)

    def forward(self, input_data):
        x = self.conv1d(input_data)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


loss_func = torch.nn.CrossEntropyLoss()
LR = 1e-4


def train():
    model = Net().cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    extractor = classification_data_extractor.ClassificationExtractor()  # 此处全使用默认的文件路径配置,获取有目标数据和无目标数据

    car_data = extractor.load_car_data()
    empty_data = extractor.load_empty_data()

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

    train_len = len(train_data)
    test_len = len(test_data)
    test_batch_num = test_len
    # test_data = test_data[0: test_batch_num * 64]
    # test_label_data = test_label_data[0:test_batch_num * 64]
    test_data = np.array(test_data).reshape(test_batch_num, 1, 64)
    # test_label_data = np.array(test_label_data).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.LongTensor(test_label).cuda(0)

    min_loss = 2
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
            if test_loss < 0.200:
                if test_loss < min_loss:
                    min_loss = test_loss
                torch.save(model, MODEL_SAVE_DIR + 'cnn_classification_3_' + str(i) + '_new.pkl')
            print(i, ' train_mean_loss: ', loss.data.cpu().numpy(), ' test_loss: ', test_loss)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    train()
