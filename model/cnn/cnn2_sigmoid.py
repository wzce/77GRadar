import torch
from torch import nn
import numpy as np
import random
from torch.autograd import Variable
# from data_process import feature_extractor

from data_process import extractor


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


bce_loss = nn.BCEWithLogitsLoss()


def random_index(start, end, exclude_list, l):
    res = []
    while len(res) < l:
        rd = random.randint(start, end)
        if (rd in exclude_list) or (rd in res):
            continue
        else:
            res.append(rd)
    return res


def find_val_index(list, val):
    res = []
    for i in range(len(list)):
        if list[i] == val:
            res.append(i)
    return res


def loss_fn(predict, target):
    p_view = predict.view(-1).data.cpu().numpy()
    t_view = target.view(-1).data.cpu().numpy()
    L = int(len(p_view) / 64)
    p_list = []
    t_list = []
    for step in range(L):
        # t = t_view[step * 64:(step + 1) * 64]

        t = t_view[step * 64:(step + 1) * 64]
        exclude_list = find_val_index(t, 1)
        loss_index = random_index(0, 63, exclude_list, 3)  # 此处先生成2个
        for index in exclude_list:
            loss_index.append(index)
        masked = [0 for i in range(64)]
        for index in loss_index:
            masked[index] = 1

        masked = torch.ByteTensor(masked)
        t_tensor = torch.FloatTensor(t)
        t_tensor = torch.masked_select(t_tensor, masked)

        t = t_tensor.data.cpu().numpy()
        for val in t:
            t_list.append(val)

        p = p_view[step * 64:(step + 1) * 64]
        p_tensor = torch.FloatTensor(p)
        p_tensor = torch.masked_select(p_tensor, masked)
        p = p_tensor.data.cpu().numpy()
        for val in p:
            p_list.append(val)

        # print('t_tensor: ', t_tensor)

    # print('p_view: ', p_view)
    predict = Variable(torch.FloatTensor(p_list), requires_grad=True)
    target = Variable(torch.FloatTensor(t_list), requires_grad=True)
    loss = bce_loss(predict.view(-1), target.view(-1))
    return loss


INPUT_SIZE = 1
LR = 1e-3
# batch_num = 100
batch_size = 7

MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_cnn1_pg_avg_sig\\'


def train_playground():
    model = Net(INPUT_SIZE).cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    data_list = data_extractor.load_data()
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

    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_avg_50_sig.npy', test_data_input)
    np.save('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_avg_50_sig.npy', test_data_label)

    test_batch_num = len(test_data_input)
    test_data_input = np.array(test_data_input).reshape(test_batch_num, 1, 64)
    test_data_label = np.array(test_data_label).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data_input).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label).cuda(0)

    train_data_input = np.array(train_data_input).reshape(batch_num, 1, 64)
    train_data_label = np.array(train_data_label).reshape(batch_num, 1, 64)
    train_data_tensor = torch.FloatTensor(train_data_input).cuda(0)
    train_label_tensor = torch.FloatTensor(train_data_label).cuda(0)

    data_len = len(train_data_input)
    input_data = train_data_input
    label_data = train_data_label
    min_loss = 2
    for i in range(3000):
        loss_sum = 0

        # optimizer.zero_grad()
        # prediction = model(train_data_tensor)
        # loss = loss_fn(prediction, train_label_tensor)
        # loss_val = loss.data.cpu().numpy()
        # loss.backward()
        # optimizer.step()

        for step in range(int(data_len / 10)):
            optimizer.zero_grad()
            x = input_data[step * 10:(step + 1) * 10]
            y = label_data[step * 10:(step + 1) * 10]
            x = np.array(x).reshape(10, 1, 64)
            y = np.array(y).reshape(10, 1, 64)
            x = torch.FloatTensor(x).cuda(0)
            y = torch.FloatTensor(y).cuda(0)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss_sum = loss_sum + loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()
        if i % 50 == 0:
            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < 1.65:
                torch.save(model, MODEL_SAVE_DIR + 'cnn3_pg' + str(i) + '_nn.pkl')
            print(i, ' train_mean_loss: ', loss_sum / (data_len / 10),
                  ' test_loss: ', test_loss, 'min_loss: ', min_loss)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    train_playground()
