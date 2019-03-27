from torch import nn
import torch
import numpy as np
from util.cv import right_distribute
from data_process import radar_data
# from data_process import feature_extractor as road_data
import os

from data_process import feature_extractor
import math

SEQ_LEN = 64


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


input_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_avg_50_sig.npy')
label_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_avg_50_sig.npy')


def model_test(model_path):
    model = torch.load(model_path)
    # # input_data, label_data = radar_data.load_pg_test_data()
    # _, _1, input_data, label_data = radar_data.load_pg_data_by_range(0, SEQ_LEN)
    # # _, _1, input_data, label_data = radar_data.load_road_data()
    # input_data = radar_data.reduce_data_length(input_data, 0, SEQ_LEN)
    # label_data = radar_data.reduce_data_length(label_data, 0, SEQ_LEN)
    # L = len(input_data)

    correct_num = 0
    # relative_correct_num = 0
    total_num = len(input_data)
    sca = np.zeros(SEQ_LEN)
    sca_r = np.zeros(SEQ_LEN)
    data_sca = np.zeros(SEQ_LEN)
    right_location_num = 0
    zero_num = 0
    for step in range(len(input_data)):
        print(step, '------------------------------------------------------------>')
        x = input_data[step:(step + 1)]
        y = label_data[step:(step + 1)]
        max_y_index = 0
        # 求y的最大值下标
        for i in range(len(y[0])):
            if y[0][i] >= y[0][max_y_index]:
                max_y_index = i

        # if not (max_y_index >= 18 and max_y_index <= 22):  # 去掉低的情况
        #     total_num = total_num - 1
        #     continue

        data_sca[max_y_index] = data_sca[max_y_index] + 1
        x = np.array(x).reshape(1, 1, 64)
        y = np.array(y).reshape(1, 1, 64)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.ByteTensor(y).cuda(0)
        prediction = model(x)

        predict = torch.sigmoid(prediction) > 0.1
        max_predict_index = 0
        mm = 0
        for i in range(1, len(predict.view(-1))):
            if predict.view(-1)[i] > mm:
                max_predict_index = i
                mm = predict.view(-1)[i]

        if max_predict_index == 0:
            zero_num = zero_num + 1

        t = label_data[step:(step + 1)]
        # t=
        t = np.array(t[0]).tolist()
        pd = predict.view(-1).data.cpu().numpy()
        pd = np.array(pd).tolist()
        result = torch.eq(y, predict)
        res = result.view(-1).data.cpu().numpy()
        res = np.array(res).tolist()
        print('target:    ', t)
        print('predict:   ', pd)
        print('difference:', res)

        #
        # if (abs(max_y_index - max_predict_index) < 3):
        #     print('step: ', step)
        #     # sca_r[max_y_index] = sca_r[max_y_index] + 1
        #     relative_correct_num = relative_correct_num + 1

        # 在某个点上有物体完全预测正确
        # if t[0][max_y_index] == pd[max_y_index]:
        #     right_location_num = right_location_num+1

        # 在某个位置前后偏离两个位置
        if max_y_index >= 2 and max_y_index <= 62:
            if t[max_y_index] == pd[max_y_index] or t[max_y_index] == pd[max_y_index - 1] \
                    or t[max_y_index] == pd[max_y_index + 1] \
                    or t[max_y_index] == pd[max_y_index + 2] \
                    or t[max_y_index] == pd[max_y_index - 2] \
                    or t[max_y_index] == pd[max_y_index - 1]:
                print('relative right')
                right_location_num = right_location_num + 1
                sca_r[max_y_index] = sca_r[max_y_index] + 1
        else:
            if t[max_y_index] == pd[max_y_index]:
                right_location_num = right_location_num + 1
                sca_r[max_y_index] = sca_r[max_y_index] + 1

        accuracy = torch.sum(result) / torch.sum(torch.ones(y.shape))
        accuracy = accuracy.data.cpu().numpy()
        correct = torch.eq(torch.sum(~result), 0)
        print('accuracy: ', accuracy)
        print('correct: {}'.format(correct))
        if correct == 1:
            # right_index = int(max_y_index/10)
            sca[max_y_index] = sca[max_y_index] + 1
            correct_num = correct_num + 1
        print('<------------------------------------------------------------------\n')

    print('total:', (step + 1), ' | correct_num:', correct_num, '| complete_correct_rate:', correct_num / total_num,
          # '| relative_correct_num : ', relative_correct_num, '| relative_correct_rate: ',
          # relative_correct_num / total_num,
          ' |right_location_rate: ', right_location_num / total_num,
          ' |zero_num:', zero_num,
          ' |zero_num_rate: ', zero_num / total_num)
    print('sca : ', sca)
    print('sca_r: ', sca_r)
    print('data_sca:', data_sca)

    right_distribute.distribute_cv(sca, data_sca, SEQ_LEN)
    right_distribute.distribute_cv(sca_r, data_sca, SEQ_LEN)


if __name__ == '__main__':
    model_location = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn_sig\\'
    # model_location = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn4-32-8-28'
    model_path = os.path.join(model_location, 'cnn3_pg2250_nn.pkl')
    model_test(model_path)
