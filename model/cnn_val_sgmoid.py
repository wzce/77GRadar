from torch import nn
import torch
import numpy as np
from util.cv import right_distribute
from data_process import radar_data
import os


def model_test(model_path):
    model = torch.load(model_path)
    # input_data, label_data = radar_data.load_pg_test_data()
    _, _1, input_data, label_data = radar_data.load_pg_data_by_range(0, 64)

    # input_data, label_data = radar_data.load_val_data(
    #     'D:\home\zeewei\projects\\77GRadar\data\\all\\all_val_data.npy')
    # L = len(input_data)

    correct_num = 0
    relative_correct_num = 0
    total_num = len(input_data)
    sca = np.zeros(64)
    sca_r = np.zeros(64)
    data_sca = np.zeros(64)
    right_location_num = 0
    zero_num = 0
    for step in range(len(input_data)):
        print('\n<----------------------------------------------------------', step)
        x = input_data[step:(step + 1)]
        y = label_data[step:(step + 1)]
        max_y_index = 0
        # 求y的最大值下标
        for i in range(len(y[0])):
            if y[0][i] >= y[0][max_y_index]:
                max_y_index = i

        data_sca[max_y_index] = data_sca[max_y_index] + 1

        x = np.array(x).reshape(1, 1, 64)
        y = np.array(y).reshape(1, 1, 64)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.ByteTensor(y).cuda(0)
        prediction = model(x)
        _, max = torch.max(prediction.data, 1)

        predict = torch.sigmoid(prediction) > 0.015625

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
        tt = []
        for item in t:
            if item == 0:
                tt.append(0)
            else:
                tt.append(1)
        t = tt
        pd = predict.view(-1).data.cpu().numpy()
        pd = np.array(pd).tolist()
        result = torch.eq(y, predict)
        res = result.view(-1).data.cpu().numpy()
        res = np.array(res).tolist()
        print('target:    ', t)
        print('predict:   ', pd)
        print('difference:', res)

        max_predict_index = torch.max(predict, 1)[1].data.cpu().numpy()[0]

        # 在某个位置前后偏离两个位置
        if max_y_index >= 2 and max_y_index <= 62:
            if t[max_y_index] == pd[max_y_index] or t[max_y_index] == pd[max_y_index - 1] \
                    or t[max_y_index] == pd[max_y_index + 1] \
                    or t[max_y_index] == pd[max_y_index - 1]:
                # or t[max_y_index] == pd[max_y_index + 2] \
                # or t[max_y_index] == pd[max_y_index - 2] \

                print('relative right')
                right_location_num = right_location_num + 1
                sca_r[max_y_index] = sca_r[max_y_index] + 1
        else:
            if t[max_y_index] == pd[max_y_index]:
                right_location_num = right_location_num + 1
                sca_r[max_y_index] = sca_r[max_y_index] + 1

        result = torch.eq(y, predict)
        accuracy = torch.sum(result) / torch.sum(torch.ones(y.shape))
        accuracy = accuracy.data.cpu().numpy()
        correct = torch.eq(torch.sum(~result), 0)
        print('accuracy: ', accuracy)
        print('correct: {}'.format(correct))
        if correct == 1:
            # right_index = int(max_y_index/10)
            sca[max_y_index] = sca[max_y_index] + 1
            correct_num = correct_num + 1
        print('-------------------------------------------------------------->\n')

    print('total:', (step + 1),
          ' \n| correct_num:', correct_num,
          ' \n| complete_correct_rate:', correct_num / total_num,
          ' \n|right_location_rate: ', right_location_num / total_num,
          ' \n|zero_num:', zero_num,
          ' \n|zero_num_rate: ', zero_num / total_num)
    print('sca : ', sca)
    print('sca_r: ', sca_r)
    print('data_sca:', data_sca)
    right_distribute.distribute_cv(sca, data_sca, 64)
    right_distribute.distribute_cv(sca_r, data_sca, 64)


if __name__ == '__main__':
    model_location = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_data_all\cnn2_1_val'
    model_path = os.path.join(model_location, 'cnn_0_2280.pkl')
    model_test(model_path)
