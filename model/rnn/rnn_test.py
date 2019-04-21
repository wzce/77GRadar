from torch import nn
import torch
import numpy as np
from util.cv import right_distribute
from data_process import radar_data
# from data_process import feature_extractor as road_data
import os
from config import data_config
from util import standard_define

SEQ_LEN = 64


def model_test(model, input_data, label_data, is_debug=False, line=0.1):
    total_num = len(input_data)

    correct_num = 0
    st2_num = 0
    st3_num = 0
    total_num = len(input_data)
    st1 = np.zeros(64)
    st2 = np.zeros(64)
    st3 = np.zeros(64)
    data_sca = np.zeros(64)

    zero_num = 0
    for step in range(len(input_data)):
        if is_debug:
            print('\n<----------------------------------------------------------', step)
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
        x = np.array(x).reshape(1, SEQ_LEN, 1)
        y = np.array(y).reshape(1, SEQ_LEN, 1)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.ByteTensor(y).cuda(0)
        prediction, _ = model(x, None)

        predict = torch.sigmoid(prediction) > line
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
        if is_debug:
            print('target:    ', t)
            print('predict:   ', pd)
            print('difference:', res)

        # 在某个位置前后偏离两个位置
        if standard_define.is_satisfied_standard2(pd, max_y_index):
            st2_num = st2_num + 1
            st2[max_y_index] = st2[max_y_index] + 1

        if standard_define.is_satisfied_standard3(pd, max_y_index):
            st3_num = st3_num + 1
            st3[max_y_index] = st3[max_y_index] + 1

        accuracy = torch.sum(result) / torch.sum(torch.ones(y.shape))
        accuracy = accuracy.data.cpu().numpy()
        correct = torch.eq(torch.sum(~result), 0)
        if is_debug:
            print('accuracy: ', accuracy)
            print('correct: {}'.format(correct))
        if correct == 1:
            # right_index = int(max_y_index/10)
            st1[max_y_index] = st1[max_y_index] + 1
            correct_num = correct_num + 1  # 标准1，完全匹配
        if is_debug:
            print('-------------------------------------------------------------->\n')

    if is_debug:
        print('total:', (step + 1), ' | correct_num:', correct_num, '| complete_correct_rate:', correct_num / total_num,
              '| st2_num: ', st2_num, ' |st2_rate: ', st2_num / total_num,
              '| st3_num: ', st3_num, ' |st3_rate: ', st3_num / total_num)
        print('st1 : ', st1)
        print('data_sca : ', data_sca)

        right_distribute.distribute_cv(st1, data_sca, 36, 'st1完全正确预测分布')
        right_distribute.distribute_cv(st2, data_sca, 36, 'st2相对预测正确率')
        right_distribute.distribute_cv(st3, data_sca, 36, 'st3相对预测正确率')

    return correct_num / total_num, st2_num / total_num, st3_num / total_num


if __name__ == '__main__':
    st1 = []
    st2 = []
    st3 = []

    config = data_config.DataConfig()
    model_location = config.rnn_model_save_dir

    model_location = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn2_1_0409'
    model_path = os.path.join(model_location, 'rnn_1525.pkl')
    model = torch.load(model_path)
    # _, _1, input_data, label_data = radar_data.load_pg_data_by_range(0, SEQ_LEN)
    input_data, label_data = radar_data.load_val_data()
    # train_data_input, train_data_label, input_data, label_data = radar_data.load_playground_data()
    line = 0.32
    st1_val, st2_val, st3_val = model_test(model, input_data, label_data, is_debug=True, line=line)
    # for i in range(1, 100, 1):
    #     line = i * 0.01
    #     print('---> ', line)
    #     st1_val, st2_val, st3_val = model_test(model, input_data, label_data, is_debug=False, line=line)
    #     st1.append(st1_val)
    #     st2.append(st2_val)
    #     st3.append(st3_val)
    #     print(st1_val, st2_val, st3_val)
    # correct = []
    # correct.append(st1)
    # correct.append(st2)
    # correct.append(st3)
    # np.save("D:\home\zeewei\projects\\77GRadar\exercise\\rnn_0407_correct.npy", correct)
