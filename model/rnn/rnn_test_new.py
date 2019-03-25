from torch import nn
import torch
import numpy as np
from util.cv import right_distribute
from data_process import radar_data
import os


def model_test(model_path):
    model = torch.load(model_path)
    # input_data, label_data = radar_data.load_pg_test_data()
    _, _1, input_data, label_data = radar_data.load_pg_data_by_range(15, 30)

    # L = len(input_data)

    correct_num = 0
    relative_correct_num = 0
    total_num = len(input_data)
    sca = np.zeros(64)
    sca_r = np.zeros(64)
    data_sca = np.zeros(64)
    for step in range(len(input_data)):
        print('\n<----------------------------------------------------------', step)
        x = input_data[step:(step + 1)]
        y = label_data[step:(step + 1)]
        max_y_index = 0
        # 求y的最大值下标
        for i in range(len(y[0])):
            if y[0][i] >= y[0][max_y_index]:
                max_y_index = i

        # if max_y_index >35:
        #     print('error: ', y)
        #     break
        # if max_y_index > 30:
        #     total_num = total_num - 1
        #     continue

        # data_index = int(max_y_index/10)
        data_sca[max_y_index] = data_sca[max_y_index] + 1

        data_sca[max_y_index] = data_sca[max_y_index] + 1
        x = np.array(x).reshape(1, 64, 1)
        # y = np.array(y).reshape(1, 64, 1)
        x = torch.FloatTensor(x).cuda(0)
        y = torch.ByteTensor(y).cuda(0)

        prediction,_ = model(x,None)
        _, max = torch.max(prediction.data, 1)
        index = 0
        mm = prediction[0][0]
        for i in range(1, len(prediction[0])):
            if prediction[0][i] > mm:
                index = i
                mm = prediction[0][i]
        predict = [j - j for j in range(0, 64)]
        predict[index] = 1
        predict = [predict]
        predict = torch.ByteTensor(predict).cuda(0)
        # predict = torch.sigmoid(prediction) > 0.5

        # predict = prediction
        print('y:      ', y.data.cpu().numpy().tolist())
        print('predict:', predict.data.cpu().numpy().tolist())
        print(torch.eq(y, predict))

        max_predict_index = torch.max(predict, 1)[1].data.cpu().numpy()[0]

        if (abs(max_y_index - max_predict_index) < 3):
            sca_r[max_y_index] = sca_r[max_y_index] + 1
            relative_correct_num = relative_correct_num + 1

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

    print('total:', (step + 1), ' | correct_num:', correct_num, '| complete_correct_rate:', correct_num / total_num,
          '| relative_correct_num : ', relative_correct_num, '| relative_correct_rate: ',
          relative_correct_num / total_num)
    print('sca : ', sca)
    print('data_sca : ', data_sca)

    right_distribute.distribute_cv(sca, data_sca, 64)
    right_distribute.distribute_cv(sca_r, data_sca, 64)


if __name__ == '__main__':
    model_location = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn2'
    model_path = os.path.join(model_location, 'rnn_loss2_8900.pkl')
    model_test(model_path)
