from torch import nn
import torch
import numpy as np
from data_process import feature_extractor
import math
from util.cv import right_distribute

class Net1(nn.Module):
    def __init__(self, INPUT_SIZEINPUT_SIZE):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            # nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool1d(kernel_size=2)
        )

        self.out = nn.Sequential(nn.Linear(16, 32),
                                 nn.ReLU(True),
                                 nn.Linear(32, 64),
                                 nn.Softmax())  # 分类器，预测位置最大的一个
        # self.fc = nn.Linear(960, 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)



model = torch.load('D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_cnn1_pg_avg\cnn1_pg870_new_70_all2.pkl')

# data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
input_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\input_data_avg_70_all2.npy')
label_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data_pg\\label_data_avg_70_all2.npy')
# data_extractor = feature_extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
# input_data, label_data = data_extractor.load_train_data()

L = len(input_data)

correct_num = 0
relative_correct_num = 0

total_num = len(input_data)

sca = [0 for i in range(64)]
sca_r = [0 for i in range(64)]
data_sca = [0 for i in range(64)]
for step in range(len(input_data)):
    print('\n<----------------------------------------------------------', step)
    x = input_data[step:(step + 1)]
    y = label_data[step:(step + 1)]

    max_y_index = 0
    # 求y的最大值下标
    for i in range(len(y[0])):
        if y[0][i] >= y[0][max_y_index]:
            max_y_index = i

    # if max_y_index > 34:
    #     print('error: ', y)
    #     total_num = total_num - 1
    #     continue
    # if max_y_index <13:
    #     total_num = total_num - 1
    #     continue

    # data_index = int(max_y_index/10)
    data_sca[max_y_index] = data_sca[max_y_index] + 1

    x = np.array(x).reshape(1, 1, 64)
    y = np.array(y).reshape(1, 1, 64)
    x = torch.FloatTensor(x).cuda(0)
    y = torch.ByteTensor(y).cuda(0)
    prediction = model(x)
    # print('prediction: ', prediction[0][0:6])
    _, max = torch.max(prediction.data, 1)
    index = 0
    mm = prediction[0][0]
    for i in range(1, len(prediction[0])):
        if prediction[0][i] > mm:
            index = i
            mm = prediction[0][i]
        # print('val: ', prediction[0][i])
    predict = [j - j for j in range(0, 64)]
    predict[index] = 1
    predict = [predict]
    predict = torch.ByteTensor(predict).cuda(0)
    # predict = torch.sigmoid(prediction) > 0.5

    # predict = prediction
    # print('y:     ', y)
    # print('predict:', predict)
    # print(torch.eq(y, predict))

    # max_y_index = torch.max(y, 1)[1]  # 行的最大值的下标
    max_predict_index = torch.max(predict, 1)[1].data.cpu().numpy()[0]

    if (abs(max_y_index - max_predict_index) < 4):
        relative_correct_num = relative_correct_num + 1
        sca_r[max_y_index] = sca_r[max_y_index] + 1
    # print('max_y_index : ', max_y_index, '  max_predict_index: ', max_predict_index)

    result = torch.eq(y, predict)
    accuracy = torch.sum(result) / torch.sum(torch.ones(y.shape))
    accuracy = accuracy.data.cpu().numpy()
    correct = torch.eq(torch.sum(~result), 0)
    # if math.fabs(0.98437 - accuracy) > 0.0001:
    # print('accuracy: ', accuracy)
    # print('correct: {}'.format(correct))
    if correct == 1:
        # right_index = int(max_y_index/10)
        sca[max_y_index] = sca[max_y_index] + 1
        # sca_r[max_y_index] = sca_r[max_y_index] + 1
        correct_num = correct_num + 1

    # print('-------------------------------------------------------------->\n')

print('total:', (step + 1), ' | correct_num:', correct_num, '| complete_correct_rate:', correct_num / total_num,
      '| relative_correct_num : ', relative_correct_num, '| relative_correct_rate: ', relative_correct_num / total_num)

print('sca : ', sca)
print('sca_r: ', sca_r)
print('data_sca:', data_sca)

# right_distribute.distribute_cv(sca, data_sca)
right_distribute.distribute_cv(sca_r, data_sca)
# print('prediction: ', prediction.data.cpu().numpy().flatten())
