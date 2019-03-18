import torch
from torch import nn
import numpy as np
import os
from data_process import classification_data_extractor
from model.cnn import cnn_classfication

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
        self.fc = nn.Linear(64, 2)

    def forward(self, input_data):
        x = self.conv1d(input_data)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def validate():
    model = torch.load(os.path.join(MODEL_SAVE_DIR, 'cnn_classification_3_12400_new.pkl'))
    extractor = classification_data_extractor.ClassificationExtractor()  # 此处全使用默认的文件路径配置,获取有目标数据和无目标数据

    car_data = extractor.load_car_data()
    empty_data = extractor.load_empty_data()

    car_train_data_len = int(2 * len(car_data) / 3)
    empty_train_data_len = int(2 * len(empty_data) / 3)

    # test_data = car_data #[car_train_data_len:len(car_data)]
    # test_label = [1 for i in range(len(test_data))]
    # empty_test_data = empty_data[empty_train_data_len:len(empty_data)]
    # for item in empty_test_data:
    #     test_data.append(item)
    #     test_label.append(0)

    test_data = np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data\\input_data.npy')
    test_label= np.load('D:\home\zeewei\projects\\77GRadar\model\cnn\\test_data\\label_data.npy')

    test_len = len(test_data)
    test_batch_num = test_len
    test_data = np.array(test_data).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.LongTensor(test_label).cuda(0)

    test_prediction = model(test_data_tensor)
    prediction = torch.max(test_prediction, 1)[1]  # 行的最大值的下标
    print(torch.eq(prediction, test_label_tensor))
    predict_y = prediction.data.cpu().numpy().squeeze()
    target_y = test_label_tensor.data.cpu().numpy()

    accuracy = sum(predict_y == target_y) / len(target_y)  # 预测中有多少和真实值一样
    print('accuracy: ', accuracy)


validate()
