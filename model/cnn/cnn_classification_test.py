import torch
from model.cnn import cnn_classification_model as cnn_model
import numpy as np
import os
from data_process import empty_radar_data


def validate(model, test_data, test_label):
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
    return accuracy


if __name__ == '__main__':
    MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_classification'
    model = torch.load(os.path.join(MODEL_SAVE_DIR, 'cnn_2930.pkl'))
    train_data, train_label, test_data, test_label = empty_radar_data.load_playground_data()
    # test_data = np.load("D:\home\zeewei\projects\\77GRadar\classification_train_data\pg_empty_val_data.numpy.npy")
    # test_label = np.load("D:\home\zeewei\projects\\77GRadar\classification_train_data\pg_empty_val_label.numpy.npy")
    print("数据量： ", len(test_data))
    validate(model, test_data, test_label)
