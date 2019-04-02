import torch
from torch import nn
import numpy as np
from model.cnn import cnn_classification_model as cnn_model
from data_process import empty_radar_data
from model.cnn import cnn_classification_test as cnn_test

MODEL_SAVE_DIR = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_classification\\'

loss_func = torch.nn.CrossEntropyLoss()
LR = 1e-3


def train():
    model = cnn_model.Net().cuda(0)
    # model = torch.load("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\model_classificationcnn_980_1980.pkl")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_data, train_label, test_data, test_label = empty_radar_data.load_playground_data()

    t_d = test_data
    t_l = test_label

    train_len = len(train_data)
    test_len = len(test_data)
    test_batch_num = test_len
    test_data = np.array(test_data).reshape(test_batch_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data).cuda(0)
    test_label_tensor = torch.LongTensor(test_label).cuda(0)

    min_loss = 2
    ep = []
    tr_loss = []
    te_loss = []
    ac = []  # 准确率
    for epoch in range(3500):
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
        train_loss = loss.data.cpu().numpy()
        if epoch % 10 == 0:
            accuracy = cnn_test.validate(model, t_d, t_l)
            if test_loss < min_loss:
                min_loss = test_loss

            if test_loss < 0.1:
                torch.save(model, MODEL_SAVE_DIR + 'cnn_' + str(epoch) + '.pkl')

            print(epoch, ' train_mean_loss: ', train_loss,
                  ' | test_loss: ', test_loss, ' | min_loss: ', min_loss, ' | accuracy: ', accuracy)

            pr = []
            ep.append(epoch)
            tr_loss.append(train_loss)
            te_loss.append(test_loss)
            pr.append(ep)
            pr.append(tr_loss)
            pr.append(te_loss)

            ac.append(accuracy)
            pr.append(ac)
            np.save("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\\classification_loss_rate.npy", pr)

    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    train()
