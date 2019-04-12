from torch import nn
import torch
import numpy as np
from data_process import radar_data
from model.bp import bp_test
from model.bp import bp_model

LR = 1e-4
bce_with_logits_loss = nn.BCEWithLogitsLoss()
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()


def loss_fn(predict, target):
    loss = bce_with_logits_loss(predict.view(-1), target.view(-1))
    return loss


def train(model, model_save_dir, epochs=1000, save_line=0.7, learn_rate=LR):
    train_data_input, train_data_label, test_data_input_, test_data_label_ = radar_data.load_playground_data()
    test_data_input, test_data_label = radar_data.load_val_data()
    td = test_data_input
    tl = test_data_label
    test_data_num = len(test_data_input)
    # test_data_input = np.array(test_data_input).reshape(test_data_num, 1, 64)
    # test_data_label = np.array(test_data_label).reshape(test_data_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data_input).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label).cuda(0)

    train_data_num = len(train_data_input)
    # train_data_input = np.array(train_data_input).reshape(train_data_num, 1, 64)
    # train_data_label = np.array(train_data_label).reshape(train_data_num, 1, 64)
    train_data_tensor = torch.FloatTensor(train_data_input).cuda(0)
    train_label_tensor = torch.FloatTensor(train_data_label).cuda(0)

    min_loss = 200
    # L = len(input_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    ep = []
    tr_loss = []
    te_loss = []
    st1_ac = []  # 准确率
    st2_ac = []
    st3_ac = []

    for epoch in range(epochs):
        # loss_sum = 0

        prediction = model(train_data_tensor)
        loss = loss_fn(prediction, train_label_tensor)
        loss_val = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if epoch % 5 == 0:
            st1, st2, st3 = bp_test.model_test(model, td, tl, line=0.1)
            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < save_line:
                torch.save(model, model_save_dir + 'bp1_' + str(
                    epoch) + '.pkl')
            pr = []
            ep.append(epoch)
            tr_loss.append(loss_val)
            te_loss.append(test_loss)
            pr.append(ep)
            pr.append(tr_loss)
            pr.append(te_loss)

            st1_ac.append(st1)
            st2_ac.append(st2)
            st3_ac.append(st3)

            pr.append(st1_ac)
            pr.append(st2_ac)
            pr.append(st3_ac)
            np.save("D:\home\zeewei\projects\\77GRadar\model\\bp\model_dir\\bp_train_log_0411.npy", pr)

    print('test_min_loss: ', min_loss)

    torch.save(model, 'BP1.pkl')

if __name__ == '__main__':
    cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\bp\model_dir\\bp1\\'
    model = bp_model.BP_Net1().cuda(0)
    # model = torch.load("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn2_1\cnn9990.pkl")
    epochs = 5000
    train(model, cnn_model_dir, epochs, 2, learn_rate=1e-3)
