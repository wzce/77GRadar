from model.cnn import cnn_model
from data_process import radar_data
import torch
import numpy as np
from torch import nn
from model.cnn import cnn_test
from config import data_config

LR = 1e-4
bce_with_logits_loss = nn.BCEWithLogitsLoss()
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()


def loss_fn(predict, target):
    loss = bce_with_logits_loss(predict.view(-1), target.view(-1))
    return loss


# SEQ_LEN = 64
#
#
# def loss_fn(predict, target):
#     one_mask = torch.eq(target, 1).cuda(0)
#     # one_mask[0] = 1
#     zero_mask = torch.randint(SEQ_LEN, size=target.shape).cuda(0) > 58
#     mask = one_mask | zero_mask
#     # print('     mask: ', mask.view(-1).processed_data.cpu().numpy())
#     # m= mask.processed_data.cpu().numpy().tolist()
#     predict = torch.masked_select(predict, mask)
#     target = torch.masked_select(target, mask)
#
#     loss = bce_loss(predict, target)
#     return loss


def train(model, model_save_dir, train_parameter_file, epochs=1000, save_line=0.7, learn_rate=LR):
    train_data_input, train_data_label, test_data_input_, test_data_label_ = radar_data.load_playground_data()
    test_data_input, test_data_label = radar_data.load_val_data()
    td = test_data_input
    tl = test_data_label
    test_data_num = len(test_data_input)
    test_data_input = np.array(test_data_input).reshape(test_data_num, 1, 64)
    test_data_label = np.array(test_data_label).reshape(test_data_num, 1, 64)
    test_data_tensor = torch.FloatTensor(test_data_input).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label).cuda(0)

    train_data_num = len(train_data_input)
    train_data_input = np.array(train_data_input).reshape(train_data_num, 1, 64)
    train_data_label = np.array(train_data_label).reshape(train_data_num, 1, 64)
    train_data_tensor = torch.FloatTensor(train_data_input).cuda(0)
    train_label_tensor = torch.FloatTensor(train_data_label).cuda(0)

    min_loss = 20

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    ep = []
    tr_loss = []
    te_loss = []
    st1_ac = []  # 准确率
    st2_ac = []
    st3_ac = []
    # pr = np.load("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\\train_cnn2_1.npy")

    # ep = pr[0].tolist()
    # tr_loss = pr[1].tolist()
    # te_loss = pr[2].tolist()
    # c_ac = pr[3].tolist()  # 准确率
    # r_ac = pr[4].tolist()
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(train_data_tensor)
        loss = loss_fn(prediction, train_label_tensor)
        loss_val = loss.data.cpu().numpy()
        loss.backward(retain_graph=True)
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if epoch % 15 == 0:

            st1, st2, st3 = cnn_test.model_test(model, td, tl, line=0.15)

            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < save_line:
                torch.save(model, model_save_dir + 'cnn_' + str(epoch) + '.pkl')
            print(
                '{:0=4} \t train_loss:{} \t test_loss: {} \t test_min_loss: {} \t difference: {} \t st1_acc:{} \t st2_acc:{} \t st3_acc:{}'
                    .format(epoch, loss_val, test_loss, min_loss, (test_loss - loss_val), st1, st2, st3))

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
            # np.save(train_parameter_file, pr)

    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    config = data_config.DataConfig()
    cnn_model_dir = config.cnn_model_save_dir
    model = cnn_model.Radar_Cnn_2_1().cuda(0)
    # model = torch.load("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn2_1\cnn9990.pkl")
    epochs = 5000
    train(model, cnn_model_dir, config.train_parameter_file, epochs, 2, learn_rate=1e-3)
