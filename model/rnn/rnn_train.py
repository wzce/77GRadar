import sys, os

sys.path.append('../../')  #

import numpy as np
import torch
from torch import nn

from data_process import radar_data
from model.rnn import rnn_model
from model.rnn import rnn_test
from configparser import ConfigParser
import config

LR = 1e-4
pos_weight = torch.FloatTensor([1.6]).cuda(0)
bce_loss = nn.BCEWithLogitsLoss()
# bce_loss = nn.BCEWithLogitsLoss()
SEQ_LEN = 64
BATCH_SIZE = 28000

cp, section = config.load_config(5)
DATA_DIR = cp.get(section, 'processed_data_dir')

train_result_log = os.path.join(DATA_DIR, cp.get(section, 'train_result_log'))
model_save_dir = cp.get(section, 'model_save_dir')


def loss_fn(predict, target, seq_len=SEQ_LEN):
    loss = bce_loss(predict.view(-1), target.view(-1))
    return loss


# def loss_fn(predict, target, seq_len=SEQ_LEN):
#     one_mask = torch.eq(target, 1).cuda(0)
#     # one_mask[0] = 1
#     zero_mask = torch.randint(seq_len, size=target.shape).cuda(0) > 28
#     mask = one_mask | zero_mask
#     # print('     mask: ', mask.view(-1).processed_data.cpu().numpy())
#     # m= mask.processed_data.cpu().numpy().tolist()
#     predict = torch.masked_select(predict, mask)
#     target = torch.masked_select(target, mask)
#
#     loss = bce_loss(predict, target)
#     return loss


def generate_batch(input_data, label_data, batch_size=BATCH_SIZE):
    batch_num = int(len(input_data) / batch_size)
    batch_data_input = []
    batch_data_label = []
    for i in range(batch_num):
        batch_input = input_data[batch_num * batch_size:(batch_num + 1) * batch_size]
        batch_label = label_data[batch_num * batch_size:(batch_num + 1) * batch_size]
        batch_data_input.append(batch_input)
        batch_data_label.append(batch_label)
    batch_data_input.append(input_data[batch_num * batch_size:])
    batch_data_label.append(label_data[batch_num * batch_size:])
    return batch_data_input, batch_data_label


def train_with_pg_data(model, model_save_dir, epochs=3000, save_line=0.7, learn_rate=LR, seq_len=SEQ_LEN):
    # train_data_input, train_data_label, test_data_input, test_data_label = radar_data.load_pg_data_by_range(0, 64)
    train_data_input, train_data_label, test_data_input, test_data_label = radar_data.load_playground_data()
    td = test_data_input
    tl = test_data_label

    train_data_input = radar_data.reduce_data_length(train_data_input, 0, seq_len)
    train_data_label = radar_data.reduce_data_length(train_data_label, 0, seq_len)
    test_data_input = radar_data.reduce_data_length(test_data_input, 0, seq_len)
    test_data_label = radar_data.reduce_data_length(test_data_label, 0, seq_len)

    test_data_num = len(test_data_input)
    test_data_input = np.array(test_data_input).reshape(test_data_num, seq_len, 1)
    test_data_label = np.array(test_data_label).reshape(test_data_num, seq_len, 1)
    test_data_tensor = torch.FloatTensor(test_data_input).cuda(0)
    test_label_tensor = torch.FloatTensor(test_data_label).cuda(0)

    train_data_num = len(train_data_input)
    train_data_input = np.array(train_data_input).reshape(train_data_num, seq_len, 1)
    train_data_label = np.array(train_data_label).reshape(train_data_num, seq_len, 1)
    train_data_tensor = torch.FloatTensor(train_data_input).cuda(0)
    train_label_tensor = torch.FloatTensor(train_data_label).cuda(0)

    min_loss = 22
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    batch_data_input, batch_data_label = generate_batch(train_data_tensor, train_label_tensor)

    h_state = None  # 第一次的时候，暂存为0
    t_state = None
    # f = open("log.txt", "w")

    ep = []
    tr_loss = []
    te_loss = []
    st1_ac = []  # 准确率
    st2_ac = []
    st3_ac = []

    for epoch in range(epochs):
        loss_val = 0
        for i in range(len(batch_data_input)):
            input = batch_data_input[i]
            label = batch_data_label[i]
            optimizer.zero_grad()
            # prediction = model(train_data_tensor)
            prediction, h_state = model(input, h_state)
            h_state = h_state.data

            loss = loss_fn(prediction, label, seq_len)
            loss_val = loss.data.cpu().numpy() + loss_val
            loss.backward(retain_graph=True)
            optimizer.step()

        loss_val = loss_val / (len(batch_data_label))
        test_prediction, _ = model(test_data_tensor, None)
        # t_state = t_state.processed_data
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if epoch % 25 == 0:

            st1, st2, st3 = rnn_test.model_test(model, td, tl, line=0.1)

            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < save_line:
                torch.save(model, model_save_dir + 'rnn_' + str(epoch) + '.pkl')
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

            # np.save('D:\home\zeewei\projects\\77GRadar\model\\rnn\\train_process_0407_1.npy', pr)
            np.save(train_result_log, pr)
            # print(log)
            # print(log, file=f)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    # model = rnn_model.RadarRnn4(INPUT_SIZE=1).cuda(0)
    # cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn4-32-8-0\\'
    # epochs = 10000
    # train_with_pg_data(model, cnn_model_dir, epochs=epochs, save_line=0.12, learn_rate=5e-4)

    model = rnn_model.RadarRnn2(INPUT_SIZE=1).cuda(0)
    # model = torch.load(
    # "D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn2-32-4-0\\rnn_loss2_9990_270_0.pkl")
    # cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn2_0407_1\\'
    epochs = 10000
    train_with_pg_data(model, model_save_dir, epochs=epochs, save_line=0.8, learn_rate=5e-4)
