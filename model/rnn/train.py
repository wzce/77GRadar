import numpy as np
import torch
from torch import nn

from data_process import radar_data
from model.rnn import rnn_model

LR = 1e-4
pos_weight = torch.FloatTensor([1.6]).cuda(0)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# bce_loss = nn.BCEWithLogitsLoss()
SEQ_LEN = 64
BATCH_SIZE = 2000


def loss_fn(predict, target, seq_len=SEQ_LEN):
    loss = bce_loss(predict.view(-1), target.view(-1))
    return loss


# def loss_fn(predict, target, seq_len=SEQ_LEN):
#     one_mask = torch.eq(target, 1).cuda(0)
#     # one_mask[0] = 1
#     zero_mask = torch.randint(seq_len, size=target.shape).cuda(0) > 56
#     mask = one_mask | zero_mask
#     # print('     mask: ', mask.view(-1).data.cpu().numpy())
#     # m= mask.data.cpu().numpy().tolist()
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
    train_data_input, train_data_label, test_data_input, test_data_label = radar_data.load_pg_data_by_range(0, 32)
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
        # t_state = t_state.data
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if epoch % 20 == 0:
            if test_loss < save_line:
                torch.save(model, model_save_dir + 'rnn_loss2_' + str(epoch) + '.pkl')
            if test_loss < min_loss:
                min_loss = test_loss
            log = '{:0=4} \t train_loss:{} \t test_loss: {} \t test_min_loss: {} \t difference: {}'.format(
                epoch, loss_val, test_loss, min_loss, (test_loss - loss_val))
            print(log)
            # print(log, file=f)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    # model = rnn_model.RadarRnn4(INPUT_SIZE=1).cuda(0)
    # cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn4-32-8-0\\'
    # epochs = 10000
    # train_with_pg_data(model, cnn_model_dir, epochs=epochs, save_line=0.12, learn_rate=5e-4)

    model = rnn_model.RadarRnn5(INPUT_SIZE=1).cuda(0)
    cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn5-64-8-0\\'
    epochs = 10000
    train_with_pg_data(model, cnn_model_dir, epochs=epochs, save_line=0.2, learn_rate=5e-5)
