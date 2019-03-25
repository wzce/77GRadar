import numpy as np
import torch
from torch import nn

from data_process import radar_data
from model.rnn import rnn_model

LR = 1e-4
pos_weight = torch.FloatTensor([1.6]).cuda(0)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# def loss_fn(predict, target):
#     loss = bce_loss(predict.view(-1), target.view(-1))
#     return loss


def loss_fn(predict, target):
    one_mask = torch.eq(target, 1).cuda(0)
    # one_mask[0] = 1
    zero_mask = torch.randint(seq_len, size=target.shape).cuda(0) > 28
    mask = one_mask | zero_mask
    # print('     mask: ', mask.view(-1).data.cpu().numpy())
    # m= mask.data.cpu().numpy().tolist()
    predict = torch.masked_select(predict, mask)
    target = torch.masked_select(target, mask)

    loss = bce_loss(predict, target)
    return loss


SEQ_LEN = 64


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

    min_loss = 2

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    h_state = None  # 第一次的时候，暂存为0
    t_state = None
    # f = open("log.txt", "w")
    for i in range(epochs):
        optimizer.zero_grad()
        # prediction = model(train_data_tensor)

        prediction, h_state = model(train_data_tensor, h_state)
        h_state = h_state.data

        loss = loss_fn(prediction, train_label_tensor)
        loss_val = loss.data.cpu().numpy()
        loss.backward(retain_graph=True)
        optimizer.step()

        test_prediction, _ = model(test_data_tensor, None)
        # t_state = t_state.data
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if i % 20 == 0:
            if test_loss < save_line and test_loss < min_loss:
                torch.save(model, model_save_dir + 'rnn_loss2_' + str(i) + '.pkl')
            if test_loss < min_loss:
                min_loss = test_loss
            log = '{:0=4} \t\n train_loss:{} \t\n test_loss: {} \t\n test_min_loss: {} \t\n difference: {}'.format(
                i, loss_val, test_loss, min_loss, (test_loss - loss_val))
            print(log)
            # print(log, file=f)
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    model = rnn_model.RadarRnn2(INPUT_SIZE=1).cuda(0)
    print(model)
    cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn4-32-8-28\\'
    # model = rnn_model.RadarCnn3().cuda(0)
    epochs = 300000
    train_with_pg_data(model, cnn_model_dir, epochs, 0.0005, learn_rate=1e-3)
