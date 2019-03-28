from model.cnn import cnn_model
from data_process import radar_data
import torch
import numpy as np
from torch import nn

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
#     # print('     mask: ', mask.view(-1).data.cpu().numpy())
#     # m= mask.data.cpu().numpy().tolist()
#     predict = torch.masked_select(predict, mask)
#     target = torch.masked_select(target, mask)
#
#     loss = bce_loss(predict, target)
#     return loss


def train(model, model_save_dir, epochs=1000, save_line=0.7, learn_rate=LR):
    train_data_input, train_data_label, test_data_input, test_data_label = radar_data.load_pg_data_by_range(0, 64)

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

    for i in range(epochs):
        optimizer.zero_grad()
        prediction = model(train_data_tensor)
        loss = loss_fn(prediction, train_label_tensor)
        loss_val = loss.data.cpu().numpy()
        loss.backward(retain_graph=True)
        optimizer.step()

        test_prediction = model(test_data_tensor)
        test_loss = loss_fn(test_prediction, test_label_tensor)
        test_loss = test_loss.data.cpu().numpy()

        if i % 10 == 0:
            if test_loss < min_loss:
                min_loss = test_loss
            if test_loss < save_line:
                torch.save(model, model_save_dir + 'cnn_2990_' + str(i) + '.pkl')
            print('{:0=4} \t train_loss:{} \t test_loss: {} \t test_min_loss: {} \t difference: {}'
                  .format(i, loss_val, test_loss, min_loss, (test_loss - loss_val)))
    print('test_min_loss: ', min_loss)


if __name__ == '__main__':
    cnn_model_dir = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn1\\'
    model = cnn_model.RadarCnn1_2().cuda(0)
    model = torch.load("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn1\cnn_2990.pkl")
    epochs = 3000
    train(model, cnn_model_dir, epochs, 0.5, learn_rate=1e-3)
