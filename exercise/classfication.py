import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from sklearn import datasets

#
iris = datasets.load_iris()

x = iris['processed_data']
x_train = x[0:130]
x_test = x[130:150]
y = iris['target']
y_train = y[0:130]
y_test = y[130:150]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)


# x = Variable(x)
# y = Variable(y)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_out))
        self.out = nn.Softmax()

    def forward(self, x):
        x = self.hidden(x)
        out = self.out(x)
        return out


bce_loss = nn.BCEWithLogitsLoss()


def loss_fn(predict, target):
    loss = bce_loss(predict.view(-1), target.view(-1))
    return loss


def train():
    net = Net(n_feature=4, n_hidden=5, n_out=3)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
    epochs = 100000
    px = [];
    py = []
    for i in range(epochs):
        predict = net(x_train)
        loss = loss_fn(predict, y_train)  # 输出层 用了log_softmax 则需要用这个误差函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(i, "loss:", loss.data.numpy())
        px.append(i)
        py.append(loss.data.numpy())
        if (i == epochs):
            plt.cla
            plt.title(u"loss curve")
            plt.xlabel(u"iterate times")
            plt.ylabel(u"loss")
            plt.plot(px, py, "r-", lw=1)
            plt.text(0, 0, "Loss = %.4f" % loss.data.numpy(), fontdict={"size": 20, 'color': 'red'})
            # if i < (epochs - 2000):
            #     plt.pause(0.1)
            # else:
            #     pass
    torch.save(net, "my_model.pkl")


def test():
    iris_model = torch.load("my_model.pkl")
    print(iris_model)
    net = torch.load("D:\home\zeewei\projects\\77GRadar\exercise\my_model.pkl")

    all_predict = net(x_test).data.numpy()

    '''
        argmax(processed_data,axis = 1)  axis = 1表示 按照行求最大值的索引
    '''
    print((np.argmax(all_predict, axis=1) == iris['target']).sum() / len(y))


train()
