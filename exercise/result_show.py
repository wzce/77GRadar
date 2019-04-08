
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as array
from pylab import mpl


# pr.append(ep)
# pr.append(tr_loss)
# pr.append(te_loss)
# pr.append(ac)

def train_result(epochs, test_loss, st1, st2, st3, file_name):

    for i in range(len(epochs)):
        epochs[i] = epochs[i]/15

    x = epochs

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(x, test_loss, color='gray', label='测试集loss')
    plt.plot(x, st1, color='black', label='标准一正确率', linestyle=':')
    plt.plot(x, st2, color='gray', label='标准二正确率', linestyle='-.')
    plt.plot(x, st3, color='gray', label='标准三正确率', linestyle='-.')
    plt.legend()  # 显示图例

    plt.xlabel(u"迭代次数")  # 指定x轴描述信息
    plt.ylabel(u"值")  # 指定y轴描述信息
    plt.title("损失函数-准确率变化曲线")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度
    plt.savefig('{}.pdf'.format(file_name))  # 保存为图片
    plt.show()


def read_file(file_path):
    f = open(file_path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    index = 0
    epochs = []
    test_loss = []
    st1 = []
    st2 = []
    st3 = []
    while line:
        print(index, '-------------------------------------------')

        ss = line.split('\t')
        epochs.append(int(ss[0]))

        test_loss_str = ss[2].strip()
        t_str = test_loss_str.split(':')
        test_loss.append(float(t_str[1].strip()))

        st1_str = ss[5].strip()
        st1_str = st1_str.split(':')
        st1.append(float(st1_str[1].strip()))

        st2_str = ss[6].strip()
        st2_str = st2_str.split(':')
        st2.append(float(st2_str[1].strip()))

        st3_str = ss[7].strip()
        st3_str = st3_str.split(':')
        st3.append(float(st3_str[1].strip()))

        # print(ss)
        line = f.readline()
        index = index + 1
    f.close()
    return epochs, test_loss, st1, st2, st3


if __name__ == '__main__':
    file_path = "D:\home\zeewei\projects\\77GRadar\exercise\\cnn_train.log"
    epochs, test_loss, st1, st2, st3 = read_file(file_path)
    train_result(epochs, test_loss, st1, st2, st3, "cnn准确率")
    print('st1: ', st1)
