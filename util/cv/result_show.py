import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as array
from pylab import mpl


# pr.append(ep)
# pr.append(tr_loss)
# pr.append(te_loss)
# pr.append(ac)

def train_result(data_path, file_name):
    data = np.load(data_path)

    # x = data[0]

    x = [i * 0.01 for i in range(1, 100)]

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(x, data[0], color='gray', label='标准一正确率')
    plt.plot(x, data[1], color='black', label='标准二正确率', linestyle=':')
    plt.plot(x, data[2], color='gray', label='标准三正确率', linestyle='-.')
    plt.legend()  # 显示图例

    my_x_ticks = np.arange(0, 1, 0.1)
    my_y_ticks = np.arange(0, 1, 0.1)

    plt.xlabel(u"阈值")  # 指定x轴描述信息
    plt.ylabel(u"准确率")  # 指定y轴描述信息
    # plt.title("损失函数-准确率变化曲线")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度

    # plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.savefig('{}.pdf'.format(file_name))  # 保存为图片
    plt.show()


def train_process_loss_correct_rate(data_path, file_name, data_len=100):
    data = np.load(data_path)

    x = data[0]
    xx = []
    for i in range(len(x)):
        xx.append(int(x[i] / 25))
    x = xx[:data_len]

    # x = [i * 0.01 for i in range(1, data_len)]

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(x, data[2][:data_len], color='gray', label='测试集loss')
    plt.plot(x, data[3][:data_len], color='black', label='标准一正确率', linestyle=':')
    plt.plot(x, data[4][:data_len], color='gray', label='标准二正确率', linestyle='-.')
    plt.plot(x, data[5][:data_len], color='black', label='标准三正确率', linestyle='--')

    # plt.plot(x, data[1][:data_len], color='gray', label='训练集loss')
    # plt.plot(x, d ata[2][:data_len], color='black', label='测试集loss', linestyle=':')
    # plt.plot(x, data[3][:data_len], color='black', label='分类准确率', linestyle='-.')

    plt.legend(loc=2)  # 显示图例

    my_x_ticks = np.arange(0, 1, 0.1)
    my_y_ticks = np.arange(0, 1, 0.1)

    plt.xlabel(u"训练迭代数")  # 指定x轴描述信息
    plt.ylabel(u"准确率")  # 指定y轴描述信息
    # plt.title("损失函数-准确率变化曲线")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度

    # plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.savefig('{}.pdf'.format(file_name))  # 保存为图片
    plt.show()


if __name__ == '__main__':
    # train_process_loss_correct_rate("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\classification_loss_rate.npy",
    #                                 'cnn_classification_and_los', data_len=300)
    train_process_loss_correct_rate("D:\home\zeewei\projects\\77GRadar\model\\rnn\\train_process_0409_2.npy",
                 'rnn_result',data_len=72)

    # train_result("D:\home\zeewei\projects\\77GRadar\model\cnn\cnn_0410_2_correct_st.npy","cnn_dif_std")
