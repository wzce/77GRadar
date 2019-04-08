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

    x = data[0]

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(x, data[2], color='gray', label='测试集loss')
    plt.plot(x, data[3], color='black', label='完全匹配正确率', linestyle=':')
    plt.plot(x, data[4], color='gray', label='相对匹配正确率', linestyle='-.')
    plt.legend()  # 显示图例

    plt.xlabel(u"迭代次数")  # 指定x轴描述信息
    plt.ylabel(u"值")  # 指定y轴描述信息
    plt.title("损失函数-准确率变化曲线")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度
    plt.savefig('{}.pdf'.format(file_name))  # 保存为图片
    plt.show()


if __name__ == '__main__':
    train_result("D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\\train_cnn2_1.npy", 'cnn_result')
