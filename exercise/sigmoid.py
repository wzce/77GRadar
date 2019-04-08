import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))


def plot_sigmoid():
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # param:起点，终点，间距
    x = np.arange(-40, 40, 0.2)
    y = sigmoid(x)
    y2= [0.1 for i in range(len(x))]
    y3 = [0.2 for i in range(len(x))]
    plt.plot(x, y,color='black', linewidth=1, markersize=6)
    plt.plot(x, y2,linestyle='dashed',color='black')
    # plt.plot(x, y3, linestyle='dashed',color='black',marker='*')

    # 线的标签
    plt.legend(('sigmoid 曲线 ', ' y = 0.2'), loc='upper right')
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()