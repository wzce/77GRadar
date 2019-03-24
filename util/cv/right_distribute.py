import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as array
from pylab import mpl


def distribute_cv(right, all, n):
    rate = []
    for i in range(n):
        if all[i] == 0:
            all[i] = 1
        rate.append(right[i] / all[i])

    x = [i for i in range(n)]

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    x_axis = x
    y_axis = rate
    plt.plot(x, rate)
    plt.bar(x_axis, y_axis, color='rgb')  # 如果不指定color，所有的柱体都会是一个颜色
    plt.xlabel(u"距离单元")  # 指定x轴描述信息
    plt.ylabel(u"准确率")  # 指定y轴描述信息
    plt.title("不同距离单元准确率分布图")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度
    # plt.savefig('{}.png'.format(time.strftime('%Y%m%d%H%M%S')))  # 保存为图片
    plt.show()
