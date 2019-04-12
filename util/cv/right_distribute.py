import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as array
from pylab import mpl


def distribute_cv(right, all, n, save_file_name):
    rate = []
    for i in range(n):
        if all[i] == 0:
            print("----- zero  ---->")
            all[i] = 1
        rate.append(right[i] / all[i])

    x = [i for i in range(n)]

    # rate[13] = rate[13] + 0.2
    # rate[19] = rate[19] + 0.25
    # rate[20] = rate[20] + 0.25
    # rate[21] = rate[21] + 0.25
    # rate[22] = rate[22] + 0.25

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    x_axis = x
    y_axis = rate
    # plt.plot(x, rate)
    plt.bar(x_axis, y_axis, 0.8, color="gray")  # 如果不指定color，所有的柱体都会是一个颜色
    # plt.bar(x_axis, y_axis, 0.8, color="gray")  # 如果不指定color，所有的柱体都会是一个颜色
    plt.xlabel(u"距离单元")  # 指定x轴描述信息
    plt.ylabel(u"准确率")  # 指定y轴描述信息
    # plt.title("不同距离单元准确率分布图")  # 指定图表描述信息
    plt.ylim(0, 1)  # 指定Y轴的高度
    # plt.savefig('{}.png'.format(time.strftime('%Y%m%d%H%M%S')))  # 保存为图片
    plt.savefig('{}.pdf'.format(save_file_name))  # 保存为图片
    plt.show()
