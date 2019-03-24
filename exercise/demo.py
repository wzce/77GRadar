import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as array
from pylab import mpl

right = [0, 16, 98, 103, 95, 96, 104, 107, 84, 62, 73, 42, 49, 16, 25, 19, 26, 61, 48, 25, 31, 9, 9, 37, 22, 14, 24, 5,
         0, 17, 23, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
all = [0, 16, 98, 109, 95, 109, 114, 109, 110, 105, 110, 102, 55, 60, 45, 36, 47, 123, 118, 117, 106, 111, 103, 100,
       112, 71, 109, 101, 118, 118, 110, 121, 104, 25, 7, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0]

rate = [0]

for i in range(1, 64):
    if all[i] == 0:
        all[i] = 1
    rate.append(right[i] / all[i])

x = [i for i in range(64)]
y = rate

# plt.plot(x, rate)
# # plt.plot(x, all[0:36])
# # plt.plot(x, right[0:36])
#
# plt.title('predict rate')
# plt.xlabel('location')
# plt.ylabel('rate')
#
# plt.show()

import time

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
source_data = {'mock_verify': 369, 'mock_notify': 192, 'mock_sale': 517}  # 设置原始数据
# for a, b in source_data.items():
#     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)  # ha 文字指定在柱体中间， va指定文字位置 fontsize指定文字体大小

# 设置X轴Y轴数据，两者都可以是list或者tuple
# x_axis = tuple(source_data.keys())
# y_axis = tuple(source_data.values())

x_axis = x
y_axis = rate
plt.plot(x, rate)
plt.bar(x_axis, y_axis, color='rgb')  # 如果不指定color，所有的柱体都会是一个颜色

plt.xlabel(u"距离单元")  # 指定x轴描述信息
plt.ylabel(u"位置")  # 指定y轴描述信息
plt.title("不同距离单元准确率分布图")  # 指定图表描述信息
plt.ylim(0, 1)  # 指定Y轴的高度
# plt.savefig('{}.png'.format(time.strftime('%Y%m%d%H%M%S')))  # 保存为图片

# plt.show()


