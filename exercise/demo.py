import numpy as np
import matplotlib.pyplot as plt
import random

right = [0, 21, 172, 177, 175, 176, 179, 204, 159, 118, 106, 80, 124, 43,
         64, 43, 51, 87, 87, 54, 73, 22, 19, 78, 52, 27, 64, 35, 1, 24, 63, 31, 2, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

all = [0, 21, 172, 193, 177, 196, 216, 232, 208, 194, 189, 175, 147, 128,
       106, 90, 85, 225, 204, 236, 212, 204, 200, 206, 210, 223, 217, 241,
       218, 207, 241, 218, 200, 51, 20, 33,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

rate = [0]

for i in range(1, 64-28):
    rate.append(right[i] / all[i])

x = [i for i in range(64-28)]
y = rate

# plt.plot(x, y)
# plt.plot(x, all[0:36])
# plt.plot(x, right[0:36])
#
# plt.title('predict rate')
# plt.xlabel('location')
# plt.ylabel('rate')
#
# plt.show()

x = [400,520,200]
print (400 in x)

def random_index(start, end, exclude_list, l):
    res = []
    while len(res) < l:
        rd = random.randint(start, end)
        if (rd in exclude_list) or (rd in res):
            continue
        else:
            res.append(rd)
    return res

exclude_list = [23,25]

list = random_index(0,63,exclude_list,6)
# print (list)

p = [1 , 0, 1,0 ,0]

d_max_index = p_max_index = p.index(max(p))

print ('d_max_index :', d_max_index)