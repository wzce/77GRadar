# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys

path1 = 'D:\home\zeewei\\20190319\\radar_data\\2_108_1'  # 所需修改文件夹所在路径
dirs = os.listdir(path1)

i = 0
for dir in dirs:
    if dir[0:5] == 'road1':
        os.rename(os.path.join(path1, str(dir)), os.path.join(path1, 'road1_' + str(dir[4:5]) + '_' + str(dir[5:])))
        print ("重命名成功: , ", dir)
        i += 1

