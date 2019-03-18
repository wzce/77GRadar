import numpy as np
import math
import matplotlib.pyplot as plt  # plt 用于显示图片
from mpl_toolkits.mplot3d import Axes3D  # 显示3D用
import os
import sys


def reverse(data_list):
    l = len(data_list)
    for index in range(int(l / 2)):
        tmp = data_list[l - index - 1]
        data_list[l - index - 1] = data_list[index]
        data_list[index] = tmp
    return data_list


def generate_speed_and_distance_img_3d(frame_chunk, frame_index):
    X = []  # 距离
    Y = []  # 速度
    Z = []  # 亮度，代表点
    for row_index in range(0, len(frame_chunk)):
        for col_index in range(0, len(frame_chunk[row_index])):
            X.append(col_index)
            Y.append(row_index)
            Z.append(frame_chunk[row_index][col_index] / 1000)
    fig = plt.figure()
    # plt.clf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Strength')
    # plt.savefig(IMG_SAVE_PATH + "speed_dis_" + str(frame_index) + ".png")
    plt.show()


def generate_speed_and_distance_img_2d(frame_chunk, frame_index):
    new_frame_chunk = []
    for row_index in range(0, len(frame_chunk)):
        new_frame_chunk.append(frame_chunk[63 - row_index])

    plt.imshow(new_frame_chunk)  # 显示图片
    plt.title('the ' + str(frame_index) + ' frame')
    # plt.axis('off')  # 不显示坐标轴
    my_x_ticks = np.arange(0, 512, 16)
    my_y_ticks = np.arange(0, 64, 4)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    # plt.savefig(IMG_SAVE_PATH + "speed_dis_" + str(frame_index) + ".png")
    plt.show()


class RadarDataDecoder:
    FRAME_SIZE = 512 * 64 * 4 + 40

    def __init__(self, integer_reverse=True):
        # self.FRAME_SIZE = 512 * 64 * 4 + 40
        self.integer_reverse = integer_reverse

    def re_arrange_bit_file(self, file_name=''):
        UInt8 = np.fromfile(file_name, dtype="uint8")  # 按Byte读取数字
        frame_num = int(len(UInt8) / self.FRAME_SIZE)
        # print('frame_num: ', frame_num)
        zero_speed_data = []
        for frame_index in range(1, frame_num + 1):
            # 获取一帧的数据，数据量 512*128/2 + 10 =(512*64)
            frame_data = UInt8[(frame_index - 1) * self.FRAME_SIZE: frame_index * self.FRAME_SIZE]
            frame_data = frame_data[29:-11]  # 去掉帧头和帧尾

            if self.integer_reverse:
                frame_data = reverse(frame_data)
            frame_data.dtype = 'uint32'
            if self.integer_reverse:
                frame_data = reverse(frame_data)

            frame_arr = np.array(frame_data).reshape(512, 64)
            frame_chunk = []
            big_chunk = []  # 64个8*8连接起来的矩阵块
            for sml_chunk_index in range(0, 512):  # 共划分为512/64个部分，1-64，每隔64是一大块
                sml_chunk = np.array(frame_arr[sml_chunk_index]).reshape(8, 8)
                sml_chunk_r = np.transpose(sml_chunk).tolist()  # 转置
                for i in range(0, len(sml_chunk_r)):
                    big_chunk.append(sml_chunk_r[i])

                if (sml_chunk_index + 1) % 64 == 0:  # 完成一个大块的，64个小块的重排转置集成, 进行集合重置，同时，将当前大块并进帧块中去
                    if len(frame_chunk) == 0:  # 第一个大块
                        for s_t in big_chunk:
                            frame_chunk.append(s_t)
                    else:  # 非第一个大块，则需要进行拼接,矩阵横向拼接
                        new_frame_chunk = []
                        for row_index in range(0, len(big_chunk)):
                            row_data = frame_chunk[row_index]
                            for v in big_chunk[row_index]:
                                row_data.append(v)
                            new_frame_chunk.append(row_data)
                        frame_chunk = new_frame_chunk
                    big_chunk = []

            positive_frame = frame_chunk[0:256]

            negative_frame = frame_chunk[256:]
            new_frame_chunk = []
            positive_frame[1] = [i - i for i in range(64)]
            negative_frame[255] = [i - i for i in range(64)]
            for d in negative_frame:
                new_frame_chunk.append(d)
            for d in positive_frame:
                new_frame_chunk.append(d)

            frame_chunk = new_frame_chunk
            zero_speed_data.append(frame_chunk[256])
        return zero_speed_data
