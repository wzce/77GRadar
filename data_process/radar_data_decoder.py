import numpy as np


def reverse(data_list):
    length = len(data_list)
    for index in range(int(length / 2)):
        tmp = data_list[length - index - 1]
        data_list[length - index - 1] = data_list[index]
        data_list[index] = tmp
    return data_list


class RadarDataDecoder:
    FRAME_SIZE = 512 * 64 * 4 + 40

    def __init__(self, integer_reverse=True):
        self.integer_reverse = integer_reverse

    def re_arrange_bit_file(self, file_name=''):
        data_int8 = np.fromfile(file_name, dtype="uint8")  # 按Byte读取数字
        frame_num = int(len(data_int8) / self.FRAME_SIZE)
        zero_speed_data = []
        for frame_index in range(1, frame_num + 1):
            # 获取一帧的数据，数据量 512*128/2 + 10 =(512*64)
            frame_data = data_int8[(frame_index - 1) * self.FRAME_SIZE: frame_index * self.FRAME_SIZE]
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
