import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片


def reverse(data_list):
    l = len(data_list)
    for index in range(int(l / 2)):
        tmp = data_list[l - index - 1]
        data_list[l - index - 1] = data_list[index]
        data_list[index] = tmp
    return data_list


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
            # static_goal = frame_chunk[254:258]  # 取相对静止的四组值
            # mean_col = np.mean(static_goal, axis=0)  # 取均值

            zero_speed_data.append(frame_chunk[256])

        return zero_speed_data


class RadarDataDecoderDeNoise(RadarDataDecoder):
    def __init__(self, integer_reverse=True):
        '''
                  计算每一列的均值,然后每一列减去均值
        '''
        # self.FRAME_SIZE = 512 * 64 * 4 + 40
        super(RadarDataDecoder, self).__init__()
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

        # 去噪方式一：对zero_speed_data进行去噪，每个数字减去均值
        '''
            计算每一列的均值,然后每一列减去均值
        '''
        mean_col = np.mean(zero_speed_data, axis=0)  #
        for i in range(len(zero_speed_data)):
            for j in range(len(zero_speed_data[i])):
                zero_speed_data[i][j] = zero_speed_data[i][j] - mean_col[j]
        return zero_speed_data


class RadarDataDecoderDeNoiseAvg(RadarDataDecoder):
    def __init__(self, integer_reverse=True, avg_len=10, avg_one=True):
        '''
            去噪方式二：
                计算每一列的均值,然后每一列减去均值
                avg_len: 多少帧取一个均值
        '''
        # self.FRAME_SIZE = 512 * 64 * 4 + 40
        super(RadarDataDecoder, self).__init__()
        self.integer_reverse = integer_reverse
        self.avg_len = avg_len
        self.avg_one = avg_one

    def re_arrange_bit_file(self, file_name=''):
        UInt8 = np.fromfile(file_name, dtype="uint8")  # 按Byte读取数字
        frame_num = int(len(UInt8) / self.FRAME_SIZE)
        # print('frame_num: ', frame_num)
        zero_speed_data = []

        frame_count = 0
        tmp_data = []
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
            tmp_data.append(frame_chunk[256])
            frame_count = frame_count + 1
            if frame_count % self.avg_len == 0:
                mean_col = np.mean(tmp_data, axis=0)
                if self.avg_one:
                    zero_speed_data.append(mean_col)
                else:
                    for i in range(len(tmp_data)):
                        for j in range(len(tmp_data[i])):
                            tmp_data[i][j] = tmp_data[i][j] - mean_col[j]
                        zero_speed_data.append(tmp_data[i])
                frame_count = 0
                tmp_data = []
            # zero_speed_data.append(frame_chunk[256])

        return zero_speed_data
