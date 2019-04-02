import os, shutil
import sys
import numpy as np
import random

# from data_process import radar_data_decode
from data_process import radar_data_decoder

ORIGIN_DATA_DIR = "D:\home\zeewei\\20190319\\radar_data\\"
# ORIGIN_DATA_DIR = "D:\home\zeewei\\20190324\\"
# ORIGIN_TEST_DATA_DIR = "D:\home\zeewei\\20190308\ml_backup\\test_data"

# PROCESSED_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data_process\pg_data_avg'
PROCESSED_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\\04_01_data'
# PROCESSED_TEST_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data\\test\\'
# INPUT_DATA_FILE_NAME2 = 'input_data_50.npy'
# INPUT_DATA_FILE_NAME = '2019_03_28_input_data_denoise_avg_10.npy'
# INPUT_DATA_FILE_NAME = '2019_03_24_input_data_all.npy'

INPUT_TRAIN_DATA_FILE_NAME = '2019_04_01_input_data_all.npy'
INPUT_TEST_DATA_FILE_NAME = '2019_04_01_test_label_all.npy'

INPUT_DATA_ALL = 'input_data_all.npy'
# OUT_DATA_FILE_NAME = 'label.npy'

# SHORT_LINE = 2  # 分道线，白色的条状，长2m
# LONG_LINE = 4  # 分道线，白色的条状的间隔，长4m
# ORIGIN_DIS = 1  # 雷达距离第一个白色线初始距离，此处为1m
OUTPUT_LIST_LEN = 64
GAP = 3  # 距离分辨率，3m

input_data = []
test_data = []


class FeatureExtractor:
    def __init__(self, origin_data_dir=ORIGIN_DATA_DIR, processed_data_dir=PROCESSED_DATA_DIR,
                 input_data_file_name=INPUT_TRAIN_DATA_FILE_NAME):
        self.origin_data_dir = origin_data_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()
        self.processed_data_dir = processed_data_dir
        self.input_data_file_name = input_data_file_name

    def generate_goal_location_list(self, full_path, output_list_len=OUTPUT_LIST_LEN, gap=GAP):
        # print('full_path: ',full_path)
        '''
                生成目标所在的位置（1-64 中的位置）
        '''
        distance_list = []
        with open(full_path, "r") as f:  # 设置文件对象
            line = f.read()  # 可以是随便对文件的操作
            items = line.split(' ')
            for s in items:
                dis = float(s)
                distance_list.append(dis)

        result = [i - i for i in range(output_list_len)]
        for val in distance_list:
            index = (int)(val / gap)
            result[index] = 1
        return result, dis

    # def generate_goal_location_list(self, distance_list, output_list_len=OUTPUT_LIST_LEN, gap=GAP):

    '''
        读取一组数据中的目标位置和信号强度
    '''

    def extract_feature_from_a_goal(self, goal_data_folder_path):
        file_lists = os.listdir(goal_data_folder_path)
        file_data_list = []
        goal_location_data = []
        for file in file_lists:
            target_file = os.path.join(goal_data_folder_path, file)
            if file[-4:] == '.txt':
                goal_location_data, dis = self.generate_goal_location_list(target_file)
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
                for item in file_data:
                    file_data_list.append(item)

        return file_data_list, goal_location_data, dis

    def load_static_radar_data(self):
        input_data_list = []
        tets_data_list = []
        data_item_list = os.listdir(self.origin_data_dir)
        print('data_item_list: ', data_item_list)
        index = 1
        for item_path in data_item_list:
            data_path = os.path.join(self.origin_data_dir, item_path)
            file_data, goal_location_data, dis = self.extract_feature_from_a_goal(data_path)
            print(str(index) + ' finish reading a origin file : ', data_path)
            for i in range(len(file_data)):
                # if i > 70:
                #     break
                frame = file_data[i]
                a_group_data = []  # 一组数据，0,1两列，第一列是信号强度，第二列是标注数据，位置数据
                a_group_data.append(frame)
                a_group_data.append(goal_location_data)

                if (int(dis - 5.5)) % 3 == 0:
                    input_data_list.append(a_group_data)
                elif (int(dis - 6.5)) % 3 == 0:
                    tets_data_list.append(a_group_data)
                # label_data_list.append(distance_data[0])
            index = index + 1

        return input_data_list,tets_data_list

    def load_data(self):
        process_file = os.path.join(self.processed_data_dir, self.input_data_file_name)
        if os.path.exists(process_file):
            print('read from processed numpy file--->: ', process_file)
            data_list = np.load(process_file)
            # label_data_list = np.load(label_data_file)
        else:
            print('there is no processed data，read from origin file--->')
            data_list = self.load_static_radar_data()
            np.save(process_file, data_list)
            # np.save(label_data_file, label_data_list)
        return data_list


def relist_data():
    extractor = FeatureExtractor(input_data_file_name='input_data_50.npy')
    input_data_list = extractor.load_data()
    random.shuffle(input_data_list)  # 随机打乱
    train_data_num = 9 * (int(len(input_data_list) / 10))
    train_data = input_data_list[0:train_data_num]
    val_data = input_data_list[train_data_num:]

    process_file = os.path.join('D:\home\zeewei\projects\\77GRadar\data\\all',
                                '2019_03_19_train_data.npy')
    np.save(process_file, train_data)


if __name__ == '__main__':
    # relist_data()
    e = FeatureExtractor()
    input_data_list, test_data_list = e.load_static_radar_data()
    np.save("D:\home\zeewei\projects\\77GRadar\\04_01_data\\"+INPUT_TRAIN_DATA_FILE_NAME, input_data_list)
    np.save("D:\home\zeewei\projects\\77GRadar\\04_01_data\\"+INPUT_TEST_DATA_FILE_NAME,test_data_list)
    print('input_data_list: ', len(input_data_list))
