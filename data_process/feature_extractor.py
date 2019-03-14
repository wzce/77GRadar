import os, shutil
import sys
import numpy as np

# from data_process import radar_data_decode
from data_process import radar_data_decoder
ORIGIN_TRAIN_DATA_DIR = "D:\home\zeewei\\20190308\ml\\train_data"
ORIGIN_TEST_DATA_DIR = "D:\home\zeewei\\20190308\ml\\test_data"

PROCESSED_TRAIN_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data\\train\\'
PROCESSED_TEST_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data\\test\\'
INPUT_DATA_FILE_NAME = 'input.npy'
OUT_DATA_FILE_NAME = 'label.npy'

SHORT_LINE = 2  # 分道线，白色的条状，长2m
LONG_LINE = 4  # 分道线，白色的条状的间隔，长4m
ORIGIN_DIS = 1  # 雷达距离第一个白色线初始距离，此处为1m
OUTPUT_LIST_LEN = 64
GAP = 3  # 距离分辨率，3m

TYPE_LOAD_TRAIN = 'LOAD_TRAIN'
TYPE_LOAD_TEST = 'LOAD_TEST'


class FeatureExtractor:
    def __init__(self, origin_train_data_dir=ORIGIN_TRAIN_DATA_DIR,
                 origin_test_data_dir=ORIGIN_TEST_DATA_DIR,
                 processed_train_data_dir=PROCESSED_TRAIN_DATA_DIR,
                 processed_test_data_dir=PROCESSED_TEST_DATA_DIR):
        self.origin_train_data_dir = origin_train_data_dir
        self.origin_test_data_dir = origin_test_data_dir
        self.processed_train_data_dir = processed_train_data_dir
        self.processed_test_data_dir = processed_test_data_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()

    def cal_distance(self, short_line_num, long_line_num):
        distance = short_line_num * SHORT_LINE + long_line_num * LONG_LINE + ORIGIN_DIS
        return distance

    def cal_distance_from_file(self, full_path):
        # print('full_path: ',full_path)
        data = []
        with open(full_path, "r") as f:  # 设置文件对象
            line = f.read()  # 可以是随便对文件的操作
            strs = line.split(' ')
            for s in strs:
                data.append(int(s))
        distance = self.cal_distance(data[0], data[1])
        if distance == 1:
            distance = 0
        dis_list = [distance]
        return self.generate_output_list(dis_list)

    def generate_output_list(self, dis_list, output_list_len=OUTPUT_LIST_LEN, gap=GAP):
        result = [i - i for i in range(output_list_len)]
        for val in dis_list:
            index = (int)(val / gap)
            if index == 0:
                index = 1
            if val != 0:
                result[index - 1] = 1
        return [result]

    def extract_feature_from_a_distance(self, a_data_folder_path):
        file_lists = os.listdir(a_data_folder_path)
        file_data = []
        distance_data = []

        for file in file_lists:
            target_file = os.path.join(a_data_folder_path, file)
            if file[-4:] == '.txt':
                distance_data = self.cal_distance_from_file(target_file)
                # continue
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
                # continue

        return file_data, distance_data

    def load_static_radar_data(self, data_folder_path):
        input_data_list = []
        label_data_list = []
        data_item_list = os.listdir(data_folder_path)
        for item_path in data_item_list:
            data_path = os.path.join(data_folder_path, item_path)
            file_data, distance_data = self.extract_feature_from_a_distance(data_path)
            print('finish reading a origin file : ', data_path)
            for frame in file_data:
                input_data_list.append(frame)
                label_data_list.append(distance_data[0])

        return input_data_list, label_data_list

    def load_data(self, load_type):
        if TYPE_LOAD_TEST == load_type:
            processed_dir = self.processed_test_data_dir
            origin_data_dir = self.origin_test_data_dir
        else:
            processed_dir = self.processed_train_data_dir
            origin_data_dir = self.origin_train_data_dir

        input_data_file = os.path.join(processed_dir, INPUT_DATA_FILE_NAME)
        label_data_file = os.path.join(processed_dir, OUT_DATA_FILE_NAME)
        if os.path.exists(input_data_file) & os.path.exists(label_data_file):
            print('read from processed numpy file--->')
            input_data_list = np.load(input_data_file)
            label_data_list = np.load(label_data_file)
        else:
            print('there is no processed data，read from origin file--->')
            input_data_list, label_data_list = self.load_static_radar_data(origin_data_dir)
            np.save(input_data_file, input_data_list)
            np.save(label_data_file, label_data_list)
        return input_data_list, label_data_list

    def load_train_data(self):
        print('start fetching train data--->')
        input_data_list, label_data_list = self.load_data(TYPE_LOAD_TRAIN)
        return input_data_list, label_data_list

    def load_test_data(self):
        print('start fetching test data--->')
        return self.load_data(TYPE_LOAD_TEST)
