import os, shutil
import sys
import numpy as np

# from data_process import radar_data_decode
from data_process import radar_data_decoder

ORIGIN_DATA_DIR = "D:\home\zeewei\\20190320\empty"
# ORIGIN_TEST_DATA_DIR = "D:\home\zeewei\\20190308\ml_backup\\test_data"

PROCESSED_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data_process\playground_data'
# PROCESSED_TEST_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data\\test\\'
INPUT_DATA_FILE_NAME = 'input_data_empty.npy'
# OUT_DATA_FILE_NAME = 'label.npy'

# SHORT_LINE = 2  # 分道线，白色的条状，长2m
# LONG_LINE = 4  # 分道线，白色的条状的间隔，长4m
# ORIGIN_DIS = 1  # 雷达距离第一个白色线初始距离，此处为1m
OUTPUT_LIST_LEN = 64
GAP = 3  # 距离分辨率，3m


class PGEmptyFeatureExtractor:
    def __init__(self, origin_data_dir=ORIGIN_DATA_DIR):
        self.origin_data_dir = origin_data_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()

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
                distance_list.append(float(s))

        result = [i - i for i in range(output_list_len)]
        for val in distance_list:
            index = (int)(val / gap)
            result[index] = 1
        return result

    # def generate_goal_location_list(self, distance_list, output_list_len=OUTPUT_LIST_LEN, gap=GAP):

    '''
        读取一组数据中的目标位置和信号强度
    '''

    def extract_feature_from_a_goal(self, goal_data_folder_path):
        file_lists = os.listdir(goal_data_folder_path)
        file_data = []
        goal_location_data = []
        for file in file_lists:
            target_file = os.path.join(goal_data_folder_path, file)
            if file[-4:] == '.txt':
                goal_location_data = self.generate_goal_location_list(target_file)
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)

        return file_data, goal_location_data

    def load_static_radar_data(self):
        # 直接读取文件
        input_data_list = []
        # label_data_list = []
        data_item_list = os.listdir(self.origin_data_dir)
        print('data_item_list: ', data_item_list)
        index = 1
        for item_path in data_item_list:
            target_file = os.path.join(self.origin_data_dir, item_path)
            file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
            print(str(index) + ' finish reading a origin file : ', target_file)
            for i in range(len(file_data)):
                # if i > 50:
                #     break
                frame = file_data[i]
                # a_group_data = []  # 一组数据，0,1两列，第一列是信号强度，第二列是标注数据，位置数据
                # a_group_data.append(frame)
                # a_group_data.append(goal_location_data)
                input_data_list.append(frame)
                # label_data_list.append(distance_data[0])
            index = index + 1

        return input_data_list

    def load_data(self, ):
        process_file = os.path.join(PROCESSED_DATA_DIR, INPUT_DATA_FILE_NAME)
        if os.path.exists(process_file):
            print('read from processed numpy file--->')
            data_list = np.load(process_file)
        else:
            print('there is no processed data，read from origin file--->')
            data_list = self.load_static_radar_data()
            np.save(process_file, data_list)
        return data_list


if __name__ == '__main__':
    extractor = PGEmptyFeatureExtractor()
    input_data_list = extractor.load_data()
    print('input_data_list: ', len(input_data_list))
