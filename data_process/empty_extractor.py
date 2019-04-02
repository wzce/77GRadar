import os
import numpy as np
import random

from data_process import radar_data_decoder

ORIGIN_DATA_DIR = "D:\home\zeewei\\20190319\\radar_data"
PROCESSED_DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\\processed_data'
INPUT_DATA_FILE_NAME = 'all_two_lines_data.npy'  # 两条线路的全部训练数据

OUTPUT_LIST_LEN = 64
GAP = 3  # 距离分辨率，3m


class FeatureExtractor:
    def __init__(self, origin_data_dir=ORIGIN_DATA_DIR, processed_data_dir=PROCESSED_DATA_DIR,
                 input_data_file_name=INPUT_DATA_FILE_NAME):
        self.origin_data_dir = origin_data_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()
        self.processed_data_dir = processed_data_dir
        self.input_data_file_name = input_data_file_name

    def extract_feature_from_empty_goal(self):
        file_lists = os.listdir(self.origin_data_dir)
        file_data_list = []
        for file in file_lists:
            target_file = os.path.join(self.origin_data_dir, file)
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
                for item in file_data:
                    file_data_list.append(item)
            print("finish read file: ", target_file)

        return file_data_list

    def load_empty_data(self):
        process_file = os.path.join(self.processed_data_dir, self.input_data_file_name)
        if os.path.exists(process_file):
            print('read from processed numpy file--->: ', process_file)
            data_list = np.load(process_file)
            # label_data_list = np.load(label_data_file)
        else:
            print('there is no processed processed_data，read from origin file--->')
            data_list = self.extract_feature_from_empty_goal()
            random.shuffle(data_list)
            np.save(process_file, data_list)
            # np.save(label_data_file, label_data_list)
        return data_list

if __name__ == '__main__':
    empty_origin_data_dir = "D:\home\zeewei\\20190320\empty"
    save_data_name = "pg_empty_goal_data.npy"
    e = FeatureExtractor(origin_data_dir=empty_origin_data_dir,
                         input_data_file_name=save_data_name)
    list = e.load_empty_data()
    print('input_data_list: ', len(list))
