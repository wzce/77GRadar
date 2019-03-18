import os, shutil
import sys
import numpy as np
from data_process import radar_data_decoder

ORIGIN_CAR_DATA_DIR = "D:\home\zeewei\\20190308\car_data"

# ORIGIN_CAR_DATA_DIR = "D:\home\zeewei\\20190308\classfication_data\cars"
ORIGIN_EMPTY_DATA_DIR = "D:\home\zeewei\\20190308\classfication_data\empty"


class ClassificationExtractor:
    def __init__(self, origin_empty_data_dir=ORIGIN_EMPTY_DATA_DIR,
                 origin_car_dir=ORIGIN_CAR_DATA_DIR):
        self.origin_empty_dir = origin_empty_data_dir
        self.origin_car_dir = origin_car_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()

    def load_data(self, data_path):
        file_lists = os.listdir(data_path)
        input_data = []

        for file in file_lists:
            target_file = os.path.join(data_path, file)
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
                for item in file_data:
                    input_data.append(item)
        return input_data

    def load_car_data(self):
        return self.load_data(self.origin_car_dir)

    def load_empty_data(self):
        return self.load_data(self.origin_empty_dir)


if __name__ == '__main__':
    extractor = ClassificationExtractor()
    car_data = extractor.load_car_data()
    print('car_data:', car_data)
