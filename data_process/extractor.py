import os, shutil
import sys
import numpy as np
import random
from configparser import ConfigParser

# from data_process import radar_data_decode
from data_process import radar_data_decoder
import config

OUTPUT_LIST_LEN = 64
GAP = 3  # 距离分辨率，3m


class FeatureExtractor:
    def __init__(self, origin_data_dir, processed_data_dir, input_data_file_name):
        self.origin_data_dir = origin_data_dir
        self.radar_data_decoder = radar_data_decoder.RadarDataDecoder()
        self.processed_data_dir = processed_data_dir
        self.input_data_file_name = input_data_file_name

    def generate_goal_location_list(self, full_path, output_list_len=OUTPUT_LIST_LEN, gap=GAP):
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
                goal_location_data = self.generate_goal_location_list(target_file)
            if file[-4:] == '.dat':  # 找出所有含数据的文件夹
                file_data = self.radar_data_decoder.re_arrange_bit_file(target_file)
                for item in file_data:
                    file_data_list.append(item)

        return file_data_list, goal_location_data

    def load_static_radar_data(self):
        input_data_list = []
        # label_data_list = []
        data_item_list = os.listdir(self.origin_data_dir)
        print('data_item_list: ', data_item_list)
        index = 1
        for item_path in data_item_list:
            data_path = os.path.join(self.origin_data_dir, item_path)
            file_data, goal_location_data = self.extract_feature_from_a_goal(data_path)
            print(str(index) + ' finish reading a origin file : ', data_path)
            for i in range(len(file_data)):
                # if i > 70:
                #     break
                frame = file_data[i]
                a_group_data = []  # 一组数据，0,1两列，第一列是信号强度，第二列是标注数据，位置数据
                a_group_data.append(frame)
                a_group_data.append(goal_location_data)
                input_data_list.append(a_group_data)
                # label_data_list.append(distance_data[0])
            index = index + 1

        return input_data_list

    def load_data(self):
        process_file = os.path.join(self.processed_data_dir, self.input_data_file_name)
        if os.path.exists(process_file):
            print('read from processed numpy file--->: ', process_file)
            data_list = np.load(process_file)
            # label_data_list = np.load(label_data_file)
        else:
            print('there is no processed processed_data，read from origin file--->')
            data_list = self.load_static_radar_data()
            random.shuffle(data_list)
            np.save(process_file, data_list)
            # np.save(label_data_file, label_data_list)
        return data_list

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
        else:
            print('there is no processed processed_data，read from origin file--->')
            data_list = self.extract_feature_from_empty_goal()
            random.shuffle(data_list)
            np.save(process_file, data_list)
        return data_list


if __name__ == '__main__':
    cp, section = config.load_config(2)

    e = FeatureExtractor(cp.get(section, 'origin_data_dir'),
                         cp.get(section, 'processed_data_dir'),
                         cp.get(section, 'train_data_file_name'))

    list = e.load_data()
    print('input_data_list: ', len(list))
