from data_process import feature_extractor
from data_process import extractor
import random
import numpy as np
import os

# DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data_process\process_data_denoise_avg'
# DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data_process\processed_data'
# DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data_process\processed_two_days_all'
DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\data\\train_test_data'

PLAYGROUND_TRAIN_DATA_INPUT = os.path.join(DATA_DIR, 'pg_train_data_input.npy')
PLAYGROUND_TRAIN_DATA_LABEL = os.path.join(DATA_DIR, 'pg_train_data_label.npy')

PLAYGROUND_TEST_DATA_INPUT = os.path.join(DATA_DIR, 'pg_test_data_input.npy')
PLAYGROUND_TEST_DATA_LABEL = os.path.join(DATA_DIR, 'pg_test_data_label.npy')


def reduce_data_length(data, start, end):
    out = []
    for item in data:
        out.append(item[start:end])
    return out


def load_playground_data():
    if os.path.exists(PLAYGROUND_TRAIN_DATA_INPUT) and os.path.exists(PLAYGROUND_TRAIN_DATA_LABEL) and os.path.exists(
            PLAYGROUND_TEST_DATA_INPUT) and os.path.exists(PLAYGROUND_TEST_DATA_LABEL):
        train_data_input = np.load(PLAYGROUND_TRAIN_DATA_INPUT)
        train_data_label = np.load(PLAYGROUND_TRAIN_DATA_LABEL)
        test_data_input = np.load(PLAYGROUND_TEST_DATA_INPUT)
        test_data_label = np.load(PLAYGROUND_TEST_DATA_LABEL)
    else:
        # data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置
        # data_list = data_extractor.load_data()
        print('划分数据集')
        data_list = np.load('D:\home\zeewei\projects\\77GRadar\data\\all\\all_train_data.npy')
        random.shuffle(data_list)  # 随机打乱
        train_num = int(7 * len(data_list) / 10)  # 训练集与测试集7:3比例
        train_data = data_list[0:train_num]
        test_data = data_list[train_num:]

        train_data_input = []
        train_data_label = []
        for item in train_data:
            train_data_input.append(item[0])
            train_data_label.append(item[1])

        test_data_input = []
        test_data_label = []
        for item in test_data:
            test_data_input.append(item[0])
            test_data_label.append(item[1])

        np.save(PLAYGROUND_TRAIN_DATA_INPUT, train_data_input)
        np.save(PLAYGROUND_TRAIN_DATA_LABEL, train_data_label)
        np.save(PLAYGROUND_TEST_DATA_INPUT, test_data_input)
        np.save(PLAYGROUND_TEST_DATA_LABEL, test_data_label)
    return train_data_input, train_data_label, test_data_input, test_data_label


def load_pg_test_data():
    test_data_input = np.load(PLAYGROUND_TEST_DATA_INPUT)
    test_data_label = np.load(PLAYGROUND_TEST_DATA_LABEL)
    return test_data_input, test_data_label


def load_pg_data_by_range(start=0, end=63):
    '''
        start,开始的距离单元
        end，结束的距离单元位置
    '''
    pg_train_data_input_range = os.path.join(DATA_DIR, str(start) + '_' + str(end) + '_pg_train_data_input.npy')
    pg_train_data_label_range = os.path.join(DATA_DIR, str(start) + '_' + str(end) + '_pg_train_data_label.npy')

    pg_test_data_input_range = os.path.join(DATA_DIR, str(start) + '_' + str(end) + '_pg_test_data_input.npy')
    pg_test_data_label_range = os.path.join(DATA_DIR, str(start) + '_' + str(end) + '_pg_test_data_label.npy')

    if os.path.exists(pg_train_data_input_range) and os.path.exists(pg_train_data_label_range) and os.path.exists(
            pg_test_data_input_range) and os.path.exists(pg_test_data_label_range):
        train_data_input = np.load(pg_train_data_input_range)
        train_data_label = np.load(pg_train_data_label_range)
        test_data_input = np.load(pg_test_data_input_range)
        test_data_label = np.load(pg_test_data_label_range)
        return train_data_input, train_data_label, test_data_input, test_data_label

    # data_extractor = extractor.FeatureExtractor(input_data_file_name='input_data_all.npy')  # 此处全使用默认的文件路径配置
    # data_list = data_extractor.load_data()
    print('划分数据集')
    data_list = np.load('D:\home\zeewei\projects\\77GRadar\data\\all\\all_train_data.npy')
    remain_list = []
    '''
        对数据进行范围过滤
    '''
    for item in data_list:
        max_index = item[1].argmax(axis=0)  # 求每一行的最大下标，即目标的位置
        if max_index >= start and max_index <= end:
            remain_list.append(item)
    data_list = remain_list
    random.shuffle(data_list)  # 随机打乱
    train_num = int(7 * len(data_list) / 10)  # 训练集与测试集7:3比例
    train_data = data_list[0:train_num]
    test_data = data_list[train_num:]

    train_data_input = []
    train_data_label = []
    for item in train_data:
        train_data_input.append(item[0])
        train_data_label.append(item[1])

    test_data_input = []
    test_data_label = []
    for item in test_data:
        test_data_input.append(item[0])
        test_data_label.append(item[1])

    np.save(pg_train_data_input_range, train_data_input)
    np.save(pg_train_data_label_range, train_data_label)
    np.save(pg_test_data_input_range, test_data_input)
    np.save(pg_test_data_label_range, test_data_label)
    return train_data_input, train_data_label, test_data_input, test_data_label


def load_road_data():
    road_data_extractor = feature_extractor.FeatureExtractor()
    train_data_input, train_data_label = road_data_extractor.load_train_data()
    test_data_input, test_data_label = road_data_extractor.load_test_data()
    return train_data_input, train_data_label, test_data_input, test_data_label


def load_all_data():
    pg_data_extractor = extractor.FeatureExtractor()  # 此处全使用默认的文件路径配置

    data_list = pg_data_extractor.load_data()  # 获取操场数据
    data_list = data_list.tolist()
    random.shuffle(data_list)
    data_list = data_list[0:3000]

    input_data, label_data, test_data, test_label_data = load_road_data() #获取道路数据
    input_data = input_data.tolist()
    label_data = label_data.tolist()
    for i in range(len(input_data)):
        a_group_data = []
        a_group_data.append(input_data[i])
        a_group_data.append(label_data[i])
        data_list.append(a_group_data)

    for i in range(len(test_data)):
        a_group_data = []
        a_group_data.append(test_data[i])
        a_group_data.append(test_label_data[i])
        data_list.append(a_group_data)
    return data_list


def load_val_data(data_path):
    val_data = np.load(data_path)
    val_data_input = []
    val_data_label = []
    for item in val_data:
        val_data_input.append(item[0])
        val_data_label.append(item[1])
    return val_data_input,val_data_label


if __name__ == '__main__':
    data = load_pg_data_by_range(10, 30)
    print('data_list: ', data)
