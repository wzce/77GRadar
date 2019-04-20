from data_process import extractor
from config import data_config
import random
import numpy as np
import os

config = data_config.DataConfig()

DATA_DIR = config.process_data_dir

PLAYGROUND_TRAIN_DATA_INPUT = os.path.join(DATA_DIR, config.train_data_input)
PLAYGROUND_TRAIN_DATA_LABEL = os.path.join(DATA_DIR, config.train_data_label)

PLAYGROUND_TEST_DATA_INPUT = os.path.join(DATA_DIR, config.test_data_input)
PLAYGROUND_TEST_DATA_LABEL = os.path.join(DATA_DIR, config.test_data_label)


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
        train_data_path = os.path.join(config.process_data_dir, config.train_data_file_name)
        data_list = np.load(train_data_path)
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


def load_val_data():
    val_data_path = os.path.join(config.process_data_dir, config.val_data_file_name)
    val_data = np.load(val_data_path)
    random.shuffle(val_data)

    val_data_input = []
    val_data_label = []
    for item in val_data:
        val_data_input.append(item[0])
        val_data_label.append(item[1])
    return val_data_input, val_data_label
