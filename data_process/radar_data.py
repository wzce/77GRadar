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
    if os.path.exists(PLAYGROUND_TRAIN_DATA_INPUT) \
            and os.path.exists(PLAYGROUND_TRAIN_DATA_LABEL) \
            and os.path.exists(PLAYGROUND_TEST_DATA_INPUT) \
            and os.path.exists(PLAYGROUND_TEST_DATA_LABEL):
        train_data_input = np.load(PLAYGROUND_TRAIN_DATA_INPUT)
        train_data_label = np.load(PLAYGROUND_TRAIN_DATA_LABEL)
        test_data_input = np.load(PLAYGROUND_TEST_DATA_INPUT)
        test_data_label = np.load(PLAYGROUND_TEST_DATA_LABEL)
    else:

        train_data_path = os.path.join(config.process_data_dir, config.train_data_file_name)
        if os.path.exists(train_data_path):
            data_list = np.load(train_data_path)
        else:
            data_extractor = extractor.FeatureExtractor(
                origin_data_dir=config.origin_train_data_dir,
                processed_data_dir=config.processed_data_dir,
                input_data_file_name=config.train_data_file_name)
            data_list = data_extractor.load_data()
        random.shuffle(data_list)  # 随机打乱
        train_num = int(7 * len(data_list) / 10)  # 训练集与测试集7:3比例,此处如果测试集是按帧取数据而不是按录取的数据组划分则这么做
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
    if os.path.exists(val_data_path):
        val_data = np.load(val_data_path)
    else:
        data_extractor = extractor.FeatureExtractor(
            origin_data_dir=config.origin_val_data_dir,
            processed_data_dir=config.processed_data_dir,
            input_data_file_name=config.val_data_file_name)
        val_data = data_extractor.load_data()
    random.shuffle(val_data)

    val_data_input = []
    val_data_label = []
    for item in val_data:
        val_data_input.append(item[0])
        val_data_label.append(item[1])
    return val_data_input, val_data_label


def generate_batch(input_data, label_data, batch_size=15000):
    batch_num = int(len(input_data) / batch_size)
    batch_data_input = []
    batch_data_label = []
    for i in range(batch_num):
        batch_input = input_data[batch_num * batch_size:(batch_num + 1) * batch_size]
        batch_label = label_data[batch_num * batch_size:(batch_num + 1) * batch_size]
        batch_data_input.append(batch_input)
        batch_data_label.append(batch_label)
    batch_data_input.append(input_data[batch_num * batch_size:])
    batch_data_label.append(label_data[batch_num * batch_size:])
    return batch_data_input, batch_data_label
