from data_process import empty_extractor
import random
import numpy as np
import os

DATA_DIR = 'D:\home\zeewei\projects\\77GRadar\classification_train_data'

PLAYGROUND_TRAIN_DATA_INPUT = os.path.join(DATA_DIR, 'pg_train_data.npy')
PLAYGROUND_TRAIN_DATA_LABEL = os.path.join(DATA_DIR, 'pg_train_label.npy')

PLAYGROUND_TEST_DATA_INPUT = os.path.join(DATA_DIR, 'pg_test_data.npy')
PLAYGROUND_TEST_DATA_LABEL = os.path.join(DATA_DIR, 'pg_test_label.npy')


def load_playground_data():
    if os.path.exists(PLAYGROUND_TRAIN_DATA_INPUT) and os.path.exists(PLAYGROUND_TRAIN_DATA_LABEL) and os.path.exists(
            PLAYGROUND_TEST_DATA_INPUT) and os.path.exists(PLAYGROUND_TEST_DATA_LABEL):
        train_data = np.load(PLAYGROUND_TRAIN_DATA_INPUT)
        train_label = np.load(PLAYGROUND_TRAIN_DATA_LABEL)
        test_data = np.load(PLAYGROUND_TEST_DATA_INPUT)
        test_label = np.load(PLAYGROUND_TEST_DATA_LABEL)
        return train_data, train_label, test_data, test_label

    empty_origin_data_dir = "D:\home\zeewei\\20190320\empty"
    save_data_name = "pg_empty_goal_data.npy"
    empty_ex = empty_extractor.FeatureExtractor(origin_data_dir=empty_origin_data_dir,
                                                input_data_file_name=save_data_name)
    empty_data = empty_ex.load_empty_data()  # 操场空数据
    random.shuffle(empty_data)

    car_data = np.load("D:\home\zeewei\projects\\77GRadar\processed_data\\all_two_lines_data.npy")
    random.shuffle(car_data)
    car_data = car_data[0:len(empty_data)]

    car = []
    for item in car_data:
        car.append(item[0])

    car_data = car
    car_train_data_len = int(2 * len(car_data) / 3)
    empty_train_data_len = int(2 * len(empty_data) / 3)
    train_data = car_data[0:car_train_data_len]
    empty_train_data = empty_data[0:empty_train_data_len]
    train_label = [1 for i in range(len(train_data))]
    for item in empty_train_data:
        train_data.append(item)
        train_label.append(0)

    test_data = car_data[car_train_data_len:len(car_data)]
    test_label = [1 for i in range(len(test_data))]

    np.save("D:\home\zeewei\projects\\77GRadar\classification_train_data\\pg_car_val_data.numpy", test_data)
    np.save("D:\home\zeewei\projects\\77GRadar\classification_train_data\\pg_car_val_label.numpy", test_label)

    empty_test_data = empty_data[empty_train_data_len:len(empty_data)]
    empty_test_label = [0 for i in range(len(empty_test_data))]

    np.save("D:\home\zeewei\projects\\77GRadar\classification_train_data\\pg_empty_val_data.numpy", empty_test_data)
    np.save("D:\home\zeewei\projects\\77GRadar\classification_train_data\\pg_empty_val_label.numpy", empty_test_label)

    for item in empty_test_data:
        test_data.append(item)
        test_label.append(0)

    np.save(PLAYGROUND_TRAIN_DATA_INPUT, train_data)
    np.save(PLAYGROUND_TRAIN_DATA_LABEL, train_label)
    np.save(PLAYGROUND_TEST_DATA_INPUT, test_data)
    np.save(PLAYGROUND_TEST_DATA_LABEL, test_label)
    return train_data, train_label, test_data, test_label
