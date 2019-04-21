class DataConfig:
    origin_train_data_dir = 'D:\\home\\zeewei\\20190319\\line1_train\\'
    origin_val_data_dir = 'D:\home\zeewei\\20190319\line1_val'
    process_data_dir = 'D:\\home\\zeewei\\projects\\77GRadar\\processed_data\\train_val_data_0409'
    train_data_file_name = 'one_line_train_0409.npy'
    val_data_file_name = 'one_line_val_0409.npy'

    train_result_log = 'train_one_line_log.npy'
    cnn_model_save_dir = 'D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\cnn2_1'
    rnn_model_save_dir = 'D:\home\zeewei\projects\\77GRadar\model\\rnn\model_save_dir\\rnn2_1_one_0409_2'

    train_data_input = 'pg_train_data_input.npy'
    train_data_label = 'pg_train_data_label.npy'
    test_data_input = 'pg_test_data_input.npy'
    test_data_label = 'pg_test_data_label.npy'

    train_parameter_file = "D:\home\zeewei\projects\\77GRadar\model\cnn\model_dir\\train_cnn2_1_0410_2.npy"

    def __init__(self):
        pass


class LinuxDataConfig(DataConfig):
    origin_train_data_dir = '/home/wzce/radar_data/data_line1/'
    process_data_dir = '/home/wzce/radar_data/processed_data/line1/'
    train_data_file_name = 'one_line_train_data_0408.npy'
    val_data_file_name = 'one_line_val_data_0408.npy'

    train_result_log = 'train_one_line_logs.npy'
    model_save_dir = '/home/wzce/radar_data/rnn/rnn2_1_0409/'

    def __init__(self):
        pass
