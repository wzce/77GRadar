from configparser import ConfigParser


def load_config(config_name, config_item):
    # 初始化类
    cp = ConfigParser()
    cp.read("radar_data.cfg")
    print('cp.sections(): ', cp.sections())
    # 得到所有的section，以列表的形式返回
    sections = cp.sections()
    section = sections[0]
    # 得到该section中的option的值，返回为string类型
    return cp.get(section, config_item)


if __name__ == '__main__':
    dir = load_config('data_line1_windows', 'origin_data_dir')
    print('dir: ', dir)
