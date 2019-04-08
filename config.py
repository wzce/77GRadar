from configparser import ConfigParser
import os, sys
import platform


def load_config(index):
    sys_str = platform.system()
    if (sys_str == "Windows"):
        current_path = 'D:\home\zeewei\projects\\77GRadar'
    else:
        current_path = '/home/wzce/projects/77GRadar'

    config_path = os.path.join(current_path, "radar_data.cfg")
    print("abs path: ", current_path)
    # 初始化类
    cp = ConfigParser()
    cp.read(config_path)
    sections = cp.sections()
    section = sections[index]
    # 得到该section中的option的值，返回为string类型
    return cp, section


if __name__ == '__main__':
    dir = load_config(0)
    print('dir: ', dir)
