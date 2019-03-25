# coding:utf8
import os


def rename(path):
    i = 0
    file_list = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for file_name in file_list:  # 遍历所有文件
        print('file: ', file_name)
        old_file = os.path.join(path,file_name)
        new_file = os.path.join(path,'2019_03_24_'+file_name)
        # i = i + 1
        # Olddir = os.path.join(path, files);  # 原来的文件路径
        # if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        #     continue;
        # filename = os.path.splitext(files)[0];  # 文件名
        # filetype = os.path.splitext(files)[1];  # 文件扩展名
        # Newdir = os.path.join(path, str(i) + filetype);  # 新的文件路径
        os.rename(old_file, new_file)  # 重命名


if __name__ == '__main__':
    path = 'D:\home\\20190324'
    rename(path)
