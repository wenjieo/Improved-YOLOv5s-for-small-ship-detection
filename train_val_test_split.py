'''
切分train、val、test
生成train.txt、val.txt、test.txt    'G:\Code\PyTorch-YOLOv3\my_data\JPEGImages\v10_1064.jpg'
Time:2021-7-1
'''

import os
import random
import shutil
from tqdm import tqdm

def mkdir_path(input_path):
    if not os.path.exists(input_path):
        os.mkdir(input_path)

def move_file(save_path, data_type):
    with open(os.path.join(save_path, '{}.txt'.format(data_type)), 'r') as f:
        file_list = f.readlines()
        for enum_file in tqdm(file_list):
            enum_file = enum_file.strip()
            if not enum_file:
                continue
            shutil.copy(enum_file,
                        os.path.join(images_file_path, data_type))
            shutil.copy(enum_file.replace("png", "txt").replace( "images", "labels",),
                        os.path.join(labels_file_path, data_type))


if __name__ == '__main__':
    save_path = r'E:\otherdata\Levir-Ship\our-division'
    labels_file_path = os.path.join(save_path, 'labels')
    images_file_path = os.path.join(save_path, 'images')
    labels_name_list = os.listdir(labels_file_path)
    images_name_list = os.listdir(images_file_path)
    num_total = len(labels_name_list)
    num_list = range(num_total)

    trainval_percent = 0.7
    train_percent = 0.9
    trainval_num = int(num_total * trainval_percent)
    train_num = int(trainval_num * train_percent)
    val_num = int(trainval_num - train_num)
    test_num = int(num_total - trainval_num)
    trainval_name = random.sample(num_list, trainval_num)
    train_name = random.sample(trainval_name, train_num)
    print('train and val size:', trainval_num)   # train and val 的个数
    print('train size:', train_num)   # train的个数
    print('val size:', val_num)  # train的个数
    print('test size:', test_num)  # train的个数

    # 写入trainval.txt\tarin.txt\val.txt\test.txt
    ftrainval = open(os.path.join(save_path, 'trainval.txt'), 'w')
    ftrain = open(os.path.join(save_path, 'train.txt'), 'w')
    fval = open(os.path.join(save_path, 'val.txt'), 'w')
    ftest = open(os.path.join(save_path, 'test.txt'), 'w')


    for i in range(num_total):
        # name = labels_name_list[i][:-4] + '\n'    #分离label文件名字
        images_name_list[i] = images_name_list[i] + '\n'
        _path = os.path.join(images_file_path, images_name_list[i])
        # print(_path)
        if i in trainval_name:
            ftrainval.write(_path)
            if i in train_name:
                ftrain.write(_path)
            else:
                fval.write(_path)
        else:
            ftest.write(_path)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    # 将图片放入train\val\test
    for i in ['train','val','test']:
        mkdir_path(os.path.join(labels_file_path, i))
        mkdir_path(os.path.join(images_file_path, i))
        move_file(save_path, i)





