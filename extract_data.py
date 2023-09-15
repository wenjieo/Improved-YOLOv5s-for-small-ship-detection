import glob
import os
import shutil
from tqdm import tqdm

def generate_gt(data_path):
    label_path = os.path.join(data_path, 'labels')
    if not os.path.exists(os.path.join(data_path, 'labels_conf')):
        os.mkdir(os.path.join(data_path, 'labels_conf'))
    label_txt_path = os.path.join(label_path, '*.txt')
    label_txt = glob.glob(label_txt_path)

    # 备份label文件
    for i in tqdm(label_txt, desc= '正在备份label文件...'):  #遍历每一个label的txt文件
        shutil.copy(i, os.path.join(data_path, 'labels_conf'))


    # 移动image和label
    for label_txt_per in tqdm(label_txt, desc= '正在移动images和labels文件'):
        with open(label_txt_per) as f:
            label_txt_per_lines = f.readlines()    #读取每一个label的txt文件的所有行
            obj_num = len(label_txt_per_lines)
            if obj_num == 1:
                with open(label_txt_per, 'w') as f:
                    for line in label_txt_per_lines:
                        line_split = line.strip().split()
                        # 标签重新写入
                        f.write(
                            line_split[0] + ' ' +
                            line_split[1] + ' ' +
                            line_split[2] + ' ' +
                            line_split[3] + ' ' +
                            line_split[4] + '\n')


                train_dir = os.path.join(data_path, 'train')
                train_images_dir = os.path.join(train_dir, 'images')
                train_labels_dir = os.path.join(train_dir, 'labels')

                suffix_list = ['.jpg', '.bmp', '.png']
                flag = False
                for suffix in suffix_list:
                    train_images_path = os.path.join(origin_img_path, os.path.splitext(os.path.basename(label_txt_per))[0] + suffix)
                    if os.path.exists(train_images_path):
                        flag = True
                        break
                train_labels_path = label_txt_per
                if not os.path.isdir(train_dir):
                    os.mkdir(train_dir)
                    if not os.path.isdir(train_images_dir):
                        os.mkdir(train_images_dir)
                    if not os.path.isdir(train_labels_dir):
                        os.mkdir(train_labels_dir)
                if flag:
                    shutil.copy(train_labels_path, train_labels_dir)
                    shutil.copy(train_images_path, train_images_dir)




            if obj_num > 1:
                print('图片：{}存在问题，有多次检测!!!'.format(os.path.splitext(label_txt_per)[0] + '.jpg'))



origin_img_path = r'C:\Users\HBX\Desktop\test'
data_path = r'C:\Users\HBX\Desktop\test'
generate_gt(data_path)