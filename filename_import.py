import glob
import io
import numpy as np
import cv2 as io
import imageio
#from scipy.misc import imsave
import os
imsave = imageio.imsave
def txt_create(name,txt_save):
    txt_save_dir = txt_save  # 新创建的txt文件的存放路径
    full_path = txt_save_dir +'/'+ name + '.txt'  #
    file = open(full_path, 'w')
    file.close()

def main(img_in,txt_save):
    filenames = list = os.listdir(img_in)  #读取所有txt文件
    for file in filenames:
        txt_create(file[:-4],txt_save)

main('E:\\yolov5-5.0-org\\VOC\\jpg','E:\\yolov5-5.0-org\\VOC\\labels')


