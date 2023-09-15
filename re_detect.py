import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import shutil
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image
from PIL import ImageFilter

import re
from operator import length_hint

import json
import pandas as pd
import glob
cudnn.benchmark = True


def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_txt = True
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                opt.save_conf = True
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                continue
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                     cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')

def cut_select(image_path):
    img = cv2.imread(image_path, 0)
    file_name = os.path.basename(image_path)
    file_name = file_name.split('.')[0]
    height, width = img.shape
    m = int((height + 512) // 512)
    n = int((width + 512) // 512)
    des_img = np.zeros((int(m * 512), int(n * 512)), dtype='uint8')
    des_img[0:height, 0:width] = img  # des_img即为补充后的方图
    # 定义一个m*n的矩阵，用于存放第m行n列处图像块左上角坐标
    patch_x = np.zeros(shape=(m, n))
    patch_y = np.zeros(shape=(m, n))

    for i in range(m):
        for j in range(n):
            pics = des_img[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512]

            # pics = pics / 255.0
            # gamma = 1.2
            # pics = np.power(pics, gamma)
            # path = 'work' + '/' + 'cut_results' + '/' + file_name + '_' + str(i + 1) + '_' + str(j + 1) + '.png'
            path = 'cut_results' + '/' + file_name + '_' + str(i + 1) + '_' + str(j + 1) + '.png'
            cv2.imwrite(path, pics)
            # 计算第m行n列处图像块左上角坐标
            patch_x[i, j] = i * 512
            patch_y[i, j] = j * 512

if __name__ == '__main__':
    parser0 = argparse.ArgumentParser()
    # 数据集路径
    parser0.add_argument("--input_path", default=r'C:\Users\dell\Desktop\4', help="input path", type=str) #input_dir
    # 输出路径
    parser0.add_argument("--output_path", default='E:\yolov5-5.0-org\runs', help="output path", type=str) #outpou_dir
    args = parser0.parse_args()
    start_time = time.time()
    # 数据集输入路径
    image_list = glob.glob(args.input_path + '/*.png')
    '''
    if glob.glob(r"input_path/*/*.png") == []:
        image_list = glob.glob(r"input_path/*.png")
    else:
        image_list = glob.glob(r"input_path/*/*.png")
    '''

    # image_list = glob.glob(args.input_dir + '/*.png')
    # name = os.listdir(path0)
    # image1 = path0 + '/' + name[0]
    image = image_list[0]
    image_name = os.path.basename(image)
    cut_select(image)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'E:\yolov5-5.0-org\runs\train\exp-5010\weights\best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\dell\Desktop\4', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--weights', nargs='+', type=str, default='/work/best1.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='/work/cut_results', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='work/runs/detect/', help='save results to project/name')
    parser.add_argument('--project', default='runs/detect/', help='save results to project/name')
    parser.add_argument('--name', default='ship', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))


    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    # 清空cut_result文件夹并重新建立
    # shutil.rmtree('work/cut_results')
    # os.mkdir('work/cut_results')
    shutil.rmtree('cut_results')
    os.mkdir('cut_results')

    # path = 'work/runs/detect/ship/labels'
    path = 'runs/detect/ship/labels'
    file_name = os.path.basename(image)
    name = os.listdir(path)
    data_all = np.zeros(10)
    for k in range(length_hint(name)):
        file_name = os.path.basename(name[k])
        file_name = file_name.split('.')[0]
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file_name)
        m = int(res[1]) - 1
        n = int(res[2]) - 1
        data = np.genfromtxt(path + '/' + name[k])  # 将文件中数据加载到data数组里
        num = np.size(data)
        if num <= 6:  # 说明该TXT文件中只包含一条船只的信息
            x1 = data[1] - data[3] / 2
            y1 = data[2] - data[4] / 2
            x2 = data[1] + data[3] / 2
            y2 = data[2] - data[4] / 2
            x3 = data[1] + data[3] / 2
            y3 = data[2] + data[4] / 2
            x4 = data[1] - data[3] / 2
            y4 = data[2] + data[4] / 2
            w = 512
            h = 512
            Tdata = np.zeros(10)
            Tdata[0] = 0
            Tdata[1] = (n + x1) * w
            Tdata[2] = (m + y1) * h
            Tdata[3] = (n + x2) * w
            Tdata[4] = (m + y2) * h
            Tdata[5] = (n + x3) * w
            Tdata[6] = (m + y3) * h
            Tdata[7] = (n + x4) * w
            Tdata[8] = (m + y4) * h
            Tdata[9] = data[5]

        else:  # 该txt文件中包含多条船只的信息
            x1 = data[:, 1] - data[:, 3] / 2
            y1 = data[:, 2] - data[:, 4] / 2
            x2 = data[:, 1] + data[:, 3] / 2
            y2 = data[:, 2] - data[:, 4] / 2
            x3 = data[:, 1] + data[:, 3] / 2
            y3 = data[:, 2] + data[:, 4] / 2
            x4 = data[:, 1] - data[:, 3] / 2
            y4 = data[:, 2] + data[:, 4] / 2
            w = 512
            h = 512
            Tdata = np.zeros((data.shape[0], 10))
            Tdata[:, 0] = np.zeros(data.shape[0])
            Tdata[:, 1] = (n + x1) * w
            Tdata[:, 2] = (m + y1) * h
            Tdata[:, 3] = (n + x2) * w
            Tdata[:, 4] = (m + y2) * h
            Tdata[:, 5] = (n + x3) * w
            Tdata[:, 6] = (m + y3) * h
            Tdata[:, 7] = (n + x4) * w
            Tdata[:, 8] = (m + y4) * h
            Tdata[:, 9] = data[:, 5]

        data_all = np.vstack((data_all, Tdata))
    data_all = np.delete(data_all, 0, axis=0)
    data_all_1 = np.zeros(data_all.shape)

    for i in range(data_all_1.shape[0]):
        X1 = data_all[i, 1]
        X2 = data_all[i, 3]
        Y1 = data_all[i, 2]
        Y2 = data_all[i, 4]
        X4 = data_all[i, 5]
        Y4 = data_all[i, 6]
        k = (Y4-Y1)/(X2-X1)
        q = math.sqrt(5*k)
        p = 5/q

        data_all_1[i, 0] = data_all[i, 0]
        data_all_1[i, 1] = (X1+X2) / 2 - (q/2)*(X2-X1) # x1
        data_all_1[i, 2] = (Y1+Y4) / 2 - (p/2)*(Y4-Y1)  # y1
        data_all_1[i, 3] = (X1+X2) / 2 + (q/2)*(X2-X1)  # x2
        data_all_1[i, 4] = data_all_1[i, 2]  # y2
        data_all_1[i, 5] = data_all_1[i, 3]  # x3
        data_all_1[i, 6] = (Y1+Y4) / 2 + (p/2)*(Y4-Y1)  # y3
        data_all_1[i, 7] = data_all_1[i, 1]  # x4
        data_all_1[i, 8] = data_all_1[i, 6]  # y4
        data_all_1[i, 9] = data_all[i, 9]

    np.savetxt(r'input_path/mask/mask.txt', data_all_1, fmt = '%d,%f,%f,%f,%f,%f,%f,%f,%f,%f', delimiter=',')

    subdf = pd.read_csv('input_path/mask/mask.txt', header=None)  # 获取数据，路径与上面保存txt的路径保持一致
    ##########################将得到的txt文件转成比赛要求的json文件并保存到输出文件夹中##################################
    # 创建空的待写入列表
    content_json = []
    # 创建列表内容
    jsontext = {
        'image_name': image_name,
        'labels': []  # 初始化标签内容
    }
    # 添加标签信息
    for row, index in subdf.iterrows():
        jsontext['labels'].append(
            {
                "category_id": "bigship",
                "points": [[index[1], index[2]], [index[7], index[8]], [index[5], index[6]], [index[3], index[4]]],
                "confidence": index[9]
            }
        )
    # 将列表内容添加到列表里
    content_json.append(jsontext)
    # 生成json数据
    jsondata = json.dumps(content_json, indent=4, separators=(',', ': '))

    # 将 json 数据写入文件
    with open(args.output_dir + '/ship_results.json', 'w') as f:
        json.dump(content_json, f, indent=4, ensure_ascii=False)

    # shutil.rmtree('work/runs/detect')
    # os.mkdir('work/runs/detect')
    shutil.rmtree('runs/detect')
    os.mkdir('runs/detect')
    print('total time:', time.time() - start_time)