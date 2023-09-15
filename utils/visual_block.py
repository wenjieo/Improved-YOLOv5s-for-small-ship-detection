'''
负责进行可视化模块
'''
import numpy as np
import cv2
import torch
import os
import math

import matplotlib.pyplot as plt
from torchvision import transforms




# 特征图可视化
# ----------------------------------------------------------------------------------------------------------------------
def feature_visualization(features, model_type, model_id, feature_num=64):     #与general.py中的可视化函数一致
    """
    features: The feature map which you need to visualization
    model_type: The type of feature map
    model_id: The id of feature map
    feature_num: The amount of visualization you need
    """
    save_dir = "./features/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print(features.shape)
    # block by channel dimension
    blocks = torch.chunk(features, features.shape[1], dim=1)

    # # size of feature
    # size = features.shape[2], features.shape[3]

    plt.figure()
    for i in range(feature_num):
        torch.squeeze(blocks[i])
        feature = transforms.ToPILImage()(blocks[i].squeeze())
        # print(feature)
        ax = plt.subplot(int(math.sqrt(feature_num)), int(math.sqrt(feature_num)), i + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature)
        # gray feature
        # plt.imshow(feature, cmap='gray')

    # plt.show()
    plt.savefig(save_dir + '{}_{}_feature_map_{}.png'
                .format(model_type.split('.')[2], model_id, feature_num), dpi=300)

# ----------------------------------------------------------------------------------------------------------------------
# 可视化targets和正样本anchor代码
def merge_imgs(imgs, row_col_num):
    """
        Merges all input images as an image with specified merge format.
        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    color = random_color(rgb=True).astype(np.float64)

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        merge_imgs = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        merge_imgs = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        merge_imgs = np.vstack(merge_imgs_col)

    return merge_imgs

# ----------------------------------------------------------------------------------------------------------------------
def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
    """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.
        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    if is_merge:
        merge_imgs1 = merge_imgs(imgs, row_col_num)

        cv2.namedWindow('merge', 0)
        cv2.moveWindow('merge', 1000, 100)
        cv2.imshow('merge', merge_imgs1)
    else:
        for img, win_name in zip(imgs, window_names):
            if img is None:
                continue
            win_name = str(win_name)
            cv2.namedWindow(win_name, 0)
            cv2.resizeWindow(win_name, 1000, 1000)
            cv2.moveWindow(win_name, 1000, 100)
            cv2.imshow(win_name, img)

    cv2.waitKey(wait_time_ms)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

# ----------------------------------------------------------------------------------------------------------------------
def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

# ----------------------------------------------------------------------------------------------------------------------
def show_bbox(image, targets_list, color, center,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.
    :param image: numpy.ndarray.
    :param targets_list: list: [pts_xyxy, prob, id]: label or prediction.
    :param color: tuple.
    :param thickness: int.
    :param fontScale: float.
    :param wait_time_ms: int
    :param names: string: window name
    :param is_show: bool: whether to display during middle process
    :return: numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    for bbox in targets_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            import random
            colors = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # colors = (255, 0 ,0)
        else:
            colors = color

        if not is_without_mask:
            # --------------------------------------------------------------------------------------------
            # 在图片上绘制gt bbox
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
            # --------------------------------------------------------------------------------------------



            # --------------------------------------------------------------------------------------------
            # 在图片上根据gt bbox坐标绘制中心点
            if center:
                circle_x = int((bbox_f[2] + bbox_f[0]) / 2)
                circle_y = int((bbox_f[3] + bbox_f[1]) / 2)
                point_size = 2
                point_color = (0, 0, 255)
                image_copy = cv2.circle(image_copy, (circle_x, circle_y), point_size, point_color, -1)
            # --------------------------------------------------------------------------------------------
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:  # 显示绘制后的图像
        show_img(image_copy, names, wait_time_ms)
    return image_copy

# ----------------------------------------------------------------------------------------------------------------------
def xywhToxyxy(bbox):
    y = np.zeros_like(bbox)
    y[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
    y[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
    y[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
    y[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
    return y
# ----------------------------------------------------------------------------------------------------------------------

def vis_bbox(imgs, targets, color, center):
    targets_array = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[1]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    targets_array = (targets_array * gain)
    for i in range(imgs.shape[0]):
        img = data[i].astype(np.uint8)
        img = img[..., ::-1]
        targets_array_bach = targets_array[targets_array[:, 0] == i][:, 2:]
        targets_xyxy = xywhToxyxy(targets_array_bach)
        show_bbox(img, targets_xyxy, color, center)
# ----------------------------------------------------------------------------------------------------------------------

def vis_match(imgs, targets, tcls, tboxs, indices, anchors, pred, ttars, color, center):
    tar = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[2]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    tar = (tar * gain)

    strdie = [8, 16, 32]
    # 对batch的每张图片进行可视化
    for j in range(imgs.shape[0]):
        img = data[j].astype(np.uint8)[..., ::-1]
        tar1 = tar[tar[:, 0] == j][:, 2:]
        y1 = xywhToxyxy(tar1)
        # img = VisualHelper.show_bbox(img1.copy(), y1, color=(255, 255, 255), is_show=False, thickness=2)
        # 对每个预测尺度进行单独可视化
        vis_imgs = []
        for i in range(3):  # i=0检测小物体，i=1检测中等尺度物体，i=2检测大物体
            s = strdie[i]
            # anchor尺度
            gain1 = np.array(pred[i].shape)[[3, 2, 3, 2]]
            b, a, gx, gy = indices[i]
            b1 = b.cpu().detach().numpy()
            gx1 = gx.cpu().detach().numpy()
            gy1 = gy.cpu().detach().numpy()
            anchor = anchors[i].cpu().detach().numpy()
            ttar = ttars[i].cpu().detach().numpy()

            # 找出对应图片对应分支的信息
            indx = b1 == j
            gx1 = gx1[indx]
            gy1 = gy1[indx]
            anchor = anchor[indx]
            ttar = ttar[indx]

            # 还原到原图尺度进行可视化
            ttar /= gain1  # 归一化
            ttar *= np.array([w, h, w, h], np.float32)  # 还原到原图
            y = xywhToxyxy(ttar)
            # targets可视化
            img1 = show_bbox(img.copy(), y, color=(0, 0, 255), thickness=2, is_show=False, center=False)

            # anchor 需要考虑偏移，在任何一层，每个gt bbox最多3*3=9个anchor进行匹配   第一个3：改层的3个anchor； 第二个3：相邻2个网格+本网格
            anchor *= s
            anchor_bbox = np.stack([gy1, gx1], axis=1)
            k = np.array(pred[i].shape, np.float)[[3, 2]]
            anchor_bbox = anchor_bbox / k
            anchor_bbox *= np.array([w, h], np.float32)
            anchor_bbox = np.concatenate([anchor_bbox, anchor], axis=1)
            anchor_bbox1 = xywhToxyxy(anchor_bbox)
            # 正样本anchor可视化
            img1 = show_bbox(img1, anchor_bbox1, color=color, is_show=False, center=center)
            vis_imgs.append(img1)
        show_img(vis_imgs, is_merge=True)
