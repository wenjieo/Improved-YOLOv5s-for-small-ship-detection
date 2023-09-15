'''
anchor聚类算法:
    --kmeans
    --kmeans++
    --kmedian
    --kmedian++
Time: 2021-8-30
Author: huangbx
'''

import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import math
#解决中文显示
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['KaiTi']
# mpl.rcParams['font.serif'] = ['KaiTi']
from pandas import DataFrame

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']



# ----------------------------------------------------------------------------------------------------------------------
#读取txt标注数据

def parse_label(label_dir):
    label_txt = glob.glob(os.path.join(label_dir, '*.txt'))
    data_w = []
    data_h = []
    for label_txt_per in label_txt:
        with open(label_txt_per) as f:
            label_txt_per_lines = f.readlines()
            for line in label_txt_per_lines:
                line_split =line.strip().split()
                w = float(line_split[-2])
                h = float(line_split[-1])
                data_w.append(w)
                data_h.append(h)
                data_w_array = np.array(data_w)
                data_h_array = np.array(data_h)
                data_w_array = data_w_array[:, np.newaxis]
                data_h_array = data_h_array[:, np.newaxis]
                data_wh_array = np.concatenate((data_w_array, data_h_array), axis=1)
    return data_w_array, data_h_array, data_wh_array


def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# iou计算
def iou(clusters_i, targets_wh):
    '''
    :box:      np.array of shape (2,) containing w and h
    :clusters: np.array of shape (N cluster, 2)
    改 ---->> 加入DIOU、CIOU
    '''
    min_width = np.minimum(clusters_i[0], targets_wh[:, 0])
    min_hight = np.minimum(clusters_i[1], targets_wh[:, 1])

    intersection = min_width * min_hight
    box_area = targets_wh[:, 0] * targets_wh[:, 1]
    cluster_area = clusters_i[0] * clusters_i[1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def roll_select(samples_distance):
    """
        samples_distance:为样本当前的距离列表，get_distance的返回结果
        选择下一个聚类中心，以距离远近作为概率值，距离越远则被选择概率越大，距离越近则概率越小
    """
    # 依据距离计算每个样本点被选择作为下一个聚类中心点的概率
    p = samples_distance / np.sum(samples_distance)
    # 采用轮盘赌选择法选择下一个聚类中心
    cum_p = np.cumsum(p)
    select_index_lst = []
    # 为确保选择聚类中心比较靠谱，做多次轮盘赌选择出现次数比较多的样本索引
    for i in range(int(0.15 * len(samples_distance))):
        rand_num = np.random.random()
        select_index = list(rand_num > cum_p).index(False)
        select_index_lst.append(select_index)
    count_dict = {i: select_index_lst.count(i) for i in np.unique(select_index_lst)}
    select_index = sorted(count_dict, key=lambda x: count_dict[x])[-1]
    return select_index




# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
'''
四种聚类算法 kmeans  kmeans++ kmedian kmedian++
'''
# ----------------------------------------------------------------------------------------------------------------------
def gen_anchor(targers_w, targets_h, targets_wh, cluster_num, imgsize,
           kmeans=False, kmeans_add=False, kmedian=False, kmedian_add=True, seed=2):

    rows = len(targers_w)  # 取出数据个数
    distances = np.empty((rows, cluster_num))  # row x cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    if kmeans or kmedian:
        # initialize the cluster centers to be k items
        clusters_w = targers_w[np.random.choice(rows, cluster_num, replace=False)]
        clusters_h = targets_h[np.random.choice(rows, cluster_num, replace=False)]
        clusters = np.concatenate((clusters_w, clusters_h), axis=1)
        # print("K-means的初始化聚类中心为-->\n", clusters)

    if kmeans_add or kmedian_add:
        # 随机选出第一个聚类中心
        centers = [targets_wh[np.random.randint(targets_wh.shape[0])]]
        print("第一个初始化聚类中心为-->\n", centers)
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for i in range(cluster_num - 1):
            # 依据距离迭代选出后n-1个聚类中心
            centers = np.array(centers)
            # ds=[d_1, d_2, ...]
            ds = []
            for j in range(centers.shape[0]):
                ds.append(1 - iou(centers[j], targets_wh))
            ds = np.array(ds)
            data_distance = np.min(ds, axis=0)

            select_index = roll_select(data_distance)
            centers = centers.tolist()
            centers.append(targets_wh[select_index])
        centers_array = np.array(centers)
        clusters = centers_array
        print("K-means++的初始化聚类中心为-->\n", clusters)


    iterations = 0
    dist_mean = np.mean
    dist_median = np.median
    meanIOU = []
    while True:
        # Step 1: allocate each item to the closest cluster centers
        for i in range(cluster_num):  # I made change to lars76's code here to make the code faster
            # clusters_tensor = torch.from_numpy(clusters)
            # targets_wh_tensor = torch.from_numpy(targets_wh)
            distances[:, i] = 1 - iou(clusters[i], targets_wh)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():  # 每个类别簇的中心是否再变化
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(cluster_num):
            if len(targets_wh[nearest_clusters == cluster]) != 0:
                if kmeans or kmeans_add:
                    clusters[cluster] = dist_mean(targets_wh[nearest_clusters == cluster], axis=0)
                if kmedian or kmedian_add:
                    clusters[cluster] = dist_median(targets_wh[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters
        iterations += 1
        WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
        meanIOU.append(1 - WithinClusterMeanDist)
    meanIOU = np.array(meanIOU)
    clusters_org = np.zeros((clusters.shape[0], clusters.shape[1]))
    clusters_org[:, 0] = np.around(clusters[:, 0] * imgsize[0])
    clusters_org[:, 1] = np.around(clusters[:, 1] * imgsize[1])
    return meanIOU, clusters, clusters_org, iterations
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# 散点图可视化数据集所有wh
def visual_targets_wh(targets_w, targets_h):
    # ----------------------------------------------------------------
    figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(targets_w, targets_h, alpha=0.9, s=40, marker="o", c='#0000FF')
    ax.scatter(targets_w*640, targets_h*640, alpha=0.9, s=40, marker="o", c='#0000FF')
    ax.grid()
    # ax.legend()
    # plt.title("SAR船舰训练数据集GroundTruth可视化",fontsize=18)
    # plt.xlabel("数据集targets_w", fontsize=14)
    # plt.ylabel("数据集targets_h", fontsize=14)
    plt.xlabel("ship_w(pixels)", fontsize=20)
    plt.ylabel("ship_h(pixels)", fontsize=20)
    plt.show()

# 将数据集中舰船的宽、高数值保存到excel文件中
# def excel_targets_wh(targets_w, targets_h):
#     df = DataFrame({'W': list(targets_w), 'H': list(targets_h)})
#     df.to_excel('targets_hw.xlsx', sheet_name='sheet1', index=True)
# ----------------------------------------------------------------------------------------------------------------------

# 散点图可视化聚类中心分布
def visual_clusters(x, y):
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # ax.scatter(x, y, s=50, marker=".", color="r", label="聚类中心", edgecolors="red")
    ax.scatter(x*640, y*640, s=50, marker=".", color="r", label="聚类中心", edgecolors="red")
    # ax.legend(prop=my_legend_font)
    plt.xlabel("anchor_w(pixels)", fontsize=20)
    plt.ylabel("anchor_h(pixels)", fontsize=20)
    ax.grid()
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 可视化随迭代次数IOU变化
def visual_meanIOU(iterations, meanIOU):
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    x_iteration = np.arange(1, iterations + 1)
    ax.plot(x_iteration, meanIOU, linewidth=1.5)
    ax.scatter(x_iteration, meanIOU, s=15, marker="o", color="red", edgecolors="red")
    # ax.legend()
    plt.tick_params(labelsize=16)
    plt.xlabel("迭代次数", fontsize=18)
    plt.ylabel("平均交并比", fontsize=18)
    # ax.set_title("K-means平均交并比（MeanIOU）变化曲线",fontsize=18)
    ax.grid()
    plt.show()


label_dir = r'E:\otherdata\Levir-Ship\our-division\labels'
targets_w, targets_h, targets_wh = parse_label(label_dir)
meanIOU, clusters, clusters_org, iterations = gen_anchor(targets_w, targets_h, targets_wh, cluster_num=12,
                                                          kmeans=False, kmeans_add=True, kmedian=False, kmedian_add=False,
                                                           seed=2, imgsize=[640, 640])
# w = []
# h = []
# w = targets_w*640
# h = targets_h*640
print(targets_w)
print(targets_h)
print("每次迭代的meanIOU值为-->\n", meanIOU)
print("最终聚类中心为（标准化）-->\n", clusters)
print("-----------------------------------------------------------------------------")
print("最终聚类中心为（转换到原图）-->\n", clusters_org)
print("=========================聚类完成=========================\n共迭代{}次".format(iterations))

# 可视化anchor
visual_clusters(clusters[:, 0], clusters[:, 1])
# 可视化meanIOU
visual_meanIOU(iterations, meanIOU)
visual_targets_wh(targets_w, targets_h)

