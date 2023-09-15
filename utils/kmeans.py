import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)# zul: 把box移到原点，也就是width和height是相对于原点的值。详情可以见下文的解析。
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y # 计算交集。详情可以见下文的解析。
    box_area = box[0] * box[1] # box的面积
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
    # zul：这里boxes.shape[0]也就是box的数量。这句代码就是遍历所有box，求每个box与每个聚类中心的IOU的值，然后选出这些IOU值中最大的那个；最后求所有box的均值。



def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.（~~？未解，使用IOU度量标准去计算k-means聚类~~）
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function（~~？距离函数是代表什么~~）
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]#有多少个BBox，BBox的数量

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows（？Forgy方法是什么）
    clusters = boxes[np.random.choice(rows, k, replace=False)]# zul：初始化k个聚类中心（方法是从原始数据集中随机选k个）
	# np.random.choice(a=5, size=3, replace=False, p=None) 参数意思分别 是从a中以概率p，随机选择3个,p没有指定的时候相当于是一致的分布。replace代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。

        nearest_clusters = np.argmin(distances, axis=1)# 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。

        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break

        # 更新聚类中心（这里把每一个类的中位数作为新的聚类中心）
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
            # numpy模块下的median作用为：计算沿指定轴的中位数，返回数组元素的中位数。
			# 更详细使用方法见https://blog.csdn.net/chaipp0607/article/details/74347025

        last_clusters = nearest_clusters

    return clusters
