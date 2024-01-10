import numpy as np
import torch
import cv2


def Image_matching(desc1, desc2, k=2, distance_threshold=0.9):

    antigen_vector = desc1.flatten()
    antibody_vector = desc2.flatten()
    # 计算两组描述符之间的点乘
    dist_matrix = torch.ger(antibody_vector, antigen_vector)
    # KNN匹配 - 找到每个描述符的k个最近邻
    dists, knn_idxs = torch.topk(dist_matrix, k, dim=1, largest=False)

    # 初始匹配列表
    preliminary_matches = []
    for i in range(knn_idxs.shape[0]):
        for j in range(knn_idxs.shape[1]):
            if dist_matrix[i, knn_idxs[i, j]] > distance_threshold:  # 检查点乘是否大于阈值
                preliminary_matches.append(cv2.DMatch(i, knn_idxs[i, j].item(), dist_matrix[i, knn_idxs[i, j]].item()))

    # 应用互相最近的两个点策略
    mutual_matches = []
    for match in preliminary_matches:
        # 找到对方描述符的两个最近邻
        dists, min_idxs = torch.topk(dist_matrix[:, match.trainIdx], 1, largest=False)

        # 检查是否互为最近的两个点
        if match.queryIdx in min_idxs:
            mutual_matches.append(match)

    return mutual_matches
