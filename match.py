import numpy as np
import torch
import cv2

def matching(antibody_vector, antigen_vector, matching_dim,k=1, distance_threshold=-1):
    # 计算两组向量之间的距离矩阵
    dist_matrix = torch.bmm(antigen_vector, antibody_vector)
    print(dist_matrix.shape)
    # KNN匹配 - 找到每个描述符的k个最近邻
    values, knn_idxs = torch.topk(dist_matrix, k, dim=matching_dim, largest=True)
    # 初始匹配列表
    preliminary_matches = []
    for i in range(knn_idxs.shape[0]):
        for j in range(knn_idxs.shape[1]):
            # 遍历每个向量的k个最近邻
            for m in range(k):
                if dist_matrix[i, j, knn_idxs[i, j, m]] > distance_threshold:
                    preliminary_matches.append(cv2.DMatch(j, knn_idxs[i, j, m].item(), dist_matrix[i, j, knn_idxs[i, j, m]].item()))

    # 应用互相最近的两个点策略
    mutual_matches = []
    for match in preliminary_matches:
        # 找到对方描述符的两个最近邻
        dists, min_idxs = torch.topk(dist_matrix[:, match.trainIdx], 1, largest=True)
        min_idxs_list = min_idxs.squeeze().tolist()  # 确保这是一个列表
        if isinstance(min_idxs_list, int):  # 如果min_idxs_list是一个整数，把它转换为一个列表
            min_idxs_list = [min_idxs_list]

        if match.queryIdx in min_idxs_list:
            mutual_matches.append(match)
        # 检查是否互为最近的两个点

    return mutual_matches
def intersection(match1, match2):
    # 假设match_results_dim1和match_results_dim2分别是两次匹配的结果列表
    matches_set_dim1 = {(m.queryIdx, m.trainIdx) for m in match1}
    matches_set_dim2 = {(m.queryIdx, m.trainIdx) for m in match2}

    # 找出交集
    intersection = matches_set_dim1.intersection(matches_set_dim2)
    return intersection

def match_points(matches):
    img_shape = (32, 32)
    coordinates = []  # 初始化用于存储所有坐标的列表
    for match in matches:
        # 获取查询图像的patch索引
        antibody_index = match.queryIdx  
        # 获取训练图像的patch索引
        antigen_index = match.trainIdx  

        # 计算查询图像的原始坐标
        antibody_x = antibody_index % img_shape[1]  # 列号
        antibody_y = antibody_index // img_shape[1]  # 行号

        # 计算训练图像的原始坐标
        antigen_x = antigen_index % img_shape[1]  # 列号
        antigen_y = antigen_index // img_shape[1]  # 行号
        # 添加坐标对到列表
        coordinates.append((antibody_x, antibody_y, antigen_x, antigen_y))

    return coordinates
def list(train_list,matches,npy_file_dir):
    antibody_list = [] 
    antigen_list = [] 
    for list in train_list:
        antibody=list.split(',')[0]
        antibody_list.append(antibody)
        antigen=list.split(',')[1]
        antigen_list.append(antigen)
    antibody_resnames=[]
    antigen_resnames=[]
    for antibody in antibody_list:
        antibody_resnames.append(antibody + "_resnames"+".npy")
    for antigen in antigen_list:
        antigen_resnames.append(antigen + "_resnames"+".npy")
    x=match_points(matches)
    for antibody,antigen,coords in zip(antibody_resnames,antigen_resnames,x):
        antibody_npy = np.load(npy_file_dir+antibody,allow_pickle=True)
        antigen_npy = np.load(npy_file_dir+antigen,allow_pickle=True)
        antibody_amino_acid = antibody_npy[coords[0], coords[1], 0]
        full_value = antibody_amino_acid
        parts = full_value.split(':')  # 将字符串按':'分割
        antibody_amino_acid = parts[2]  # 获取第三个部分
        antigen_amino_acid = antigen_npy[coords[2], coords[3], 0]
        full_value = antigen_amino_acid
        parts = full_value.split(':')  # 将字符串按':'分割
        antigen_amino_acid = parts[2]  # 获取第三个部分
        print("Antibody Amino Acid:", antibody_amino_acid, "Antigen Amino Acid:", antigen_amino_acid)
