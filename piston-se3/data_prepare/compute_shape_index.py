import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


def load_point_cloud(file_path):
    data = torch.load(file_path)
    pos = data['G'].ndata['pos']  # 点的位置信息
    return pos


def compute_curvature(pos, k=20):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pos)
    _, indices = nbrs.kneighbors(pos)

    curvature = []
    for idx in indices:
        neighbors = pos[idx]
        cov_matrix = np.cov(neighbors, rowvar=False)
        eigen_values, _ = np.linalg.eigh(cov_matrix)
        curvature.append(np.max(eigen_values))
    curvature = np.array(curvature)
    return curvature


def compute_shape_index(curvature):
    shape_index = (2 / np.pi) * np.arctan(curvature)
    return shape_index.flatten()


def process_file(file_path):
    # 确保输出目录存在
    output_dir = '../output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pos = load_point_cloud(file_path)

    curvature = compute_curvature(pos)
    shape_index = compute_shape_index(curvature).reshape(-1, 1)

    # 提取文件名作为链名
    base_name = os.path.basename(file_path)
    chain_name = os.path.splitext(base_name)[0]

    # 保存形状指数为.npy文件
    output_file_path = os.path.join(output_dir, f'{chain_name}_shape_index.npy')
    np.save(output_file_path, shape_index)
    print(f"Shape index saved as {output_file_path}")


def main():
    pt_folder_path = r'/home/wxy/Desktop/se/se3/data'
    list_file_path = r'/home/wxy/Desktop/se/se3/train.txt'  

    with open(list_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        chain_names = line.strip().split(',')  # 分割每行的链名
        for chain_name in chain_names:
            file_name = f'{chain_name}.pt'  # 构造文件名
            file_path = os.path.join(pt_folder_path, file_name)  # 构造完整的文件路径
            if os.path.exists(file_path):
                process_file(file_path)
            else:
                print(f"File does not exist: {file_path}")


if __name__ == "__main__":
    main()