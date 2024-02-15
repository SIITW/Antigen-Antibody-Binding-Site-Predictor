import numpy as np
import torch
from torch_geometric.data import Data
import os


def load_point_cloud(file_path):
    data = torch.load(file_path)
    pos = data['G'].ndata['pos']  # 点的位置信息
    return pos


def estimate_normals(points, k=20):
    normals = torch.empty(points.size(0), points.size(1))
    tree = torch.cdist(points, points)
    _, knn_idxs = torch.topk(tree, k, largest=False)
    for i, idxs in enumerate(knn_idxs):
        neighbors = points[idxs]
        mean = neighbors.mean(dim=0)
        neighbors = neighbors - mean
        cov_matrix = neighbors.T @ neighbors / (k - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        normals[i] = eigvecs[:, 0]
    return normals


def compute_ddc(points, normals):
    n_points = points.size(0)
    distances = torch.cdist(points, points)
    normal_diffs = torch.norm(normals.unsqueeze(1) - normals.unsqueeze(0), dim=2)
    ddc = normal_diffs / distances
    ddc[distances == 0] = float('inf')
    ddc[(ddc > 0.7) | (ddc < -0.7)] = float('inf')
    ddc_mean = torch.zeros(n_points)
    for i in range(n_points):
        valid_values = ddc[i][ddc[i] != float('inf')]
        ddc_mean[i] = valid_values.mean() if len(valid_values) > 0 else float('nan')
    return ddc_mean


def process_file(file_path):
    # 确保输出目录存在
    output_dir = '../output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points = load_point_cloud(file_path)
    normals = estimate_normals(points)
    ddc_values = compute_ddc(points, normals)
    ddc_values_np = ddc_values.numpy().reshape(-1, 1)

    base_name = os.path.basename(file_path)
    chain_name = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_dir, f'{chain_name}_ddc.npy')
    np.save(output_file_path, ddc_values_np)
    print(f'DDC values saved to {output_file_path} with shape {ddc_values_np.shape}')


def main(list_file_path, pt_folder_path):
    with open(list_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        chain_names = line.strip().split(',')
        for chain_name in chain_names:
            file_name = f'{chain_name}.pt'
            file_path = os.path.join(pt_folder_path, file_name)
            if os.path.exists(file_path):
                process_file(file_path)
            else:
                print(f"File does not exist: {file_path}")


if __name__ == "__main__":
    pt_folder_path = r'/home/wxy/Desktop/se/se3/data'
    list_file_path = r'/home/wxy/Desktop/se/se3/train.txt'
    main(list_file_path, pt_folder_path)
