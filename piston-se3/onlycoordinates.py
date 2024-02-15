import os
import torch
from torch_geometric.data import Data
import numpy as np
import torch_geometric.transforms as T
import torch
import numpy as np
import dgl
from dgl.data.utils import save_graphs
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, radius, global_mean_pool, knn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()
torch.manual_seed(1)

def calculate_edge_distances(edge_index, pos):
    row, col = edge_index

    # Get the coordinates of the nodes for each edge
    edge_src = pos[col]
    edge_dst = pos[row]

    # Calculate the Euclidean distance between the nodes of each edge
    edge_diff = edge_dst - edge_src
    return edge_diff


def generate_features_from_txt(txt_file, device):
    with open(txt_file, 'r') as f:
        pair_list = [line.strip() for line in f.readlines()]

    for pair in pair_list:
        chain_names = pair.split(',')
        name = chain_names[0][:4]  # Extract PPI from the first chain name
        for chain in chain_names:
            file_list = [f'./output/{chain}_{feature}.npy' for feature in
                         ['coordinates']]
            data_list = [np.load(file) for file in file_list]

            # 使用np.concatenate函数将列表中的数组沿着指定的轴进行拼接
            concatenated_array = np.concatenate(data_list, axis=1)

            labellist = np.ones(len(concatenated_array))
            pos = concatenated_array[:, 0:3]
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(concatenated_array[:, 3:], dtype=torch.float)
            y = torch.tensor(labellist)

            data = Data(x=x, y=y)
            num_points = len(pos)
            batch = torch.zeros(num_points)

            # 出度入度 半径 太多会爆显存
            row, col = radius(pos, pos, 2, batch, batch, max_num_neighbors=16)
            edge_index = torch.stack([col, row], dim=0)

            # 创建点云数据
            points = pos  # 点的坐标矩阵，形状为 (num_points, 3)
            features = x.unsqueeze(2)  # 点的特征矩阵，形状为 (num_points, feature_dim)
            graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_points)

            # 设置节点特征
            graph.ndata['pos'] = torch.tensor(points, dtype=torch.float32)
            graph.ndata['f'] = torch.tensor(features, dtype=torch.float32)

            # Add node index as feature
            node_indices = torch.arange(num_points).unsqueeze(1).float()
            graph.ndata['idx'] = node_indices

            # 边特征
            target_features = graph.ndata['f']
            target_features = torch.tensor(target_features, dtype=torch.float32)
            edge_feature1 = target_features[edge_index[1]]
            edge_feature2 = target_features[edge_index[0]]
            edge_features = torch.cat([edge_feature1, edge_feature2], dim=1)
            edge_features = torch.transpose(edge_features, 1, 2)
            graph.edata['w'] = edge_features

            dist = calculate_edge_distances(edge_index, pos)
            dist = dist.unsqueeze(1)
            graph.edata['d'] = torch.tensor(dist, dtype=torch.float32)

            data.G = graph

            output_filename = f"./data/{name}_{chain[-1]}.pt"
            torch.save(data, output_filename)

if __name__ == '__main__':
    txt_file = r'/home/wxy/Desktop/se/se3/train.txt'  # 输入txt文件的路径
    generate_features_from_txt(txt_file, device)
