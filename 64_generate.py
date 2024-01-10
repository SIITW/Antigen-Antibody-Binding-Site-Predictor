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
# set seqnum
seqnum = 1

# Load ESM-2 model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()
torch.manual_seed(1)

#
def read_pssm_file(file_path):
    pssm_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            try:
                if line[0] >= '0' and line[0] <= '9' and len(line) >= 100:
                    pssm_values = line.split()[2:22]
                    pssm_list.append([int(value) for value in pssm_values])
            except:
                continue
    return pssm_list

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def calculate_edge_distances(edge_index, pos):
    row, col = edge_index

    # Get the coordinates of the nodes for each edge
    edge_src = pos[col]
    edge_dst = pos[row]

    # Calculate the Euclidean distance between the nodes of each edge
    # edge_distances = torch.norm(edge_dst - edge_src, dim=1)
    edge_diff = edge_dst - edge_src
    # Calculate the Euclidean distance between the nodes of each edge
    #edge_distances = torch.norm(edge_diff, dim=1)
    return edge_diff


def generate_features(device):

    count = 1
    namelist = []
    chain_listA = []
    chain_listB = []
    fr = open('./example/final_training.txt')
    list = fr.readlines()

    for i in range(len(list)):
        name = list[i].split(',')[0][0:4]
        namelist.append(name)
        chain_listA.append(list[i].split(',')[0][-1])
        chain_listB.append(list[i].split(',')[1][-2])
    for i in range(len(list)):
        name = namelist[i]
        chain_list = [chain_listA[i],chain_listB[i]]
        for chain in chain_list:
            # 定义要加载和拼接的npy文件列表
            file_list = ['./example/training/grid_16R/'+name+'_'+chain+'.npy']
            # file_list = ['./example/'+name+'/intermediate_files/05-patches_16R/'+name+'/'+name+'_'+chain+'_coordinates.npy',
            #              './example/'+name+'/intermediate_files/05-patches_16R/'+name+'/'+name+'_'+chain+'_input_feat.npy',
            #              './example/'+name+'/intermediate_files/05-patches_16R/'+name+'/'+name+'_'+chain+'_dssp_features.npy']
            data_list = [np.load(file) for file in file_list]
            # 使用np.concatenate函数将列表中的数组沿着指定的轴进行拼接
            concatenated_array = np.concatenate(data_list, axis=1)
            #打印拼接后的数组形状
            #print(concatenated_array.shape)
            #
            labellist = np.ones(len(concatenated_array))
            pos = concatenated_array[:, :, 7:].reshape(64*64,3)
            pos = torch.tensor(pos, dtype=torch.float)
            x = concatenated_array[:, :, :7].reshape(64*64,7)
            x = torch.tensor(x, dtype=torch.float)  # 6
            y = torch.tensor(labellist)
            data = Data(x=x, y=y)
            num_points = len(pos)
            batch = torch.zeros(num_points)
            # 出度入度 半径 太多会爆显存
            row, col = radius(pos, pos, 2, batch, batch, max_num_neighbors=8)
            #
            edge_index = torch.stack([col, row], dim=0)
            # 创建点云数据
            points = pos  # 点的坐标矩阵，形状为 (num_points, 3)
            features = x.unsqueeze(2)  # 点的特征矩阵，形状为 (num_points, feature_dim)
            # 创建图
            graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_points)
            # 设置节点特征
            # 点坐标
            graph.ndata['pos'] = torch.tensor(points, dtype=torch.float32)
            # 点特征
            graph.ndata['f'] = torch.tensor(features, dtype=torch.float32)
            #graph.ndata['y'] = torch.tensor(labellist, dtype=torch.int)
            len_edge_index = len(edge_index[0])
            ones_tensor = torch.ones(len_edge_index)
            ones_tensor = ones_tensor.unsqueeze(1).unsqueeze(1)
            # 边特征
            target_features = graph.ndata['f']
            target_features = torch.tensor(target_features, dtype=torch.float32)
            # 使用目标节点的特征作为边的特征
            edge_feature1 = target_features[edge_index[1]]
            edge_feature2 = target_features[edge_index[0]]
            edge_features = torch.cat([edge_feature1, edge_feature2], dim=1)
            edge_features = torch.transpose(edge_features, 1, 2)
            graph.edata['w'] = edge_features

            dist = calculate_edge_distances(edge_index, pos)
            dist = dist.unsqueeze(1)
            # 坐标相减
            graph.edata['d'] = torch.tensor(dist, dtype=torch.float32)

            data.G = graph

            data = data.to(device)
            torch.save(data, './se3/dataset/fea/'+str(name)+'_' + str(chain) + '.pt')
            count += 1

if __name__ == '__main__':
    generate_features(device)