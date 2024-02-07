###
# Define a toy model: more complex models in experiments/qm9/models.py
###
#import dgl
import os

import numpy as np
import torch
import torch.nn as nn
import umap
from matplotlib import pyplot as plt
from torch.nn import Sequential as Seq, Dropout, GELU, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Softmax
#from dgl import load_graphs
from torch_geometric.data import DataLoader
from torch_geometric.graphgym import optim
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cross_attn import CrossAttention
from sklearn import metrics
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormSE3, GConvSE3, GMaxPooling
from kfold import divide_5fold_bep3
from zy_pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial.distance import cdist

def extract_atom_coords(residue):
    atom_coords = []
    for atom in residue.get_atoms():
        atom_coords.append(atom.get_coord())
    return atom_coords

def extract_amino_acids(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    amino_acids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip():  # Check if the residue has a name
                    amino_acids.append(residue)

    return amino_acids

def calculate_min_distance_between_amino_acids(pdb_file1, pdb_file2):
    amino_acids1 = extract_amino_acids(pdb_file1)
    amino_acids2 = extract_amino_acids(pdb_file2)

    num_amino_acids1 = len(amino_acids1)
    num_amino_acids2 = len(amino_acids2)

    min_distances_matrix = np.zeros((num_amino_acids1, num_amino_acids2))

    for i, amino_acid1 in enumerate(amino_acids1):
        coords1 = extract_atom_coords(amino_acid1)
        for j, amino_acid2 in enumerate(amino_acids2):
            coords2 = extract_atom_coords(amino_acid2)
            distances = cdist(coords1, coords2)
            min_distance = np.min(distances)
            min_distances_matrix[i, j] = min_distance

    return min_distances_matrix


torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# The maximum feature type is harmonic degree 3

num_degrees = 1
num_features = 39  # 修改为实际的特征维度
edge_dim = 78

fiber_in = Fiber(1, num_features)  # 使用新的 num_features
fiber_mid = Fiber(num_degrees, 32)  # 保持不变
fiber_out = Fiber(1, 128)  # 保持不变


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GSE3Res = GSE3Res(fiber_in, fiber_mid, edge_dim)
        self.GNormSE3 = GNormSE3(fiber_mid)
        self.GConvSE3 = GConvSE3(fiber_mid, fiber_out, edge_dim=edge_dim, self_interaction=True)
        self.crossattention = CrossAttention(dim=128)
        self.mlp_esm = Seq(Lin(128, 45), LN(45), GELU(), Lin(45, 1))
        self.mlp_for_esm = Seq(Lin(1280, 128))
        self.mlp_for_esm2 = Seq(Lin(1280, 2))
        self.soft_max = Softmax(dim=1)

    def bulid_se3(self):
        se3 = nn.ModuleList([self.GSE3Res,
                             self.GNormSE3,
                             self.GConvSE3
                             ])
        return se3

    def forward(self, data):
        pool_batch = data.aa
        if hasattr(data, 'edge_attr'):
            edge_features = data.edge_attr
            # 这里处理边的特征
        else:
            print("edge_attr not found in data object")
            # 可选的处理，如果边特征不存在

        # 然后是其余的模型处理

        G = data.G[0]
        basis, r = get_basis_and_r(G, num_degrees - 1)
        Se3Transformer = self.bulid_se3()
        features = {'0': G.ndata['f']}

        for layer in Se3Transformer:
            features = layer(features, G=G, r=r, basis=basis)

        # 在这里添加更多的打印语句来检查维度
        out = features['0'][..., -1].unsqueeze(0)

        out = global_mean_pool(out, pool_batch)

        # 在每个线性层之前和之后添加打印语句
        embedding = out
        embedding = embedding.squeeze(0)
        print(embedding.shape)
        return embedding

def embedding(model,anti):
    model.train()
    for data in anti:
        data = data.to(device)
        # 假设模型返回的是(out, embedding)，我们在这里打印out的形状
        embedding = model(data)
    # 返回最后一个out和embedding，注意这里我们返回的是一个元组
    return embedding


def train_one_epoch_with_clip(antigen_model, antibody_model, labels,device, optimizer, criterion):
    antigen_model.train()
    antibody_model.train()
    running_loss = 0.0
    antigen_data = torch.load('./dataset/fea/11_2.pt', map_location=device)
    antigen_loader = DataLoader([antigen_data], batch_size=1, shuffle=True, drop_last=True)
    # 加载抗体数据
    antibody_data = torch.load('./dataset/fea/11_1.pt', map_location=device)
    antibody_loader = DataLoader([antibody_data], batch_size=1, shuffle=True, drop_last=True)
    optimizer.zero_grad()
    # 训练模型
    antigen_features= embedding(antigen_model, antigen_loader)
    antibody_features = embedding(antibody_model, antibody_loader)
    antigen_features = torch.Tensor(antigen_features)
    antibody_features= torch.Tensor(antibody_features)

    print(antigen_features.shape)
    print(antibody_features.shape)
    antigen_features_normalized = antigen_features / torch.norm(antigen_features, dim=1, keepdim=True)
    print(antigen_features_normalized.shape)
    antibody_features_normalized = antibody_features / torch.norm(antibody_features, dim=1, keepdim=True)
    print(antibody_features_normalized.shape)
    antigen_features_normalized = antigen_features_normalized.transpose(0, 1)
    predict = torch.mm(antibody_features_normalized, antigen_features_normalized)
    print("predict",predict.size())
    labels = labels.to(device)
    loss = criterion(predict,labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    avg_loss = running_loss / len(labels)
    return avg_loss

def save_model(model, save_path, filename):
    """
    保存模型的状态字典。
    :param model: 要保存的模型。
    :param save_path: 模型保存路径。
    :param filename: 保存的文件名。
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    print(f"Model saved to {os.path.join(save_path, filename)}")


# 定义验证函数
def validate_model(antigen_model, antibody_model, val_loader, device, criterion):
    antigen_model.eval()
    antibody_model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            antigen_data, antibody_data, labels = batch
            antigen_data, antibody_data, labels = antigen_data.to(device), antibody_data.to(device), labels.to(device)
            antigen_features = antigen_model(antigen_data)
            antibody_features = antibody_model(antibody_data)
            # 计算损失...
            val_loss = criterion(antigen_features, antibody_features, labels)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss


# 示例使用
save_path = './model_saves'
num_epochs = 10

def main():
    # 两个PDB文件的路径
    pdb_file1 = r'/home/wxy/Desktop/新建文件夹 1/se3/dataset/bep3/11/3MJ9_A.pdb'
    pdb_file2 = r'/home/wxy/Desktop/新建文件夹 1/se3/dataset/bep3/11/3MJ9_H.pdb'

    min_distances_matrix = calculate_min_distance_between_amino_acids(pdb_file1, pdb_file2)
    min_distances_matrix = np.where(min_distances_matrix > 4.5, 0, 1)
    min_distances_matrix = torch.Tensor(min_distances_matrix)
    labels = min_distances_matrix
    print("labels",labels.size())
    # 创建两个模型实例
    antigen_model = Net().to(device)
    antibody_model = Net().to(device)
    # 优化器和损失函数
    optimizer = optim.Adam(list(antigen_model.parameters()) + list(antibody_model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练和验证
    save_path = './model_saves'
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch_with_clip(antigen_model, antibody_model,labels, device, optimizer, criterion)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss}")

        # 训练完成后保存模型
        save_model(antigen_model, save_path, 'final_antigen_model.pth')
        save_model(antibody_model, save_path, 'final_antibody_model.pth')
        print("Training completed and models saved.")

    print("Training completed.")

if __name__ == "__main__":
    main()