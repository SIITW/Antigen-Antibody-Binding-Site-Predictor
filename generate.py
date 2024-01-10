import os
from rdkit import Chem
import torch
from torch_geometric.data import Data
import numpy as np
import torch_geometric.transforms as T
import torch
import esm
import numpy as np
import dgl
from dgl.data.utils import save_graphs
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, radius, global_mean_pool, knn
# set seqnum
seqnum = 1

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

def EsmEmbedding(seqname,fasta):
    data = [
                (seqname,fasta.upper())
            ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.to(device)
    #get esm2 represention
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations[:,1:-1]
    return token_representations

def parse_pdb_file(pdb_file):
    atom_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_lines.append(line)
    return atom_lines

def get_pdb_features(pdb_file):
    atom_lines = parse_pdb_file(pdb_file)
    pdb_features = []

    for line in atom_lines:
        atom_serial = int(line[6:11])
        #residue_seq = int(line[22:26])
        #residue_name = line[17:20].strip()
        #atom_name = line[12:16].strip()
        atom_coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        residue_seq = line.split()[5]
        if len(line.split()[4]) > 1:
            residue_seq = line.split()[4][1:]
            atom_coords = [float(line.split()[5]), float(line.split()[6]), float(line.split()[7])]
        feature = (atom_serial, residue_seq, atom_coords)
        pdb_features.append(feature)

    return pdb_features

def get_atom_properties(file):
    if (Chem.MolFromPDBFile(file, removeHs=False, flavor=1, sanitize=False)):
        mol = Chem.MolFromPDBFile(file, removeHs=False, flavor=1, sanitize=False)
        ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B',
                     'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
        total_atom_feature = []
        for atom in mol.GetAtoms():
            atom_feature = onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) \
                           + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) \
                           + onek_encoding_unk(atom.GetFormalCharge(),[-1, -2, 1, 2, 0]) \
                           + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]) \
                           + [atom.GetIsAromatic()]
            atom_feature = list(map(int, atom_feature))
            total_atom_feature.append(atom_feature)
    return total_atom_feature

def generate_features(name, device):
    count = 1
    # 读取fasta文件
    fr = open('./'+str(name)+'.fasta', 'r')
    list = fr.readlines()
    for i in range(len(list)-1):
        data_list = []
        seq = list[i+1].split()[0]
        seq_name = list[i].split()[0]
        labellist = []
        # 大写表位 小写非表位
        for j in range(len(seq)):
            if seq[j].isupper() == True:
                labellist.append(1)
            else:
                labellist.append(0)
        seqlist = []
        coordlist = []
        atom_feature = []
        #print(seq_name)
        if seq_name[0] == '>':
            chain = seq_name[-1]
            seq_name = seq_name[1:-2]
            fasta = list[i + 1].split()[0]
            #可能存在 1a2y_A 1a2y_a
            if chain.isupper() ==True:
                file = 'bep3/'+str(name)+'/' + seq_name + '_'+ chain.upper() +'.pdb'
            else:
                file = 'bep3/'+str(name)+'/' + seq_name + '_' + chain + 'low.pdb'
            # todo change 读pdb特征
            features = get_pdb_features(file)
            for feature in features:
                atom_serial, seq_num, atom_coords = feature
                coord = [0, 0, 0]
                coord[0] = atom_coords[0]
                coord[1] = atom_coords[1]
                coord[2] = atom_coords[2]
                coordlist.append(coord.copy())
                seqlist.append(seq_num)

            # point_esm = EsmEmbedding(seq_name, fasta)
            # atom
            atom_feature = get_atom_properties(file)
            aa_y = []
            c = -1
            # 氨基酸序号问题
            # 检查pdb
            new_seq = []
            for s in range(len(seqlist)):
                before = seqlist[s-1]
                if seqlist[s] != before:
                    before = seqlist[s]
                    c+=1
                if seqlist[s] == before:
                    new_seq.append(c)


            x = torch.tensor(atom_feature, dtype=torch.float)  # 39
            y = torch.tensor(labellist)
            pos = torch.tensor(coordlist, dtype=torch.float)  # 3
            mask = torch.tensor(labellist)
            aa_y = labellist
            #fea_esm = point_esm
            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)
            #num = -1
            # starts with 0
            #flag = -100
            pool_batch = []
            pool_batch = new_seq
            #检查有没有序号错误
            '''if len(fea_esm[0]) != pool_batch[-1]+1:
                print('seq',len(fea_esm[0]),'stru',pool_batch[-1]+1,chain,seq_name)
                print(pool_batch)'''

            aa = torch.tensor(pool_batch)
            # print(aa)
            number = len(aa_y)
            aa_y = torch.tensor(aa_y)
            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask
            num_points = len(pos)
            batch = torch.zeros(num_points)
            # 出度入度 半径 太多会爆显存
            row, col = radius(pos, pos, 3, batch, batch, max_num_neighbors=32)
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
            torch.save(data, './fea/'+str(name)+'_' + str(count) + '.pt')

            count += 1

if __name__ == '__main__':
    from Bio.PDB import PDBParser
    from Bio import SeqIO
    #generate_features(343,device)
    #generate_features(24, device)
    #generate_features(15, device)
    generate_features('test', device)