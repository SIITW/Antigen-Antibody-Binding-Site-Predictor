###
# Define a toy model: more complex models in experiments/qm9/models.py
###
import dgl
import numpy as np
import torch
import torch.nn as nn
import umap
from matplotlib import pyplot as plt
from torch.nn import Sequential as Seq, Dropout, GELU, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Softmax
from dgl import load_graphs
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cross_attn import CrossAttention
from sklearn import metrics
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormSE3, GConvSE3, GMaxPooling
from kfold import divide_5fold_bep3
from zy_pytorchtools import EarlyStopping
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, radius, global_mean_pool, knn
from sklearn.metrics import confusion_matrix, precision_score, recall_score

torch.manual_seed(1)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# The maximum feature type is harmonic degree 3

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, p_weight):
        #pos_weight = torch.FloatTensor([p_weight]).to(device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=p_weight, reduction='none')
        #BCE_loss = F.binary_cross_entropy(inputs, targets)
        #BCE_loss = BCE_loss.item()
        # pt = torch.exp(-BCE_loss)
        # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        # return F_loss.mean()
        return BCE_loss.mean()

def drawpic(reducer, embedding_list, label_list, fold, count):
    embedding_list = reducer.fit_transform(embedding_list)  # 对数据进行降维
    cmap = plt.cm.colors.ListedColormap(['blue', 'red'])
    plt.scatter(embedding_list[:, 0], embedding_list[:, 1], s=1, c=label_list, cmap=cmap)  # 绘制散点图
    red_indices = np.where(label_list == 1)[0]  # 假设红色点对应的标签为1
    plt.scatter(embedding_list[red_indices, 0], embedding_list[red_indices, 1], s=5, c='red')
    #todo change dictionary
    plt.savefig('./pic/se3only2_'+str(fold)+'_' + str(count) + '.jpg')  # 显示图形
    plt.clf()



def calculate_edge_distances(edge_index, pos):
    row, col = edge_index
    # Get the coordinates of the nodes for each edge
    edge_src = pos[col]
    edge_dst = pos[row]

    # Calculate the Euclidean distance between the nodes of each edge
    # edge_distances = torch.norm(edge_dst - edge_src, dim=1)
    edge_diff = edge_dst - edge_src
    # Calculate the Euclidean distance between the nodes of each edge
    # edge_distances = torch.norm(edge_diff, dim=1)
    return edge_diff

def load_data(data,r):
    num_points = len(data.pos)
    batch = torch.zeros(num_points)
    row, col = radius(data.pos,data.pos, r, batch, batch, max_num_neighbors=64)
    edge_index = torch.stack([col, row], dim=0)
    # 创建点云数据
    points = data.pos  # 点的坐标矩阵，形状为 (num_points, 3)
    features = data.x.unsqueeze(2)  # 点的特征矩阵，形状为 (num_points, feature_dim)
    # 创建图
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_points)
    graph = graph.to(device)
    # 设置节点特征
    graph.ndata['pos'] = torch.tensor(points, dtype=torch.float32)
    graph.ndata['f'] = torch.tensor(features, dtype=torch.float32)
    # graph.ndata['y'] = torch.tensor(labellist, dtype=torch.int)
    len_edge_index = len(edge_index[0])
    target_features = graph.ndata['f']
    target_features = torch.tensor(target_features, dtype=torch.float32)
    # 使用目标节点的特征作为边的特征
    edge_feature1 = target_features[edge_index[1]]
    edge_feature2 = target_features[edge_index[0]]
    edge_features = torch.cat([edge_feature1,edge_feature2],dim=1)
    edge_features = torch.transpose(edge_features, 1, 2)
    graph.edata['w'] = edge_features
    #graph.edata['w'] = torch.tensor(ones_tensor, dtype=torch.float32)
    dist = calculate_edge_distances(edge_index, data.pos)
    dist = dist.unsqueeze(1)
    graph.edata['d'] = torch.tensor(dist, dtype=torch.float32)
    data.G = graph
    return data



class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GSE3Res = GSE3Res(fiber_in, fiber_mid,edge_dim)
        self.GSE3Res2 = GSE3Res(fiber_in2, fiber_mid, edge_dim)
        self.GNormSE3 = GNormSE3(fiber_mid)
        self.GConvSE3 = GConvSE3(fiber_mid, fiber_out,edge_dim=edge_dim,self_interaction=True)
        self.GConvSE32 = GConvSE3(fiber_mid, fiber_out, edge_dim=edge_dim, self_interaction=True)
        self.crossattention = CrossAttention(dim=1280)
        self.mlp_esm = Seq(Lin(1280, 45), LN(45), GELU(), Lin(45, 1))
        self.mlp_for_se3 = Seq(Lin(128, 1))
        self.mlp_for_esm2 = Seq(Lin(1280, 2))
        self.soft_max = Softmax(dim=1)

    def bulid_se3(self):
        se3 = nn.ModuleList([self.GSE3Res,
                    self.GNormSE3,
                    self.GConvSE3,
                       ])
        return se3
    def forward(self, data):
        pool_batch, esm_list = data.aa, data.esm_list
        G = data.G[0]
        basis, r = get_basis_and_r(G, num_degrees - 1)
        Se3Transformer = self.bulid_se3()
        features = {'0': G.ndata['f']}
        for layer in Se3Transformer:
            features = layer(features, G=G, r=r, basis=basis)
        out = features['0'][..., -1].unsqueeze(0)
        out = global_mean_pool(out, pool_batch)
        embedding = out
        embedding = embedding.squeeze(0)
        out = self.mlp_for_se3(out)
        '''out_seq = self.crossattention(out, esm_list)
        out_atom = self.crossattention(esm_list, out)
        out_seq, out_atom = out_seq[0], out_atom[0]
        out = out_seq + out_atom'''
        out = out.squeeze(0)
        data.label = data.aa_y
        return out,embedding

def train_model(model, patience, n_epochs, checkpoint):
    train_losses = []
    valid_losses = []
    label_total = []
    score_total = []
    avg_train_losses = []
    avg_valid_losses = []

    reducer = umap.UMAP(n_components=2)
    early_stopping = EarlyStopping(patience=patience, path=checkpoint, verbose=True)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            out,embedding = model(data)
            # label = data.aa_y.float()
            label = data.label.float()
            label = label.unsqueeze(1)
            loss = focalloss(out, label, train_p_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for data in bep3testloader:
                data = data.to(device)
                out,embedding = model(data)
                score = torch.sigmoid(out)
                score_total.extend(score.detach().cpu().numpy())
                label = data.label.float()
                label = label.unsqueeze(1)
                label_total.extend(label.detach().cpu().numpy())
                loss = focalloss(out, label, val_p_weight)
                valid_losses.append(loss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        auc = metrics.roc_auc_score(label_total, score_total)
        ap = metrics.average_precision_score(label_total, score_total)

        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'AUC: {auc:.5f}' +
                     f'AP: {ap:.5f}')
        print(print_msg)
        train_losses = []
        valid_losses = []
        label_total = []
        score_total = []

        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model = torch.load(checkpoint)

    return avg_train_losses, avg_valid_losses

def model_test(fold):
    pred_total = []
    aa_total = []
    out_total = []
    model_list = []
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_list[fold]))
    model.eval()
    model_list.append(model)
    with torch.no_grad():
        count = 1
        reducer = umap.UMAP(n_components=2)
        for data in testloader:
            data = data.to(device)
            out,embedding = model(data)
            out = torch.sigmoid(out)
            out_total.extend(out.detach().cpu().numpy())
            pred = out.ge(0.5).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())
            embedding2 = embedding.detach().cpu().numpy()
            label2 = data.label.detach().cpu().numpy()
            drawpic(reducer, embedding2, label2, fold, count)
            count += 1
    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()
    aa_total = torch.tensor(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    print('correct: ',correct)
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    print('tn' + str(tn) + 'tp' + str(tp) + 'fn' + str(fn) + 'fp' + str(fp))
    recall = metrics.recall_score(aa_total, pred_total)
    print('recall:' + str(recall))
    precision = metrics.precision_score(aa_total, pred_total)
    print('precision:' + str(precision))
    mcc = metrics.matthews_corrcoef(aa_total, pred_total)
    print('mcc:' + str(mcc))
    auc = metrics.roc_auc_score(aa_total, out_total)
    print('AUC:' + str(auc))
    ap = metrics.average_precision_score(aa_total, out_total)
    print('AP:' + str(ap))
    f1 = metrics.f1_score(aa_total, pred_total)
    print('f1:' + str(f1))
    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open('result/se3only_(' + str(fold) + ').txt', 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')


num_degrees = 1
num_features = 39
edge_dim = 78

fiber_in = Fiber(1, num_features)
fiber_in2 = Fiber(1, 128)
fiber_mid = Fiber(num_degrees, 32)
fiber_out = Fiber(1, 128)

test = []
bp3test = []
for i in range(1, 25):
    #data = torch.load('Bep3_dataset/test_solved/feature/withoutpssm/' + str(i) + '.pt',map_location=device)
    data = torch.load('dataset/fea/24_' + str(i) + '.pt', map_location=device)
    data = load_data(data=data,r=4.5)
    test.append(data)
for i in range(1,16):
    data = torch.load('dataset/fea/15_' + str(i) + '.pt', map_location=device)
    data = load_data(data=data, r=4.5)
    bp3test.append(data)

testloader = DataLoader(test, batch_size=1)
bep3testloader = DataLoader(bp3test, batch_size=1)
# InDdataset = torch.load('./272_dataset.pt')
checkpoint_list = []
train_set, val_set = divide_5fold_bep3(343,5)
for fold in [0, 1, 2, 3, 4]:
    train = []
    val = []
    for i in range(len(train_set[fold])):
        data = torch.load('./dataset/fea/343_' + str(train_set[fold][i]) + '.pt',
                          map_location=device)
        data = load_data(data=data, r=4.5)
        train.append(data)
    for i in range(len(val_set[fold])):
        data = torch.load('./dataset/fea/343_' + str(val_set[fold][i]) + '.pt',
                          map_location=device)
        data = load_data(data=data, r=4.5)
        val.append(data)

    trainloader = DataLoader(train, batch_size=1, shuffle=True, drop_last=True)
    valloader = DataLoader(val, batch_size=1)

    model = Net()
    focalloss = FocalLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  #
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    n_epochs = 15000
    patience = 7
    # todo change name
    checkpoint = 'checkpoint/se3only_' + str(fold) + '.pt'
    checkpoint_list.append(checkpoint)

    # train model
    train_label_list = []
    for data in trainloader:
        train_label_list.extend(data.y)
    label_tensor = torch.tensor(train_label_list)  # 将label_list转换为PyTorch张量
    count = torch.sum(label_tensor == 1).item()  # 计算label_tensor中值为1的元素数量
    train_p_weight = (len(train_label_list) - count) / count# 计算p_weight # 将p_weight转换为PyTorch张量
    train_p_weight = torch.tensor(train_p_weight)

    val_label_list = []
    for data in bep3testloader:
        val_label_list.extend(data.y)
    label_tensor = torch.tensor(val_label_list)  # 将label_list转换为PyTorch张量
    count = torch.sum(label_tensor == 1).item()  # 计算label_tensor中值为1的元素数量
    val_p_weight = (len(val_label_list) - count) / count
    val_p_weight = torch.tensor(val_p_weight)

    print(train_p_weight, val_p_weight)
    train_loss, valid_loss = train_model(model, patience, n_epochs, checkpoint)

for fold in [0, 1, 2, 3, 4]:
    model_test(fold)


