###
# Define a toy model: more complex models in experiments/qm9/models.py
###

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import DataLoader
from se3.equivariant_attention.fibers import Fiber
from se3.equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormSE3, GConvSE3

torch.manual_seed(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# The maximum feature type is harmonic degree 3
num_degrees = 1
num_features = 6    #node data dim
edge_dim = 12  #edge data dim

fiber_in = Fiber(1, num_features)
fiber_mid = Fiber(num_degrees, 32)
fiber_out = Fiber(1, 128)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GSE3Res = GSE3Res(fiber_in, fiber_mid,edge_dim)
        self.GNormSE3 = GNormSE3(fiber_mid)
        self.GConvSE3 = GConvSE3(fiber_mid, fiber_out,edge_dim=edge_dim,self_interaction=True)

    def bulid_se3(self):
        se3 = nn.ModuleList([self.GSE3Res,
                       self.GNormSE3,
                       self.GConvSE3
                       ])
        return se3
    def forward(self, data1, data2):

        #antigen
        G1 = data1.G[0]
        basis, r = get_basis_and_r(G1, num_degrees - 1)
        AntigenSe3Transformer = self.bulid_se3()
        features = {'0': G1.ndata['f']}
        for layer in AntigenSe3Transformer:
            features = layer(features, G=G1, r=r, basis=basis)
        AntigenOut = features['0'][..., -1].unsqueeze(0)
        #antibody
        G2 = data2.G[0]
        basis, r = get_basis_and_r(G2, num_degrees - 1)
        AntibodySe3Transformer = self.bulid_se3()
        features = {'0': G2.ndata['f']}
        for layer in AntibodySe3Transformer:
            features = layer(features, G=G2, r=r, basis=basis)
        AntibodyOut = features['0'][..., -1].unsqueeze(0)
        #
        out = torch.matmul(AntigenOut.squeeze(),AntibodyOut.squeeze().T)
        return out





def train_model(model, n_epochs):
    for epoch in range(1, n_epochs + 1):
        model.train()
        for data1,data2 in zip(antigenloader,antibodyloader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
    return out


train1 = []
train2 = []
data1 = torch.load('./se3/dataset/fea/1a2y_1.pt',
                  map_location=device)
train1.append(data1)
data2 = torch.load('./se3/dataset/fea/1a2y_2.pt',
                  map_location=device)
train2.append(data2)
antigenloader = DataLoader(train1, batch_size=1, shuffle=True, drop_last=True)
antibodyloader = DataLoader(train2, batch_size=1, shuffle=True, drop_last=True)

model = Net()
model = model.to(device)

n_epochs = 1
out = train_model(model, n_epochs)
print(out.size())
