"""
Objective:
   The PIsToN-Hybrid component.
   The method combines empirically computed energies with the interface maps.
    The $Q$ energy terms are projected on to a latent vector using a fully connected network (FC),
                                        which is then concatenated to the vector obtained from ViT

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import torch
from torch import nn
from .ViT_pytorch import Transformer

class ViT_Hybrid(nn.Module):
    # n_individual 是一个什么样的参数
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        # 创建了一个Transformer的实例并且初始化他
        # 这个地方应该是vit的输入
        self.transformer = Transformer(config, img_size, channels, vis)
        # 这个应该是能量项的一个输入
        # self.individual_nn = nn.Linear(n_individual, n_individual)
        # 将他进行一个链接，输出的维度与vit输出的维度一致
        # self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)
        # 将整合后的再经过一个线性层，得到分类的数量
        # self.classifier_nn = nn.Linear(config.hidden_size, num_classes)
        # 定义两个激活函数
        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()


    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        # 只保留x的第一个元素，就是我们加入的特殊的可分类的标识
        # x = x[:, 0] # classification token
        # # 经过一个线性层，在经过gelu的激活函数
        # individual_x = self.individual_nn(individual_feat)
        # individual_x = self.af_ind(individual_x)
        # # 对两个在列的维度上进行拼接
        # x = torch.cat([x, individual_x], dim=1)
        # 连接之后，在经过了一个线性层，再经过了一个
        # x = self.combine_nn(x)
        # x = self.af_combine(x)
        #
        # logits = self.classifier_nn(x)

        return x, attn_weights

class ViT_Hybrid_encoder(nn.Module):
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid_encoder, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, channels, vis)
        # self.individual_nn = nn.Linear(n_individual, n_individual)
        #
        # self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)

        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()


    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        # x = x[:, 0] # classification token
        #
        # individual_x = self.individual_nn(individual_feat)
        # individual_x = self.af_ind(individual_x)
        #
        # x = torch.cat([x, individual_x], dim=1)
        #
        # x = self.combine_nn(x)
        # x = self.af_combine(x)

        #logits = self.head(x[:, 0])

        return x, attn_weights

