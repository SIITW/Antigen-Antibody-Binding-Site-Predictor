"""
Objective:
    The main PIsToN model.
    The network efficiently combines the ViT and Hybrid components to include spatial and attention features.
    Each branch contains an independent transformer network that learns separate latent
     vectors with spatial attention maps. The latent vectors of each branch are then aggregated with a class token into
     the transformer encoder for a final prediction. Since the interatomic distances affect the energies from each group,
     the patch distance feature was appended as an extra channel to each branch.
     The class probability was computed as the Euclidean distance to the class centroid.

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import torch
from torch import nn
from networks.ViT_pytorch import Encoder
from networks.ViT_hybrid import ViT_Hybrid_encoder
from losses.proto_loss import ProtoLoss
from losses.supCon_loss import SupConLoss

class PIsToN_multiAttn(nn.Module):

    def __init__(self, config, img_size=24, zero_head=False):
        super(PIsToN_multiAttn, self).__init__()
        # First tuple is the feature index in the image, and the second tuple is the feature index of energy terms
        self.index_dict = {
            "all_feature": (0, 1, 2, 3, 4, 5, 6)
        }


        self.img_size = img_size
        self.zero_head = zero_head

        # 创建了一个空的module list 的容器
        self.spatial_transformers_list = nn.ModuleList()
        # channel是输入能量特征的channel数，就是有多少个维度，n_individual 就是能量项的维度
        for feature in self.index_dict.keys():
            self.spatial_transformers_list.append(self.init_transformer(config, channels=len(self.index_dict[feature][0])))

        self.classifier = config.classifier
        # 特征提取器采用transformer的Encoder进行编码
        self.feature_transformer = Encoder(config, vis=True)
    def init_transformer(self, config, channels):
        """
        Initialize Transformer Network for a given tupe of features
        :param model_config:
        :param channels:
        :param n_individual:
        :return:
        """
        return ViT_Hybrid_encoder(config, img_size=self.img_size, channels=channels, vis=True)


    def forward(self, img, labels=None):
        # 遍历我们设置的每一个元组
        for i, feature in enumerate(self.index_dict.keys()):
            # img 可能是一个思维的张量(batch_size, num_channels, height, width)
            # 在npy文件中提取出对应的特征
            img_tmp = img[:,self.index_dict[feature],:,:]
            x, attn = self.spatial_transformers_list[i](img_tmp)

        return x, attn




