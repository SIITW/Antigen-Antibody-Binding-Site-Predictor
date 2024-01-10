import torch
import numpy as np
from networks.PIsToN_multiAttn import PIsToN_multiAttn
from networks.ViT_pytorch import get_ml_config
from utils.dataset import PISToN_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_prepare.data_prepare import preprocess
from data_prepare.get_structure import download
from utils.infer import infer_cmd

## 第一步：从输入的pdb文件中得到需要的interface map
default_params = {'dim_head': 16,
          'hidden_size': 16,
          'dropout': 0,
          'attn_dropout': 0,
          'n_heads': 8,
          'patch_size': 4,
          'transformer_depth': 8}

ppi_list,config = infer_cmd()

## 第二步：加载数据，将得到的interface map 进行一些数据处理，归一化等
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 第三步：定义模型
model = PIsToN_multiAttn()



