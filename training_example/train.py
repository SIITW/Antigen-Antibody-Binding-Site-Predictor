import numpy as np

import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
#### Local import
import sys
### 这样的操作通常用于解决模块导入的问题，特别是当你的代码和模块位于不同的目录中时
sys.path.append('../../src')
from utils.dataset import PDB_complex_training
from utils.utils import get_processed
import torch
from torch import nn
from networks.ViT_pytorch import Encoder
from networks.ViT_hybrid import ViT_Hybrid_encoder
from networks.ViT_pytorch import get_ml_config
import torch.optim as optim

DIM=16

DATA_DIR = os.getcwd()+'/data_preparation/'
print(DATA_DIR)

# 目前训练只采用训练集
TRAIN_LIST_FILE = '../data/lists/training.txt'
# VAL_LIST_FILE = '../data/lists/final_val.txt'

PATIENCE=10 # number of consequitive times when model don't improve before early stopping
SEED_ID = 7272
BATCH_SIZE=1
# 最大的训练次数
MAX_EPOCH = 200

# 加载piston的训练好的数据来最为我们模型训练的初始化参数
MODEL_NAME=f'PIsToN_multiAttn_contrast'
MODEL_DIR=f'./savedModels/{MODEL_NAME}'
IMG_SIZE=32

config = {}

config['dirs'] = {}
config['dirs']['data_prepare'] = DATA_DIR
config['dirs']['grid'] = config['dirs']['data_prepare'] + '07-grid'
config['dirs']['docked'] = config['dirs']['data_prepare'] + 'docked/'
config['dirs']['tmp'] = '/aul/homes/vsteb002/tmp'

config['ppi_const'] = {}
config['ppi_const']['patch_r'] = 16 # 16

os.environ["TMP"] = config['dirs']['tmp']
os.environ["TMPDIR"] = config['dirs']['tmp']
os.environ["TEMP"] = config['dirs']['tmp']
print(config['dirs']['grid'])

# 目前阶段只采用训练集的数据
train_list = [x.strip('\n') for x in open(TRAIN_LIST_FILE, 'r').readlines()]
print(train_list)
# val_list = [x.strip('\n') for x in open(VAL_LIST_FILE, 'r').readlines()]

train_list_updated = get_processed(train_list, config)
# val_list_updated = get_processed(val_list, config)

print(f"{len(train_list_updated)}/{len(train_list)} training complexes were processed for 12A")
# print(f"{len(val_list_updated)}/{len(val_list)} validation complexes were processed for 12A")

## get all antigen and antibody lists
grid_antigen_list = []
grid_antibody_list = []
for ppi in train_list:
     antigen = ppi.split(',')[0]
     antibody = ppi.split(',')[1]
     antigen_grid_path = f"{config['dirs']['grid']}/{antigen}.npy"
     antibody_grid_path = f"{config['dirs']['grid']}/{antibody}.npy"
     if os.path.exists(antigen_grid_path ) and os.path.exists(antibody_grid_path):
         grid_antigen_list.append(np.load(antigen_grid_path,allow_pickle=True))
         grid_antibody_list.append(np.load(antibody_grid_path,allow_pickle=True))

print(f"Loaded {len(grid_antigen_list)} antigen complexes ")
print(f"Loaded {len(grid_antibody_list)} antibody complexes ")
# n个抗原，每个抗原的形状是(n,32,32,7)，抗体同理
antigen_all_grid = np.stack(grid_antigen_list,axis=0)
antibody_all_grid = np.stack(grid_antibody_list,axis=0)
radius = config['ppi_const']['patch_r']
antigen_std_array = np.ones(7)
antigen_mean_array = np.zeros(7)
antibody_std_array = np.ones(7)
antibody_mean_array = np.zeros(7)
antigen_feature_pairs = {
    'shape_index': (0,),
    'ddc': (1,),
    'electrostatics':(2,),
    'charge': (3,),
    'hydrophobicity': (4,),
    'patch_dist':(5,),
    'SASA': (6,)
    }
antibody_feature_pairs = {
    'shape_index': (0,),
    'ddc': (1,),
    'electrostatics':(2,),
    'charge': (3,),
    'hydrophobicity': (4,),
    'patch_dist':(5,),
    'SASA': (6,)
}

## compute mean and std values of antigen
for feature in antigen_feature_pairs.keys():
    print(f"Obtaining pixel values for {feature}")
    pixel_values = []
    for feature_i in antigen_feature_pairs[feature]:
        print(f"Index {feature_i}")
        for image_i in tqdm(range(antigen_all_grid.shape[0])):
            for row_i in range(antigen_all_grid.shape[1]):
                for column_i in range(antigen_all_grid.shape[2]):
                    # Check if coordinates are within the radius
                    x = column_i - radius
                    y = radius - row_i
                    if x ** 2 + y ** 2 < radius ** 2:
                        pixel_values.append(antigen_all_grid[image_i][row_i][column_i][feature_i])

    antigen_mean_value = np.mean(pixel_values)
    antigen_std_value = np.std(pixel_values)
    print(f"antigen  : Feature {feature}; Mean: {antigen_mean_value}; std: {antigen_std_value}")
    for feature_i in antigen_feature_pairs[feature]:
        antigen_mean_array[feature_i] = antigen_mean_value
        antigen_std_array[feature_i] = antigen_std_value

## compute mean and std values of antibody
for feature in antibody_feature_pairs.keys():
    print(f"Obtaining pixel values for {feature}")
    pixel_values = []
    for feature_i in antibody_feature_pairs[feature]:
        print(f"Index {feature_i}")
        for image_i in tqdm(range(antibody_all_grid.shape[0])):
            for row_i in range(antibody_all_grid.shape[1]):
                for column_i in range(antibody_all_grid.shape[2]):
                    # Check if coordinates are within the radius
                    x = column_i - radius
                    y = radius - row_i
                    if x ** 2 + y ** 2 < radius ** 2:
                        pixel_values.append(antibody_all_grid[image_i][row_i][column_i][feature_i])

    antibody_mean_value = np.mean(pixel_values)
    antibody_std_value = np.std(pixel_values)
    print(f"antibody : Feature {feature}; Mean: {antibody_mean_value}; std: {antibody_std_value}")
    for feature_i in antigen_feature_pairs[feature]:
        antibody_mean_array[feature_i] = antibody_mean_value
        antibody_std_array[feature_i] = antibody_std_value

print("Antigen Mean array:")
print(list(antigen_mean_array))
print("")
print("Antigen Standard deviation array:")
print(list(antigen_std_array))
print("Antibody Mean array:")
print(list(antibody_mean_array))
print("")
print("Antibody Standard deviation array:")
print(list(antibody_std_array))


class PISToN_proto(nn.Module):

    # 定义模型，从父类方法中继承
    def __init__(self, config, img_size=24, zero_head=False):
        # 构造调用模型的父类
        super(PISToN_proto, self).__init__()
        self.index_dict = {
            'all features': (0, 1, 2, 3, 4, 5, 6)
        }

        self.img_size = img_size
        self.zero_head = zero_head
        ### 创建一个ModuleList，将多个模块连接起来，方便管理
        self.spatial_transformers_list = nn.ModuleList()
        for feature in self.index_dict.keys():
            self.spatial_transformers_list.append(self.init_transformer(config, channels=len(self.index_dict[feature])))

        self.feature_transformer = Encoder(config, vis=True)

    def init_transformer(self, config, channels, n_individual):
        """
        Initialize Transformer Network for a given tupe of features
        :param model_config:
        :param channels:
        :param n_individual:
        :return:
        """
        return ViT_Hybrid_encoder(config, n_individual, img_size=self.img_size, channels=channels, vis=True)

    def forward(self, img, labels=None):

        all_x = []
        all_spatial_attn = []
        for i, feature in enumerate(self.index_dict.keys()):
            # 从图片中得到我们所提取的特征
            img_tmp = img[:, self.index_dict[feature], :, :]
            x, attn = self.spatial_transformers_list[i](img_tmp)

            all_x.append(x)
            all_spatial_attn.append(attn)

        # 把所有的特征综合起来（其实这里只有一个，因为我们只有一组）
        x = torch.stack(all_x, dim=1)
        # 将所有的特征在经过过一个transformer encoder的编码
        x, feature_attn = self.feature_transformer(x)
        # 进行L2归一化，使向量在[-1,1]之间
        x = nn.functional.normalize(x)
        # 最后返回的是x，npy文件经过特征之后vit和transformer encoder得到的向量
        return x, feature_attn


def train_piston(search_space, train_list, SEED_ID, IMG_SIZE,
                 antigen_std, antigen_mean, antibody_std, antibody_mean,
                 MAX_EPOCHS=50, N_FEATURES=7,
                 feature_subset=None, data_prepare_dir='./data_preparation/'):
    assert len(antigen_mean) == N_FEATURES
    if feature_subset is not None:
        assert len(antigen_mean) == len(feature_subset)
    ## 加载所有训练集的值
    train_db = PDB_complex_training(train_list,
                                    training_mode=True,
                                    feature_subset=feature_subset,
                                    data_prepare_dir=data_prepare_dir,
                                    neg_pos_ratio=search_space['neg_pos_ratio'],
                                    antigen_mean=antigen_mean,
                                    antigen_std=antigen_std,
                                    antibody_mean=antibody_mean,
                                    antibody_std=antibody_std)

    ##### Initialize data loaders
    def worker_init_fn(worker_id):
        random.seed(SEED_ID + worker_id)

    ###load the test data
    trainloader = DataLoader(train_db, batch_size=1, shuffle=True, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##### Initailize model  search_space 是什么样的参数
    ## 创建了一个模型的实例
    model_config = get_ml_config(search_space)
    model = PISToN_proto(model_config, img_size=IMG_SIZE).float()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in MAX_EPOCHS:
        for data in trainloader:
            antigen_grid, antibody_grid, ppi = data
            antigen = antigen_grid.to(device=device, dtype=torch.float)
            antibody = antibody_grid.to(device=device, dtype=torch.float)
            antigen_output = model(antigen)
            antibody_output = model(antibody)
            antigen_vector, antigen_att = antigen_output
            antibody_vector, antibody_att = antibody_output
            print(antigen_vector.shape)

            ## 计算点乘结果，损失函数
            dot_product = torch.dot(antigen_vector, antibody_vector)
            target_dot_product = torch.tensor(1.0)
            if dot_product < 1.0:
                loss = (1 - dot_product) ** 2
            else:
                loss = (dot_product - 1) ** 2

            ## 梯度清零，反向传播，参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch:", epoch + 1, "Loss:", loss.item())

params = {'dim_head': DIM,
          'hidden_size': DIM,
          'dropout': 0,
          'attn_dropout': 0,
          'lr': 0.0001,
          'n_heads': 8,
          'neg_pos_ratio': 5,
          'patch_size': 4,
          'transformer_depth': 8,
          'weight_decay': 0.0001,
          }
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_piston(params,train_list = train_list,SEED_ID=SEED_ID,IMG_SIZE=
             IMG_SIZE,antigen_mean=antigen_mean_array,antigen_std=antigen_std_array,antibody_mean=antigen_mean_array,antibody_std=antigen_std_array,data_prepare_dir=DATA_DIR)


