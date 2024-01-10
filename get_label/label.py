import pymesh
import numpy as np
from sklearn.neighbors import KDTree


mesh1 = pymesh.load_mesh('1a2y_C.ply')
mesh2 = pymesh.load_mesh('1a2y_B.ply')

# 获取坐标
antigen_x = mesh1.get_attribute('vertex_x')
antigen_y = mesh1.get_attribute('vertex_y')
antigen_z = mesh1.get_attribute('vertex_z')
antigen_coord = np.stack([antigen_x, antigen_y, antigen_z], axis=1)

antibody_x = mesh2.get_attribute('vertex_x')
antibody_y = mesh2.get_attribute('vertex_y')
antibody_z = mesh2.get_attribute('vertex_z')
antibody_coord = np.stack([antibody_x, antibody_y, antibody_z], axis=1)

# 使用KD树
antibody_tree = KDTree(antibody_coord)

# 定义距离阈值
threshold = 4.5

# 查找每个抗原原子的最近邻抗体原子
distances, _ = antibody_tree.query(antigen_coord, k=1)

# 初始化标记数组
labels = np.zeros(len(antigen_coord), dtype=int)

# 根据距离设置标记
labels[distances[:, 0] < threshold] = 1

# 向mesh1添加label属性
mesh1.add_attribute('label')
mesh1.set_attribute('label', labels)
print(len(mesh1.vertices))
num_ones = np.count_nonzero(labels)
print("Number of 1s in labels:", num_ones)
print(antigen_coord)
print(antibody_coord)_feat.npy')
print(np.sum(test[:,5]))
test = np.load('1a2y_C_input