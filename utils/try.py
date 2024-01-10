import numpy as np

# # 设置随机数种子以便复现结果
# np.random.seed(0)
#
# array1 = np.random.rand(32, 32, 3)
# array2 = np.random.rand(32, 32, 3)
# # print(array1)
# # print(array2)
# # 保存这些数组为 .npy 文件
# np.save('array1.npy', array1)
# np.save('array2.npy', array2)
# # 加载之前保存的文件
# array1 = np.load('array1.npy')
# array2 = np.load('array2.npy')
#
#
# def compute_patches_mean(array):
#     # 初始化一个空数组来存储每个 patch 的均值
#     patches_mean = np.zeros((64, 3))
#
#     # 划分并计算每个 patch 的均值
#     patch_index = 0
#     for i in range(0, 32, 4):
#         for j in range(0, 32, 4):
#             patch = array[i:i + 4, j:j + 4, :]
#             patch_mean = np.mean(patch, axis=(0, 1))
#             patches_mean[patch_index] = patch_mean
#             patch_index += 1
#
#     return patches_mean
#
#
# # 计算两个数组的 patches 均值
# patches_mean1 = compute_patches_mean(array1)
# patches_mean2 = compute_patches_mean(array2)
#
# print(patches_mean1.shape)
# # 选择性保存这些 patches 均值
# np.save('patches_mean1.npy', patches_mean1)
# np.save('patches_mean2.npy', patches_mean2)
#
# patches_mean1 = np.load('patches_mean1.npy')
# patches_mean2 = np.load('patches_mean2.npy')
#
# # 初始化一个 64x64 的矩阵来存储距离
# distance_matrix = np.zeros((64, 64))
#
# # 计算 patches_mean1 和 patches_mean2 之间的两两距离
# for i in range(64):
#     for j in range(64):
#         distance_matrix[i, j] = np.linalg.norm(patches_mean1[i] - patches_mean2[j])
#
# print(distance_matrix.shape)
# distance_matrix[distance_matrix > 1] =0
# distance_matrix[distance_matrix <= 1] =1
# print(distance_matrix)

import numpy as np
IMG_SIZE = 32
patch = 4
num =np.square (IMG_SIZE/patch)
print(num)