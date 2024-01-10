# import numpy as np
#
# # test = np.load("1FL6_L.npy")
# # matrix = test[: , :, 5]
# # matrix = np.matrix(matrix)
# # for i in range(matrix.shape[0]):
# #     for j in range(matrix.shape[1]):
# #         print(matrix[i,j])
# #
# # matrix[matrix < 4.5] = 1
# # matrix[matrix >= 4.5] = 0
# # print(matrix)
#
# import numpy as np
#
# # 假设 matrix 是你的 32x32 矩阵
# matrix = np.random.rand(32, 32)  # 示例：创建一个随机填充的 32x32 矩阵
#
# # 初始化一个 4x4 矩阵来存储每个 patch 的平均值
# patch_avg_matrix = np.zeros((4, 4))
#
# # 设置 patch 的大小
# patch_size = 8
#
# # 遍历每个 patch
# for i in range(4):
#     for j in range(4):
#         # 计算当前 patch 的范围
#         start_row, start_col = i * patch_size, j * patch_size
#         end_row, end_col = start_row + patch_size, start_col + patch_size
#
#         # 计算当前 patch 的平均值
#         patch_avg_matrix[i, j] = np.mean(matrix[start_row:end_row, start_col:end_col])
#
# # 输出结果
# print("Original Matrix:\n", matrix)
# print("\nPatch Average Matrix:\n", patch_avg_matrix)
import numpy as np

# # 假设 matrix 是你的 32x32 矩阵
# matrix = np.random.rand(32, 32)  # 示例：创建一个随机填充的 32x32 矩阵
#
# # 初始化一个长度为 16 的数组来存储每个 patch 的平均值
# patch_avg_array = np.zeros(64)
#
# # 设置 patch 的大小
# patch_size = 4
#
# # 遍历每个 patch
# for i in range(8):
#     for j in range(8):
#         # 计算当前 patch 的范围
#         start_row, start_col = i * patch_size, j * patch_size
#         end_row, end_col = start_row + patch_size, start_col + patch_size
#
#         # 计算当前 patch 的平均值，并存储在一维数组中
#         patch_avg_array[i * 8 + j] = np.mean(matrix[start_row:end_row, start_col:end_col])
#
# # 输出结果
# print("Original Matrix:\n", matrix)
# print("\nPatch Average Array:\n", patch_avg_array)
test = np.load("1FL6_L.npy")
print(test.shape)
