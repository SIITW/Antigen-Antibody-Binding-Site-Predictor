# 导入所需的库
import numpy as np

def read_npy_file_shape(file_path):
    # 读取.npy文件
    data = np.load(file_path)
    # 获取并返回数据的形状
    return data.shape

# 示例使用
file_path0 = r'/home/wxy/Desktop/se/se3/output/1A2Y_B_input_feat.npy'  # 替换为你的.npy文件路径
data=np.load(file_path0)
np.set_printoptions(threshold=np.inf)
print(data)
# 注意：这里无法直接运行，因为我无法访问外部文件系统
# 你需要在你自己的环境中运行此代码，并确保文件路径是正确的

# 调用函数并打印形状
print(read_npy_file_shape(file_path0))

