import numpy as np
import torch
from torchvision.transforms import functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification

# 加载你的npy文件
data = np.load('your_file.npy')

# 转换为PyTorch张量
tensor_data = torch.tensor(data, dtype=torch.float32)

# 将数据转换为图像，并进行必要的预处理
transformer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
inputs = transformer(images=tensor_data).pixel_values  # pixel_values是ViTFeatureExtractor的输出

# 构建ViT模型
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# 将数据输入模型进行编码
with torch.no_grad():
    outputs = model(inputs)

# 提取特征向量
feature_vector = outputs.last_hidden_state.mean(dim=1)  # 取平均作为特征向量

# 将特征向量展平为256x1的向量
flattened_vector = feature_vector.view(-1)

print(flattened_vector.shape)
