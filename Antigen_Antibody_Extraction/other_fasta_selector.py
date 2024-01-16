import os
import shutil

# 源文件夹路径
source_folder = r"/home/wxy/Desktop/piston1/sift/train"

# 目标文件夹路径，您可以根据需要更改
destination_folder = r"/home/wxy/Desktop/piston1/sift/old"

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹中的所有文件
file_list = os.listdir(source_folder)

# 遍历文件列表，将所有.fasta文件移动到目标文件夹
for file_name in file_list:
    if file_name.endswith(".fasta"):
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        shutil.move(source_file_path, destination_file_path)

print("所有.fasta文件已移动到新文件夹：", destination_folder)
