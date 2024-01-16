import os

def match_entries(file1_path, file2_path, output_dir):
    # 从两个文件中读取数据
    with open(file1_path, 'r') as file1:
        data1 = file1.readlines()

    with open(file2_path, 'r') as file2:
        data2 = file2.readlines()

    # 移除换行符并创建字典存储匹配的行
    data1 = [line.strip() for line in data1]
    data2 = [line.strip() for line in data2]
    matched_lines = []

    # 查找匹配的行
    for line1 in data1:
        for line2 in data2:
            # 如果前四个字符相同，则视为匹配
            if line1[:4] == line2[:4]:
                matched_lines.append(f"{line1},{line2}\n")

    # 构建输出文件路径
    output_file_name = "output.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    # 写入输出文件
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(matched_lines)

    return f"配对完成，输出文件为: {output_file_path}"

# 调用函数，传入两个源文件路径和输出文件夹路径
file1_path = r'/home/wxy/Desktop/piston1/sift/antigen1.txt'  # 第一个文件路径
file2_path = r'/home/wxy/Desktop/piston1/sift/antibody.txt'  # 第二个文件路径
output_dir = r'/home/wxy/Desktop/piston1/sift/'  # 输出文件夹的路径
print(match_entries(file1_path, file2_path, output_dir))
