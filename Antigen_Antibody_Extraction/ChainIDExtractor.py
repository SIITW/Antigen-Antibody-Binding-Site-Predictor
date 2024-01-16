import os
import re

def extract_chain_id(input_string):
    chain_id = None
    # 去掉字符串开头的 '>'
    input_string = input_string.lstrip('>')
    parts = input_string.split('|')
    if len(parts) >= 2:
        # 从第一个部分获取PDB ID
        pdb_id = parts[0].split('_')[0]
        chain_part = parts[1].strip()

        # 使用正则表达式提取链ID
        # 使用正则表达式提取链ID
        auth_match = re.search(r'\[auth (\w+)\]', chain_part)
        chain_match = re.search(r'Chain[s]? (\w+)', chain_part)

        if auth_match:
            chain_id = auth_match.group(1)  # 如果找到[auth X]格式，提取X
        elif chain_match:
            chain_id = chain_match.group(1)  # 否则，尝试提取Chain(s)后的字符

    return f"{pdb_id}_{chain_id}" if chain_id else None

def transform_chain_data(input_filename, output_dir, output_filename):
    # 打开输入文件并读取内容
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # 转换每一行数据
    transformed_lines = []
    for line in lines:
        chain_id = extract_chain_id(line.strip())
        if chain_id:
            transformed_lines.append(chain_id)

    # 构造输出文件的完整路径
    output_filepath = os.path.join(output_dir, output_filename)

    # 将转换后的数据写入输出文件
    with open(output_filepath, 'w') as file:
        for line in transformed_lines:
            file.write(line + '\n')

# 示例调用函数处理数据，并自定义输出文件名
input = r'/home/wxy/Desktop/piston1/sift/heavy_chains.txt'
output = r'/home/wxy/Desktop/piston1/sift'
output_filename='antibody.txt'
transform_chain_data(input, output, output_filename)