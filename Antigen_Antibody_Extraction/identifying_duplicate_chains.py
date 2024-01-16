import os

def extract_pdb_id(chain_data):
    """ 提取并返回PDB ID """
    return chain_data.split('|')[0].split('_')[0]

def find_matching_chains(input_filename, output_filename):
    """ 在文件中查找具有相同PDB ID的链并写入输出文件 """
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # 创建一个字典来保存相同PDB ID的链
    pdb_chains = {}
    for line in lines:
        pdb_id = extract_pdb_id(line.strip())
        if pdb_id in pdb_chains:
            pdb_chains[pdb_id].append(line.strip())
        else:
            pdb_chains[pdb_id] = [line.strip()]

    # 将匹配的链写入输出文件
    with open(output_filename, 'w') as output_file:
        for pdb_id, chains in pdb_chains.items():
            if len(chains) > 1:
                output_file.write(f"{pdb_id}: {', '.join(chains)}\n")

# 示例使用
input_filename = r'/home/wxy/Desktop/piston1/sift/light_chains.txt'  # 输入文件路径
output_filename = r'/home/wxy/Desktop/piston1/sift/1.txt'  # 输出文件路径
find_matching_chains(input_filename, output_filename)
