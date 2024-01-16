import os

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch


# 从文本文件中读取序列标识符
def read_sequence_identifiers(sequence_file):
    identifiers = []
    with open(sequence_file, 'r') as file:
        for line in file:
            identifier = line.strip()  # 移除换行符
            identifiers.append(identifier)
    return identifiers


# 在指定的PDB文件夹中查找对应的PDB文件
def find_pdb_file(pdb_folder, identifier):
    identifier_prefix = identifier.split('_')[0]  # 提取前四个字符作为PDB文件名的前缀
    for filename in os.listdir(pdb_folder):
        if filename.lower().startswith(identifier_prefix.lower()):  # 不区分大小写比较文件名前缀
            pdb_file = os.path.join(pdb_folder, filename)
            return pdb_file
    return None


# 找到物理距离最近的链
# def find_closest_chain(pdb_file, target_chain_id):
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure('structure', pdb_file)
#
#     # 创建邻居搜索器
#     neighbor_search = NeighborSearch(list(structure.get_atoms()))
#
#     # 转换目标链ID为小写
#     target_chain_id_lower = target_chain_id.lower()
#
#     # 找到目标链的模型
#     target_model = None
#     for model in structure:
#         for chain in model:
#             if chain.id.lower() == target_chain_id_lower:  # 不区分大小写比较链ID
#                 target_model = chain
#                 break
#
#     if not target_model:
#         print(f"未找到目标链 {target_chain_id} 的模型")
#         return None
#
#     # 计算距离最近的链
#     closest_chain = None
#     min_distance = float('inf')
#
#     for model in structure:
#         for chain in model:
#             if chain != target_model:
#                 # 对于每个原子，计算到目标链的最近距离
#                 for atom in chain.get_atoms():
#                     for target_atom in target_model.get_atoms():
#                         distance = atom - target_atom
#                         if distance < min_distance:
#                             min_distance = distance
#                             closest_chain = chain
#
#     if closest_chain:
#         print(f"找到与目标链 {target_chain_id} 最近的链: {closest_chain.id}")
#     else:
#         print(f"未找到与目标链 {target_chain_id} 最近的链")
#
#     return closest_chain
def calculate_centroid(chain):
    """ 计算链的质心 """
    coords = [atom.get_coord() for atom in chain.get_atoms() if atom.get_name() == 'CA']
    return np.mean(coords, axis=0) if coords else None

def find_closest_chain(pdb_file, target_chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # 转换目标链ID为小写
    target_chain_id_lower = target_chain_id.lower()

    # 找到目标链并计算其质心
    target_centroid = None
    for model in structure:
        for chain in model:
            if chain.id.lower() == target_chain_id_lower:
                target_centroid = calculate_centroid(chain)
                break
        if target_centroid is not None:
            break

    if target_centroid is None:
        print(f"未找到目标链 {target_chain_id} 或无法计算其质心")
        return None

    # 计算距离最近的链
    closest_chain = None
    min_distance = float('inf')

    for model in structure:
        for chain in model:
            if chain.id.lower() != target_chain_id_lower:
                chain_centroid = calculate_centroid(chain)
                if chain_centroid is not None:
                    distance = np.linalg.norm(chain_centroid - target_centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_chain = chain

    if closest_chain:
        print(f"找到与目标链 {target_chain_id} 最近的链: {closest_chain.id}")
    else:
        print(f"未找到与目标链 {target_chain_id} 最近的链")

    return closest_chain


if __name__ == "__main__":
    sequence_file = "/home/wxy/Desktop/piston1/sift/antibody.txt"  # 包含序列标识符的文本文件
    pdb_folder = "/home/wxy/Desktop/piston1/sift/train"  # 包含PDB文件的文件夹
    output_file = "/home/wxy/Desktop/piston1/sift/1.txt"  # 新的输出文本文件

    identifiers = read_sequence_identifiers(sequence_file)

    with open(output_file, 'w') as output:
        for identifier in identifiers:
            pdb_file = find_pdb_file(pdb_folder, identifier)
            if pdb_file:
                closest_chain = find_closest_chain(pdb_file, identifier.split('_')[1])
                if closest_chain:
                    output.write(f"For {identifier}, closest chain is {closest_chain.id}\n")
                else:
                    output.write(f"No closest chain found for {identifier}\n")
            else:
                output.write(f"PDB file not found for {identifier}\n")

    print(f"结果已写入 {output_file}")
