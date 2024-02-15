from Bio.PDB import PDBParser, DSSP
import numpy as np
import os

# 设置 LD_LIBRARY_PATH 环境变量
os.environ["LD_LIBRARY_PATH"] = "/home/wxy/anaconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

def compute_sasa_and_assign_to_atoms(pdb_file, chain_id, dssp_executable):
    parser = PDBParser()
    structure = parser.get_structure('PDB_structure', pdb_file)
    model = structure[0]  # 通常使用第一个模型

    # 使用DSSP计算二级结构和SASA，不直接指定链ID
    try:
        dssp = DSSP(model, pdb_file, dssp=dssp_executable)
    except Exception as e:
        print(f"Error using DSSP in {pdb_file}: {e}")
        return []

    # 现在，筛选出特定链的DSSP结果
    residue_sasa = {key: value[3] for key, value in dssp.property_dict.items() if key[0] == chain_id}

    atom_sasa = []
    if chain_id in model:
        for residue in model[chain_id]:
            residue_id = (chain_id, residue.id)
            sasa = residue_sasa.get(residue_id, 0)
            for atom in residue:
                atom_sasa.append((atom.get_full_id(), sasa))
    else:
        print(f"Chain {chain_id} not found in {pdb_file}")

    return atom_sasa

def process_pdb_list(file_path, pdb_folder, dssp_executable):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        chain_ids_with_pdb = line.strip().split(',')
        if len(chain_ids_with_pdb) != 2:
            print(f"Invalid format in line: {line.strip()}")
            continue

        for chain_id_with_pdb in chain_ids_with_pdb:
            pdb_id, chain_id = chain_id_with_pdb.split('_')
            # 修改文件名的格式以匹配如5E8E_L.pdb
            pdb_file = f'{pdb_folder}/{pdb_id}_{chain_id}.pdb'

            if not os.path.exists(pdb_file):
                print(f"File not found: {pdb_file}")
                continue

            atom_sasa = compute_sasa_and_assign_to_atoms(pdb_file, chain_id, dssp_executable)
            if not atom_sasa:
                continue  # 如果链不存在或有其他问题，跳过

            sasa_values = [sasa for _, sasa in atom_sasa]
            sasa_array = np.array(sasa_values).reshape(-1, 1)

            output_file = f'../output/{pdb_id}_{chain_id}_dssp_features.npy'
            np.save(output_file, sasa_array)
            print(f"SASA values for {pdb_id}_{chain_id} have been saved to {output_file}.")


# 你的文本文件路径
list_file_path = r'/home/wxy/Desktop/se/se3/train.txt'
# PDB文件夹位置
pdb_folder = r'/home/wxy/Desktop/se/se3/pdb'
# DSSP可执行文件路径
dssp_executable = '/home/wxy/anaconda3/envs/se31/bin/mkdssp'

process_pdb_list(list_file_path, pdb_folder, dssp_executable)
