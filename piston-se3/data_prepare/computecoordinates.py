import numpy as np
import os

from Bio.PDB import PDBParser


def extract_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    atoms = structure.get_atoms()
    coordinates = [atom.get_coord() for atom in atoms]
    return np.array(coordinates)


def save_pdb_folder_to_npy(pdb_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_folder, filename)
            coordinates = extract_coordinates(pdb_path)

            # 构建输出文件名（将.pdb替换为.npy）
            npy_filename = filename.replace(".pdb", "_coordinates.npy")
            npy_path = os.path.join(output_folder, npy_filename)

            # 保存为.npy文件
            np.save(npy_path, coordinates)
            print(f"Saved {npy_path}")
