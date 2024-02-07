from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial.distance import cdist

def extract_atom_coords(residue):
    atom_coords = []
    for atom in residue.get_atoms():
        atom_coords.append(atom.get_coord())
    return atom_coords

def extract_amino_acids(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    amino_acids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip():  # Check if the residue has a name
                    amino_acids.append(residue)

    return amino_acids

def calculate_min_distance_between_amino_acids(pdb_file1, pdb_file2):
    amino_acids1 = extract_amino_acids(pdb_file1)
    amino_acids2 = extract_amino_acids(pdb_file2)

    num_amino_acids1 = len(amino_acids1)
    num_amino_acids2 = len(amino_acids2)

    min_distances_matrix = np.zeros((num_amino_acids1, num_amino_acids2))

    for i, amino_acid1 in enumerate(amino_acids1):
        coords1 = extract_atom_coords(amino_acid1)
        for j, amino_acid2 in enumerate(amino_acids2):
            coords2 = extract_atom_coords(amino_acid2)
            distances = cdist(coords1, coords2)
            min_distance = np.min(distances)
            min_distances_matrix[i, j] = min_distance

    return min_distances_matrix

# 两个PDB文件的路径
pdb_file1 = r'/home/wxy/Desktop/新建文件夹 1/se3/dataset/bep3/11/4UUJ_B.pdb'
pdb_file2 = r'/home/wxy/Desktop/新建文件夹 1/se3/dataset/bep3/11/4UUJ_C.pdb'

min_distances_matrix = calculate_min_distance_between_amino_acids(pdb_file1, pdb_file2)
min_distances_matrix = np.where(min_distances_matrix > 4.5, 0, 1)

# 打印最小距离矩阵
print("Minimum distances matrix between amino acids:")
print(min_distances_matrix)
if 1 in min_distances_matrix:
    print("min_distances_matrix contains 1")
else:
    print("min_distances_matrix does not contain 1")