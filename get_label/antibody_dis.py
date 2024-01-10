import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
import os
#def calculate_distance(atom1, atom2):
    # 计算距离
   # distance = atom1 - atom2
    #return distance
def calculate_distance(coord1, coord2):
    # 计算距离
    distance = np.linalg.norm(coord1 - coord2)
    return distance
def get_surface_residues(structure, cutoff=5.0):
    # 获取所有原子
    atoms = [atom for atom in structure.get_atoms()]

    # 创建NeighborSearch对象
    ns = NeighborSearch(atoms)

    # 获取表面残基
    surface_residues = []
    for atom in atoms:
        neighbors = ns.search(atom.coord, cutoff)
        if len(neighbors) > 2:
            surface_residues.append(atom.get_parent())

    return surface_residues

def get_epitope_label(input_file, pdb_path="", cutoff=5.0):
    parser = PDBParser()
    #antigen_file = "./data/one_pdb/1JTO-L.pdb"
    #antibody_file = "./data/one_pdb/1JTO-A.pdb"

    with open(input_file, 'r') as f:
        lines = f.readlines()
        pairs = [line.strip().split() for line in lines]
    os.makedirs("./output", exist_ok=True)  # 创建output文件
    with open('antibody_new_label.txt', 'w') as f:
        for pair in pairs:
            antigen_file = pdb_path + pair[0] + ".pdb"
            antibody_file = pdb_path + pair[1] + ".pdb"
            antigen_structure = parser.get_structure('antigen', antigen_file)
            antibody_structure = parser.get_structure('antibody', antibody_file)

            antigen_residues = [residue for residue in antigen_structure.get_residues()]
            antibody_residues = [residue for residue in antibody_structure.get_residues()]

            antigen_surface_residues = get_surface_residues(antigen_structure, cutoff)
            antibody_surface_residues = get_surface_residues(antibody_structure, cutoff)

            surface_list = ['1' if residue in antibody_surface_residues else '0' for residue in antibody_residues]
            label_distance = []
            for antibody_residue in antibody_residues:
                min_distance = float('inf')
                for antigen_residue in antigen_residues:
                    for antigen_atom in antigen_residue:
                        for antibody_atom in antibody_residue:
                            distance = calculate_distance(antibody_atom.coord, antigen_atom.coord)
                            if distance < min_distance:
                                min_distance = distance
                if min_distance < cutoff:
                    label_distance.append('1')
                else:
                    label_distance.append('0')
            label = ['1' if surface == '1' and distance == '1' else '0' for surface, distance in
                    zip(surface_list, label_distance)]
    # 将结果保存到文件中
    #with open('antibody_new_label.txt', 'w') as f:
    #    f.write(''.join(label))
            #output_file = pair[1] + "_label.txt"
            output_file = os.path.join("output", pair[1] + "_label.txt")  # 输出文件路径
            with open(output_file, 'w') as f:
                f.write(pair[1] + " ")
                f.write(''.join(label) + " ")
# 调用函数
#get_epitope_label("./data/one_pdb/1JTO-L.pdb", "./data/one_pdb/1JTO-A.pdb")
get_epitope_label("input.txt")