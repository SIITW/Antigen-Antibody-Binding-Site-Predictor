import numpy as np
from Bio.PDB import PDBParser, NeighborSearch

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

def get_epitope_label(antigen_file, antibody_file, cutoff=5.0):
    parser = PDBParser()
    antigen_file = antigen_file
    antibody_file = antibody_file
    antigen_structure = parser.get_structure('antigen', antigen_file)
    antibody_structure = parser.get_structure('antibody', antibody_file)

    antigen_residues = [residue for residue in antigen_structure.get_residues()]
    antibody_residues = [residue for residue in antibody_structure.get_residues()]

    antigen_surface_residues = get_surface_residues(antigen_structure, cutoff)
    antibody_surface_residues = get_surface_residues(antibody_structure, cutoff)

    surface_list = ['1' if residue in antigen_surface_residues else '0' for residue in antigen_residues]
    label_distance = []
    for antigen_residue in antigen_residues:
        min_distance = float('inf')
        for antibody_residue in antibody_residues:
            for antibody_atom in antibody_residue:
                for antigen_atom in antigen_residue:
                    distance = calculate_distance(antigen_atom.coord, antibody_atom.coord)
                    if distance < min_distance:
                        min_distance = distance
        if min_distance < cutoff:
            label_distance.append('1')
        else:
            label_distance.append('0')
    label = ['1' if surface == '1' and distance == '1' else '0' for surface, distance in
             zip(surface_list, label_distance)]
    # 将结果保存到文件中
    with open('new_label.txt', 'w') as f:
        f.write(''.join(label))


# 调用函数
get_epitope_label("1a2y_C.pdb", "1a2y_B.pdb")
