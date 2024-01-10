"""读取抗原和抗体的 PDB 文件。
计算每种蛋白质的α碳（Cα）之间的距离。
根据距离准则识别结合位点原子。
相应地标记这些原子，其中 1 表示结合位点原子，0 表示非结合位点原子。
考虑蛋白质表面三角测量，其中每个原子可以由多个顶点表示。
将标签传播到顶点，将与结合位点原子对应的所有顶点标记为 1，将其他顶点标记为 0"""
from Bio.PDB import PDBParser, Selection, NeighborSearch


def read_pdb(file_path):
    parser = PDBParser()
    structure = parser.get_structure('protein', file_path)
    return structure


def find_binding_sites(antigen, antibody, distance_threshold=4.5): # 设置阈值
    antigen_atoms = Selection.unfold_entities(antigen, 'A')  # Get all atoms from the antigen
    antibody_atoms = Selection.unfold_entities(antibody, 'A')  # Get all atoms from the antibody

    ns = NeighborSearch(antibody_atoms)
    binding_sites = set()

    for atom in antigen_atoms:
        if atom.id == 'CA':  # We're interested in alpha carbon atoms
            neighbors = ns.search(atom.coord, distance_threshold)
            for neighbor in neighbors:
                if neighbor.id == 'CA':
                    binding_sites.add(atom)
                    binding_sites.add(neighbor)

    return binding_sites


def label_atoms(structure, binding_sites):
    labeled_atoms = {}
    for atom in Selection.unfold_entities(structure, 'A'):
        labeled_atoms[atom] = 1 if atom in binding_sites else 0
    return labeled_atoms


# Triangulation and vertex labeling would require a custom function
# and possibly an external library depending on how the protein surface is represented

# Main code
antigen_file = '2a6j.pdb'
antibody_file = '5xmf.pdb'

antigen_structure = read_pdb(antigen_file)
antibody_structure = read_pdb(antibody_file)

binding_sites = find_binding_sites(antigen_structure, antibody_structure)
labeled_atoms = label_atoms(antigen_structure, binding_sites)
print(labeled_atoms)

# Add code for surface triangulation and vertex labeling
