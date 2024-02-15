import numpy as np
from Bio.PDB import PDBParser

# Kyte Doolittle scale for amino acids
kd_scale = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
}

def compute_hydrophobicity_for_atoms(structure):
    hydrophobicity_scores = []
    atom_names = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Consider only standard residues
                    aa_name = residue.get_resname()
                    hydrophobicity_score = kd_scale.get(aa_name, np.nan)  # Use np.nan for unknown amino acids
                    for atom in residue:
                        hydrophobicity_scores.append(hydrophobicity_score)
                        atom_names.append(atom.get_full_id())
    return hydrophobicity_scores, atom_names

def compute_hydrophobicity_scores(pdb_file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file_path)
    return compute_hydrophobicity_for_atoms(structure)