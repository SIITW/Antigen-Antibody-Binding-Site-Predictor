import numpy as np
import pandas as pd
from Bio.PDB import *
from scipy.spatial import cKDTree
import pdb
from Bio import Entrez,SeqIO, BiopythonWarning
import warnings
import os


x_coord = np.load("1a2y_B_X_all.npy")
y_coord = np.load("1a2y_B_Y_all.npy")
z_coord = np.load("1a2y_B_Z_all.npy")
patch_coord = np.column_stack((x_coord ,y_coord ,z_coord))
parser = PDBParser()
pdb_struct = parser.get_structure('1a2y', "1a2y_B.pdb")

## Get heavy atoms
heavy_atoms=[]
heavy_orig_map = {}
k=0
for i, atom in enumerate(pdb_struct.get_atoms()):
    tags = atom.parent.get_full_id()
    if atom.element!='H' and tags[3][0]==' ': # if heavy atom and not heteroatom
        heavy_orig_map[k]=i #map heavy atom index to original pdb index
        heavy_atoms.append(atom)
        k+=1

atom_coord = np.array([list(atom.get_coord()) for atom in heavy_atoms])
atom_names = np.array([atom.get_id() for atom in heavy_atoms])
residue_id = np.array([atom.parent.id[1] for atom in heavy_atoms])
residue_name = np.array([atom.parent.resname for atom in heavy_atoms])
chain_id = np.array([atom.get_parent().get_parent().get_id() for atom in heavy_atoms])

# get start residue

#Create KD Tree
# patch_tree = cKDTree(patch_coord)
pdb_tree = cKDTree(atom_coord)

dist, idx = pdb_tree.query(patch_coord) #idx is the index of pdb heavy atoms that close to every patch from [0 to N patches]
result_pdb_idx=[]
for i in idx:
    result_pdb_idx.append(heavy_orig_map[i])

result_pdb_idx = np.array(result_pdb_idx) #index in original pdb

df = pd.DataFrame({"patch_ind": range(0, len(result_pdb_idx)),
                   "atom_ind": result_pdb_idx,
                   "res_ind": residue_id[idx],
                   "atom_name": atom_names[idx],
                   "residue_name": residue_name[idx],
                   "chain_id": chain_id[idx],
                   "dist": dist,
                   })