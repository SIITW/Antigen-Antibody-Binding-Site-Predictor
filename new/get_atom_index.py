from Bio import PDB
import numpy as np


def get_atom_number(structure, chain_id, residue_number, residue_name, atom_name):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[1] == residue_number and residue.resname == residue_name:
                        for atom in residue:
                            if atom.id == atom_name:
                                return atom.serial_number
    return None


def main():
    pdb_file_path = '1a2y_B.pdb'
    npy_output_path = 'output.npy'

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    result_array = np.zeros((64, 64, 1), dtype=int)

    data_array = np.load('1a2y_B_resnames.npy',allow_pickle=True)

    for i in range(64):
        for j in range(64):
            chain_id, residue_number, full_residue_name, atom_name = data_array[i, j, 0].split(':')

            # 处理 full_residue_name，仅保留第一个冒号之前的部分
            residue_name = full_residue_name.split('-')[0]

            atom_number = get_atom_number(structure, chain_id, int(residue_number), residue_name, atom_name)
            result_array[i, j, 0] = atom_number if atom_number is not None else -1

    np.save(npy_output_path, result_array)


if __name__ == "__main__":
    main()

