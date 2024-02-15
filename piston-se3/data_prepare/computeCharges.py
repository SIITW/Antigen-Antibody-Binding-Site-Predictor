from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

from default_config.chemistry import (
    polarHydrogens,
    radii,
    acceptorAngleAtom,
    acceptorPlaneAtom,
    hbond_std_dev,
    donorAtom,
)
# Compute the absolute value of the deviation from theta
def computeAngleDeviation(a, b, c, theta):
    return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)


# Compute the angle deviation from a plane
def computePlaneDeviation(a, b, c, d):
    dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


# angle_deviation from ideal value. TODO: do a more data-based solution
def computeAnglePenalty(angle_deviation):
    # Standard deviation: hbond_std_dev
    return max(0.0, 1.0 - (angle_deviation / (hbond_std_dev)) ** 2)


def isPolarHydrogen(atom_name, res):
    if atom_name in polarHydrogens[res.get_resname()]:
        return True
    else:
        return False


def isAcceptorAtom(atom_name, res):
    if atom_name.startswith("O"):
        return True
    else:
        if res.get_resname() == "HIS":
            if atom_name == "ND1" and "HD1" not in res:
                return True
            if atom_name == "NE2" and "HE2" not in res:
                return True
    return False


# Compute the charge of a vertex in a residue.
def computeChargeHelper(atom_name, res, v):
    res_type = res.get_resname()
    # Check if it is a polar hydrogen.
    if isPolarHydrogen(atom_name, res):
        donor_atom_name = donorAtom[atom_name]
        a = res[donor_atom_name].get_coord()  # N/O
        b = res[atom_name].get_coord()  # H
        # Donor-H is always 180.0 degrees, = pi
        angle_deviation = computeAngleDeviation(a, b, v, np.pi)
        angle_penalty = computeAnglePenalty(angle_deviation)
        return 1.0 * angle_penalty
    # Check if it is an acceptor oxygen or nitrogen
    elif isAcceptorAtom(atom_name, res):
        acceptor_atom = res[atom_name]
        b = acceptor_atom.get_coord()
        try:
            a = res[acceptorAngleAtom[atom_name]].get_coord()
        except:
            return 0.0
        # 120 degress for acceptor
        angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
        # TODO: This should not be 120 for all atoms, i.e. for HIS it should be
        #       ~125.0
        angle_penalty = computeAnglePenalty(angle_deviation)
        plane_penalty = 1.0
        if atom_name in acceptorPlaneAtom:
            try:
                d = res[acceptorPlaneAtom[atom_name]].get_coord()
            except:
                return 0.0
            plane_deviation = computePlaneDeviation(d, a, b, v)
            plane_penalty = computeAnglePenalty(plane_deviation)
        return -1.0 * angle_penalty * plane_penalty
        # Compute the
    return 0.0

# Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
def computeSatisfied_CO_HN(atoms):
    ns = NeighborSearch(atoms)
    satisfied_CO = set()
    satisfied_HN = set()
    for atom1 in atoms:
        res1 = atom1.get_parent()
        if atom1.get_id() == "O":
            neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
            for atom2 in neigh_atoms:
                if atom2.get_id() == "H":
                    res2 = atom2.get_parent()
                    # Ensure they belong to different residues.
                    if res2.get_id() != res1.get_id():
                        # Compute the angle N-H:O, ideal value is 180 (but in
                        # helices it is typically 160) 180 +-30 = pi
                        angle_N_H_O_dev = computeAngleDeviation(
                            res2["N"].get_coord(),
                            atom2.get_coord(),
                            atom1.get_coord(),
                            np.pi,
                        )
                        # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                        angle_H_O_C_dev = computeAngleDeviation(
                            atom2.get_coord(),
                            atom1.get_coord(),
                            res1["C"].get_coord(),
                            8 * np.pi / 9,
                        )
                        ## Allowed deviations: 30 degrees (pi/6) and 20 degrees
                        #       (pi/9)
                        if (
                            angle_N_H_O_dev - np.pi / 6 < 0
                            and angle_H_O_C_dev - np.pi / 9 < 0.0
                        ):
                            satisfied_CO.add(res1.get_id())
                            satisfied_HN.add(res2.get_id())
    return satisfied_CO, satisfied_HN


def computeCharges(pdb_filename):
    parser = PDBParser(QUIET=False)
    struct = parser.get_structure('PDB', pdb_filename)
    residues = {}
    print(f"Total number of models in the structure: {len(struct)}")
    for model in struct:
        print(f"Model ID: {model.id}, Number of chains: {len(model)}")
        for chain in model:
            print(f"Chain ID: {chain.id}, Number of residues: {len(chain)}")

    for res in struct.get_residues():
        chain_id = res.get_parent().get_id()
        residues[(chain_id, res.get_id())] = res
        print(f"Residue: {res.get_resname()} (Chain {chain_id}, ID {res.get_id()})")

    atoms = Selection.unfold_entities(struct, "A")
    print(f"Total number of atoms selected: {len(atoms)}")
    # 初始化电荷数组，长度与atoms相同，所有值默认为0
    charges = np.zeros(len(atoms))

    # 添加一些原子信息的打印语句，以便于跟踪
    for i, atom in enumerate(atoms):
        print(
            f"Atom {i}: {atom.get_name()}, Residue: {atom.get_parent().get_resname()}, Chain: {atom.get_parent().get_parent().get_id()}, Charge: {charges[i]}")

    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)

    # 遍历每个原子计算电荷
    for ix, atom in enumerate(atoms):
        chain_id = atom.get_parent().get_parent().get_id()
        res = atom.get_parent()
        res_id = res.get_id()
        atom_name = atom.get_name()

        # 检查原子是否属于已满足的C=O:H-N对，如果是，则跳过
        if (atom_name == "H" and res_id in satisfied_HN) or (atom_name == "O" and res_id in satisfied_CO):
            continue

        # 计算并更新当前原子的电荷
        charge = computeChargeHelper(atom_name, residues[(chain_id, res_id)], atom.get_coord())
        charges[ix] = charge

    return charges