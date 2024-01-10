"""
Objective:
   Dataset object to read interface maps and energy terms

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""
from torch.utils.data import Dataset
import numpy as np
import random
import os

import torch

from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
#
# from collections import defaultdict

# import torchvision.transforms as transforms
from scipy import ndimage


def read_energies(energies_dir, ppi):
    """
        (0) - indx
        (1) - Lrmsd     - ligand rmsd of the final position, after the rigid-body optimization.
        (2) -Irmsd     - interface rmsd of the final position, after the rigid-body optimization.
        (3) - st_Lrmsd  - initial ligand rmsd.
        (4) - st_Irmsd  - initial ligand rmsd.
    0 - (5) - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
    1 - (6) - aVdW      - attractive van der Waals
    2 - (7) - rVdW      - repulsive van der Waals
    3 - (8) - ACE       - Atomic Contact Energy | desolvation (10.1006/jmbi.1996.0859)
    4 - (9) - inside    - "Insideness" measure, which reflects the concavity of the interface.
    5 - (10) - aElec     - short-range attractive electrostatic term
    6 - (11) - rElec     - short-range repulsive electrostatic term
    7 - (12) - laElec    - long-range attractive electrostatic term
    8 - (13) - lrElec    - long-range repulsive electrostatic term
    9 - (14) - hb        - hydrogen and disulfide bonding
    10 - (15) - piS	     - pi-stacking interactions
    11 - (16) - catpiS	  - cation-pi interactions
    12 - (17) - aliph	  - aliphatic interactions
         (18) - prob      - rotamer probability
    :param ppi:
    :return:
    """
    energies_path = f"{energies_dir}/refined-out-{ppi}.ref"

    to_read = False
    all_energies = None

    with open(energies_path, 'r') as f:
        for line in f.readlines():
            if to_read:
                all_energies = line.split('|')
                all_energies = [x.strip(' ') for x in all_energies]
                all_energies = all_energies[5:18]
                all_energies = [float(x) for x in all_energies]
                all_energies = np.array(all_energies)
                break
            if 'Sol # |' in line:
                to_read = True
    if all_energies is None:
        # energies couldn't be computed. Assign them to zero
        all_energies = np.zeros(13)

    all_energies = np.nan_to_num(all_energies)
    return all_energies

def learn_background_mask(grid):
    """
    Returns the mask with zero elements outside the patch
    :param grid: example of a grid image
    :return: mask
    """
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0] / 2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            # Check if coordinates are within the radius
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask

class PDB_complex_training(Dataset):
    # Including FireDock energies

    def __init__(self, ppi_list, training_mode, data_prepare_dir,
                 antigen_mean, antigen_std, neg_pos_ratio=1,
                 antibody_mean= None, antibody_std= None, feature_subset=None,):

        self.ppi_list = ppi_list
        self.training_mode = training_mode
        self.data_prepare_dir = data_prepare_dir
        self.feature_subset = feature_subset
        self.antigen_std = antigen_std
        self.antigen_mean = antigen_mean
        self.antibody_mean = antibody_mean
        self.antibody_std = antibody_std
        print("Length of PPI list:", len(ppi_list))

    def __len__(self):
        return len(self.ppi_list)



    def __getitem__(self, i):
        ppi = self.ppi_list[i]
        antigen = ppi.split(",")[0]
        antibody = ppi.split(",")[1]
        antigen_grid_path = f"{self.data_prepare_dir}/{antigen}.npy"
        antibody_grid_path = f"{self.data_prepare_dir}/{antibody}.npy"
        antigen_true_grid = np.load(antigen_grid_path, allow_pickle=True)
        antibody_true_grid = np.load(antibody_grid_path, allow_pickle=True)

        antigen_grid = antigen_true_grid
        antibody_grid = antibody_true_grid
        ## extract antigen or antibody coordinates
        antigen_coord = antigen_grid[:, :, -3:]
        antibody_coord = antibody_grid[:, :, -3:]
        ## extract features
        antigen_grid = antigen_grid[:, :, :7]
        antibody_grid = antibody_grid[:, :, :7]
        antigen_grid = np.swapaxes(antigen_grid, -1, 0).astype(np.float32)
        antibody_grid = np.swapaxes(antibody_grid, -1, 0).astype(np.float32)
        # Perform standard scaling of antigen
        for feature_i in range(antigen_grid.shape[0]):
            antigen_grid[feature_i, :, :] = (antigen_grid[feature_i, :, :] - self.antigen_mean[feature_i]) / \
                                           self.antigen_std[feature_i]

        # Perform standard scaling of antibody
        for feature_i in range(antibody_grid.shape[0]):
            antibody_grid[feature_i, :, :] = (antibody_grid[feature_i, :, :] - self.antibody_mean[feature_i]) / \
                                            self.antibody_std[feature_i]

        ## 假设在我们数据的最后三列是存储的我们抗原抗体的坐标信息
        return antigen_grid, antibody_grid, antigen_coord, antibody_coord, ppi






class PISToN_dataset(Dataset):
    def __init__(self, grid_dir, ppi_list, attn=None):

        ### Empirically learned mean and standard deviations:
        antigen_mean_array = [-3.68963866e-02, -3.59934378e-02, -2.66131011e-02, -3.86390779e-02, -2.84675873e-01, 5.64285555e+01, 3.26685254e-01]
        antigen_std_array = [ 0.45224719, 0.13015785, 0.19561894, 0.2836702, 0.58175112, 19.09688306, 0.22096787]
        antibody_mean_array = [-3.33899314e-03, 1.37988014e-02, -3.88253358e-02, -1.78488375e-02, -2.01681826e-01, 5.64285555e+01, 3.56284656e-01]
        antibody_std_array =[ 0.42468883, 0.14906625, 0.19346182, 0.17383524, 0.46184016, 19.09688306, 0.2192611 ]
        antigen_all_grids = []
        antibody_all_grids = []

        ppi_to_idx = {} # map ppi id to idx

        i=0
        for ppi in ppi_list:
            antigen_ppi = ppi.split(",")[0]
            antibody_ppi = ppi.split(",")[1]
            if os.path.exists(f"{grid_dir}/{antigen_ppi}.npy") and os.path.exists(f"{grid_dir}/{antibody_ppi}.npy"):
                ppi_to_idx[ppi] = i
                antigen_grid = np.load(f"{grid_dir}/{antigen_ppi}.npy", allow_pickle=True)
                antibody_grid = np.load(f"{grid_dir}/{antibody_ppi}.npy", allow_pickle=True)
                antigen_grid = antigen_grid[:, :, :7]
                antibody_grid = antibody_grid[:, :, :7]
                antigen_all_grids.append(antigen_grid)
                antibody_all_grids.append(antibody_grid)
                i+=1
        self.ppi_to_idx = ppi_to_idx



        grid_antigen = np.stack(antigen_all_grids, axis=0)
        grid_antibody = np.stack(antibody_all_grids, axis=0)
        # (n,32,32,7)
        grid_antigen = np.swapaxes(grid_antigen, -1, 1).astype(np.float32)
        grid_antibody = np.swapaxes(grid_antibody, -1, 1).astype(np.float32)

        print(f"Interaction maps shape of antigen: {grid_antigen.shape}")
        print(f"Interaction maps shape of antibody: {grid_antibody.shape}")

        ### Standard scaling

        # Interactinon maps of antigen:
        for feature_i in range(grid_antigen.shape[1]):
            grid_antigen[:, feature_i, :, :] = (grid_antigen[:, feature_i, :, :] - antigen_mean_array[feature_i]) / antigen_std_array[feature_i]
        # Interaction maps of antibody
        for feature_i in range(grid_antibody.shape[1]):
            grid_antibody[:, feature_i, :, :] = (grid_antibody[:, feature_i, :, :] - antibody_mean_array [feature_i]) / antibody_std_array[feature_i]

        self.grid_antigen = grid_antigen
        self.grid_antibody = grid_antibody
        self.grid_dir = grid_dir
        self.ppi_list = ppi_list

    def __len__(self):
        return self.grid.shape[0]

    def read_scaled(self, ppi, device):
        idx = self.ppi_to_idx[ppi]
        grid_antigen = torch.from_numpy(np.expand_dims(self.grid_antigen[idx], 0))
        grid_antibody = torch.from_numpy(np.expand_dims(self.grid_antibody[idx], 0))
        return grid_antigen.to(device), grid_antibody.to(device)


    def __getitem__(self, idx):
        return self.grid_antigen[idx], self.grid_antibody[idx]