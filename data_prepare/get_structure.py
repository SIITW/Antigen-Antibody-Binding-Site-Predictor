import shutil

import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from Bio.PDB import *
from subprocess import Popen, PIPE
from masif.source.input_output.protonate import protonate
import os
import pdb
from scipy.spatial import cKDTree
import numpy as np
import time
from utils.utils import get_date, extract_pdb_chain
from tqdm import tqdm

def protonate_pdb(ppi, config):
    """
    downlaod and add hydrogens to PPI
    """
    antigen = ppi.split(',')[0]
    antibody = ppi.split(',')[1]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]

    # Download pdb
    antigen_pdb_filename = config['dirs']['raw_pdb'] + antigen_pid + '.pdb'
    antibody_pdb_filename = config['dirs']['raw_pdb'] + antibody_pid + '.pdb'
    if not os.path.exists(antigen_pdb_filename) and not os.path.exists(antibody_pdb_filename):
        pdbl = PDBList()
        antigen_pdb_filename = pdbl.retrieve_pdb_file(antigen_pid, pdir=config['dirs']['raw_pdb'], file_format='pdb')
        antibody_pdb_filename = pdbl.retrieve_pdb_file(antibody_pid, pdir=config['dirs']['raw_pdb'], file_format='pdb')
    else:
        ## Remove MODEL line of antigen chain
        tmp_filename = config['dirs']['raw_pdb'] + antigen_pid + '_tmp.pdb'
        os.rename(antigen_pdb_filename, tmp_filename)
        with open(antigen_pdb_filename, 'w') as out:
            with open(tmp_filename, 'r') as f:
                for line in f:
                    if "MODEL" not in line:
                        out.write(line)
        os.remove(tmp_filename)
        ## remove the MODEL line of the antibody chain
        tmp_filename = config['dirs']['raw_pdb'] + antibody_pid + '_tmp.pdb'
        os.rename(antibody_pdb_filename, tmp_filename)
        with open(antibody_pdb_filename, 'w') as out:
            with open(tmp_filename, 'r') as f:
                for line in f:
                    if "MODEL" not in line:
                        out.write(line)
        os.remove(tmp_filename)

    # Protonate downloaded file
    # protonate the antigen chain and save document
    antigen_protonated_file = config['dirs']['protonated_pdb']+"/"+antigen_pid+".pdb"
    protonate(antigen_pdb_filename, antigen_protonated_file)
    # protonate the antibody chain and save the document
    antibody_protonated_file = config['dirs']['protonated_pdb'] + "/" + antibody_pid + ".pdb"
    protonate(antibody_pdb_filename, antibody_protonated_file)

# download the original document of antigen and antibody
def download(ppi_list, config, to_write=None):
    start = time.time()
    print("**** [ {} ] Start Downloading PDBs...".format(get_date()))
    print(ppi_list)

    processed_ppi = []
    for i in tqdm(range(len(ppi_list))):
        ppi = ppi_list[i]
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antibody_pid = antibody.split('_')[0]

        antigen_raw_pdb_filename = config['dirs']['protonated_pdb']+"/"+antigen_pid+".pdb"
        antibody_raw_pdb_filename = config['dirs']['protonated_pdb']+"/"+antibody_pid+".pdb"

        if not os.path.exists(antigen_raw_pdb_filename) and not os.path.exists(antibody_raw_pdb_filename):
            protonate_pdb(ppi, config)
        else:
            print("PDB file {} already exists. Skipping...".format(antigen_pid))
            print("PDB file {} already exists. Skipping...".format(antibody_pid))

        if os.path.exists(antigen_raw_pdb_filename) and os.path.exists(antibody_raw_pdb_filename):
            processed_ppi.append(ppi)

    if to_write is not None:
        with open(to_write, 'w') as out:
            for ppi in processed_ppi:
                out.write(ppi+'\n')

    print("**** [ {} ] Done with downloading PDBs.".format(get_date()))
    print("**** [ {} ] Took {:.2f}min.".format(get_date(), (time.time()-start)/60))
    return processed_ppi


# if pdb file has mutiple models we choose only one model
def select_single_model(pdb_path, pdb_path_updated):
    with open(pdb_path_updated, 'w') as out:
        with open(pdb_path, 'r') as f:
            for line in f.readlines():
                if line[:5]=="MODEL" or line[:6]=="REMARK":
                    pass
                elif line[:6]=="ENDMDL":
                    break
                else:
                    out.write(line)

def get_coord_dict(pid, pdb_path, chain):

    parser = PDBParser(QUIET=True)
    #pdb.set_trace()
    try:
        pdb_struct = parser.get_structure(pid, pdb_path)
    except ValueError: # tbe PDB file contain multiple models
        pdb_path_updated = pdb_path.replace('.pdb','') + '_singleModel.pdb'
        select_single_model(pdb_path, pdb_path_updated)
        pdb_struct = parser.get_structure(pid, pdb_path_updated)

    RES_dict = {'atom_id': [], 'res_id': [], 'chain_id': [], 'atom_coord': []}

    all_atom_res_chain_pairs = []

    for i, atom in enumerate(pdb_struct.get_atoms()):
        # atom_name = atom.name
        # res_name = atom.parent.resname
        res_id = atom.parent.id[1]
        chain_id = atom.get_parent().get_parent().get_id()
        atom_coord = list(atom.get_coord())
        atom_id = atom.serial_number

        if chain_id in chain:
            # The condition below will make sure that if PDB has multiple models, only the first one will be included.
            if (atom_id, res_id, chain_id) not in all_atom_res_chain_pairs:
                all_atom_res_chain_pairs.append((atom_id, res_id, chain_id))
                RES_dict['atom_id'].append(atom_id)
                RES_dict['res_id'].append(res_id)
                RES_dict['chain_id'].append(chain_id)
                RES_dict['atom_coord'].append(atom_coord)
    return RES_dict

def crop_pdb_one(ppi, config, use_refined=False):
    antigen= ppi.split(',')[0]
    antibody = ppi.split(',')[1]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]
    if use_refined:
        raise ValueError("error,we do not refine antigen and antibody")
    if not use_refined :
        print(f"Loading original PDB...")
        antigen_pdb_file = f"{config['dirs']['protonated_pdb']}/{antigen_pid}.pdb"
        antibody_pdb_file = f"{config['dirs']['protonated_pdb']}/{antibody_pid}.pdb"

    crop_r = config['ppi_const']['crop_r']
    contact_d = config['ppi_const']['contact_d']
    antigen_out_file = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
    antibody_out_file = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
    if os.path.exists(antigen_out_file) and os.path.exists(antibody_out_file):
        # Skip if file already exists
        print("Cropped PDB already exists. Skipping")
        return

    res_dict_1 = get_coord_dict(antigen_pid, antigen_pdb_file, antigen_ch)
    res_dict_2 = get_coord_dict(antibody_pid,antibody_pdb_file,antibody_ch)

    if len(res_dict_1['res_id']) == 0 and len(res_dict_2['res_id']) == 0:
        # The PDB file is empty. Skpping...
        print("ERROR::PDB file is empty")
        return

    # Search for contact points within

    with open(antigen_out_file, 'w') as out:
        with open(antigen_pdb_file, 'r') as f:
            for line in f.readlines():
                out.write(line)

    with open(antibody_out_file, 'w') as out:
        with open(antibody_pdb_file, 'r') as f:
            for line in f.readlines():
                out.write(line)


    extract_pdb_chain(antigen_out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(antigen_pid, antigen_ch), antigen_ch)
    extract_pdb_chain(antibody_out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(antibody_pid, antibody_ch), antibody_ch)

def crop_pdb(ppi_list, config):
    print("**** [ {} ] Cropping complexes to {}A radius from interaction center...".format(get_date(),
                                                                                   config['ppi_const']['crop_r']))
    processed_ppi = []
    for ppi in tqdm(ppi_list):
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]
        antigen_pdb_file = f"{config['dirs']['protonated_pdb']}/{antigen_pid}.pdb"
        antibody_pdb_file = f"{config['dirs']['protonated_pdb']}/{antibody_pid}.pdb"
    crop_r = config['ppi_const']['crop_r']
    contact_d = config['ppi_const']['contact_d']
    antigen_out_file = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
    antibody_out_file = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
    if os.path.exists(antigen_out_file) and os.path.exists(antibody_out_file):
        # Skip if file already exists
        print("Cropped PDB already exists. Skipping")
        return

    res_dict_1 = get_coord_dict(antigen_pid, antigen_pdb_file, antigen_ch)
    res_dict_2 = get_coord_dict(antibody_pid, antibody_pdb_file, antibody_ch)

    if len(res_dict_1['res_id']) == 0 and len(res_dict_2['res_id']) == 0:
        # The PDB file is empty. Skpping...
        print("ERROR::PDB file is empty")
        return

    # Search for contact points within

    with open(antigen_out_file, 'w') as out:
        with open(antigen_pdb_file, 'r') as f:
            for line in f.readlines():
                out.write(line)

    with open(antibody_out_file, 'w') as out:
        with open(antibody_pdb_file, 'r') as f:
            for line in f.readlines():
                out.write(line)

    extract_pdb_chain(antigen_out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(antigen_pid, antigen_ch),antigen_ch)
    extract_pdb_chain(antibody_out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(antibody_pid, antibody_ch), antibody_ch)
    return processed_ppi
