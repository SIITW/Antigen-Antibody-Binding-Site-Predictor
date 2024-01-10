import pdb

import torch
import numpy as np
from networks.PIsToN_multiAttn import PIsToN_multiAttn
from networks.ViT_pytorch import get_ml_config
from utils.dataset import PISToN_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_prepare.data_prepare import preprocess
from data_prepare.get_structure import download
import shutil

default_params = {'dim_head': 16,
          'hidden_size': 16,
          'dropout': 0,
          'attn_dropout': 0,
          'n_heads': 8,
          'patch_size': 4,
          'transformer_depth': 8}
def construct_default_config(pdb_dir, out_dir):

    config = {}
    config['dirs'] = {}
    config['dirs']['data_prepare'] = out_dir + '/intermediate_files/'
    config['dirs']['raw_pdb'] = pdb_dir
    config['refine'] = False
    config['dirs']['protonated_pdb'] = config['dirs']['data_prepare'] + '01-protonated_pdb/'
    # config['dirs']['refined'] = config['dirs']['data_prepare'] + '02-refined_pdb/'
    config['dirs']['cropped_pdb'] = config['dirs']['data_prepare'] + '02-cropped_pdbs/'
    config['dirs']['chains_pdb'] = config['dirs']['data_prepare'] + '03-chains_pdbs/'

    config['dirs']['surface_ply'] = config['dirs']['data_prepare'] + '04-surface_ply/'
    # config['dirs']['patch_ply'] = config['dirs']['data_prepare'] + '05-patch_ply/'
    config['dirs']['patches'] = config['dirs']['data_prepare'] + '05-patches_16R/'
    config['dirs']['grid'] = out_dir + '/grid_16R/'

    config['dirs']['tmp'] = os.getcwd() + '/tmp/'

    config['dirs']['vis'] = out_dir + '/patch_vis'

    config['ppi_const'] = {}
    config['ppi_const']['contact_d'] = 5  # minimum distance between residues to be considered as "contact point"
    config['ppi_const']['surf_contact_r'] = 1  # minimum distance between two surface points to be considered as "contact point"
    config['ppi_const']['patch_r'] = 32  # 16
    config['ppi_const']['crop_r'] = config['ppi_const']['patch_r'] + 1  # radius to crop (in Angstroms)

    config['ppi_const']['points_in_patch'] = 400  # 400 for 16 radius

    config['interact_feat'] = {}
    config['interact_feat']['atom_dist'] = True
    config['interact_feat']['dssp'] = True

    config['model'] = os.path.dirname(os.path.abspath(__file__)) + "/../saved_models/PIsToN_multiAttn_contrast.pth"

    config['mesh'] = {}
    config['mesh']['mesh_res'] = 1.0  # resolution of the mesh

    # Create Directories
    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)

    # DL parameters
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    return config

def infer_from_model(ppi_list, grid_dir, model_path, params, device, radius):
    """
    Obtain PIsToN scores
    ------------------------------------------------------------------------------
        ppi_list - list of protein complexes
        grid_dir - directory with pre-processed interface maps
        model_path - path to pre-train PIsToN model
        params - model parameters (dim_head, hidden_size, n_heads, transformer_depth)
        device - device to use for inference (ex. cpu)
        radius - radius of the patch (12A, 16A, or 20A)
    ------------------------------------------------------------------------------
    Return:
        output - score for each complex in ppi_list
        attn - list of attention maps for each complex in ppi_list
    """
    device = torch.device(device)

    model_config = get_ml_config(params)

    model_antigen = PIsToN_multiAttn(model_config, img_size=radius * 2).float().to(device)
    model_antibody = PIsToN_multiAttn(model_config, img_size=radius * 2).float().to(device)
    # 加载预训练模型
    model_antigen.load_state_dict(torch.load(model_path, map_location=device))
    model_antibody.load_state_dict(torch.load(model_path, map_location=device))
    model_parameters_antigen = filter(lambda p: p.requires_grad, model_antigen.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters_antigen])
    print(f"Loaded PIsToN model with {n_params} trainable parameters. Radius of the patch: {radius}A")

    ## Constructing a dataset
    dataset = PISToN_dataset(grid_dir, ppi_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    with torch.no_grad():
        for instance in tqdm(dataloader):
            grid_antigen, grid_antibody, _, _,  = instance
            grid_antigen = grid_antigen.to(device)
            grid_antigen = grid_antigen.to(device)
            model_antigen = model_antigen.to(device)
            model_antibody = model_antibody.to(device)
            antigen_vector = model_antigen(grid_antigen)
            antibody_vector = model_antibody(grid_antigen)
    return antigen_vector,antibody_vector

def infer_cmd(args):
    """
    Obtain PIsToN scores (from command line)
    """
    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')
    if (args.list is not None):
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif (args.ppi is not None):

        ppi_list = [args.ppi]

    out_dir = args.out_dir
    pdb_dir = args.pdb_dir
    # print(out_dir,pdb_dir)
    print(f"Obtaining scores for {len(ppi_list)} complexes...")

    ## Step 1 - Prepare interface map from PDB file
    config = construct_default_config(pdb_dir, out_dir)
    download(ppi_list, config)
    preprocess(ppi_list, config)
    ## Remove intermediate files
    # shutil.rmtree(config['dirs']['data_prepare'])

    # Step 2 - Run PIsToN model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # antigen_vector,antibody_vector =infer_from_model(ppi_list,
    #                             grid_dir=config['dirs']['grid'],
    #                             model_path=config['model'],
    #                             params=default_params,
    #                             device=device,
    #                             radius=config['ppi_const']['patch_r'])

    print("contrastive learning finished")
    # add image registration part here
    # antigen_vector is the feature using vit dimension is (1,64,1)
    



    return











