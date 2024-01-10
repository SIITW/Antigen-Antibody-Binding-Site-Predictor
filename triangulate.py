import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
import numpy as np
import os
import pdb

from shutil import copyfile, rmtree

# Local includes
from utils.utils import get_date, extract_pdb_chain

# MaSIF includes
import pymesh
import traceback

from masif.source.default_config.masif_opts import masif_opts
from masif.source.triangulation.computeMSMS import computeMSMS
from masif.source.triangulation.fixmesh import fix_mesh
from masif.source.input_output.extractPDB import extractPDB
from masif.source.input_output.save_ply import save_ply
from masif.source.triangulation.computeHydrophobicity import computeHydrophobicity
from masif.source.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from masif.source.triangulation.computeAPBS import computeAPBS
from masif.source.triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

from tqdm import tqdm

def triangulate_one(pid, ch, config, pdb_filename):
    """
    triangulate one chain
    """
    chains_pdb_dir = config['dirs']['chains_pdb']

    tmp_pdb_dir = chains_pdb_dir + pid + '_' + ch + '/'
    if not os.path.exists(tmp_pdb_dir):
        os.mkdir(tmp_pdb_dir)

    #pdb_filename = config['dirs']['cropped_pdb'] + pid + '.pdb'

    # Extract chains for each interacting protein
    out_filename1 = tmp_pdb_dir + pid + '_' + ch
    extractPDB(pdb_filename, out_filename1+ '.pdb', ch)

    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+ '.pdb', protonate=True)

    #exit()
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    vertex_hphobicity = computeHydrophobicity(names1)


    vertices2 = vertices1
    faces2 = faces1

    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, config['mesh']['mesh_res'])


    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

    # hbonds
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                                          vertex_hbond, masif_opts)

    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, \
                                               vertex_hphobicity, masif_opts)


    # tmp_charges_dir = out_filename1 + '/'
    # if not os._exists(tmp_charges_dir):
    #     os.mkdir(tmp_charges_dir)
    # tmp_base = tmp_charges_dir + pid + '_' + ch
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)

    #copyfile(out_filename1, chains_pdb_dir + '{}_{}.pdb'.format(pid, ch))
    extract_pdb_chain(config['dirs']['protonated_pdb'] + pid + '.pdb',  chains_pdb_dir + '{}_{}.pdb'.format(pid, ch), ch)
    rmtree(tmp_pdb_dir)

    iface = np.zeros(len(regular_mesh.vertices))

    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)  # Square d, because this is how it was in the pyflann version.
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.

    outply = config['dirs']['surface_ply'] + pid + '_' + ch
    save_ply(outply + ".ply", regular_mesh.vertices, \
             regular_mesh.faces, normals=vertex_normal, charges=vertex_charges, \
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, \
             iface=iface)

    return

def triangulate_single(ppi, config, overwrite=False):
    antigen = ppi.split(',')[0]
    antibody = ppi.split(',')[1]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]

    antigen_outply = config['dirs']['surface_ply'] + antigen_pid + '_' + antigen_ch + '.ply'
    antibody_outply = config['dirs']['surface_ply'] + antibody_pid + '_' + antibody_ch + '.ply'
    if not overwrite and os.path.exists(antigen_outply) \
            and os.path.exists(antibody_outply) or os.path.exists(f"{config['dirs']['grid']}/{antigen}.npy") and os.path.exists(f"{config['dirs']['grid']}/{antibody}.npy"):
        # Skip if ply file is already existsc
        print("Triangulated structures already exist for {}. Skipping...".format(ppi))
        return

    #pdb_filename = config['dirs']['cropped_pdb'] + pid + '.pdb'
    try:
        antigen_pdb_filename = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
        antibody_pdb_filename = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
        triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename)
        triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename)
    except:
        print("WARNING:: can't triangulate cropped PDB")
        antigen_pdb_filename = config['dirs']['protonated_pdb'] + antigen_pid + '.pdb'
        antibody_pdb_filename = config['dirs']['protonated_pdb'] + antibody_pid + '.pdb'
        triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename)
        triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename)
    return

def triangulate(ppi_list, config):
    print("\t[ {} ] Start triangulation... ".format(get_date()))
    print(ppi_list)

    processed_ppi = []

    for ppi in tqdm(ppi_list):
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]

        antigen_outply = config['dirs']['surface_ply'] + antigen_pid + '_' + antigen_ch + '.ply'
        antibody_outply = config['dirs']['surface_ply'] + antibody_pid + '_' + antibody_ch + '.ply'
        if os.path.exists(antigen_outply) and os.path.exists(antibody_outply) or os.path.exists(f"{config['dirs']['grid']}/{antigen}.npy") and os.path.exists(f"{config['dirs']['grid']}/{antibody}.npy"):
            # Skip if ply file is already exists
            print("Triangulated structures already exist for {}. Skipping...".format(ppi))
            processed_ppi.append(ppi)
            continue

        try:
            antigen_pdb_filename = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
            antibody_pdb_filename = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
            triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename)
            triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename)
        except:
            print("Can't process {}".format(ppi))
            traceback.print_exc()
            #exit()

        if os.path.exists(antigen_outply) and os.path.exists(antibody_outply):
            # append to processed if output fiile exists
            processed_ppi.append(ppi)
            continue

    return processed_ppi