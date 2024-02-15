import os
import numpy as np
from computecoordinates import save_pdb_folder_to_npy
from computeAPBS import computeAPBS_from_pdb
from computeCharges import computeCharges
from computeHydrophobicity import compute_hydrophobicity_scores

def compute_and_save_features(pdb_folder, output_folder):
    # 计算并保存坐标为.npy文件
    save_pdb_folder_to_npy(pdb_folder, output_folder)

    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_folder, filename)
            pdb_basename = os.path.basename(filename)
            chain_name = os.path.splitext(pdb_basename)[0]

            # 尝试加载所有相关的.npy文件
            try:
                shape_index_path = os.path.join(output_folder, chain_name + "_shape_index.npy")
                shape_index = np.load(shape_index_path)

                ddc_path = os.path.join(output_folder, chain_name + "_ddc.npy")
                ddc = np.load(ddc_path)

                dssp_features_path = os.path.join(output_folder, chain_name + "_dssp_features.npy")
                dssp_features = np.load(dssp_features_path)

                # 加载之前保存的坐标.npy文件以计算其他特征
                coordinates_path = os.path.join(output_folder, chain_name + "_coordinates.npy")
                coordinates = np.load(coordinates_path)

                # 使用其他三个函数计算特征
                charges = computeAPBS_from_pdb(coordinates, pdb_path, output_folder)
                hbond = computeCharges(pdb_path)
                hphobicity, _ = compute_hydrophobicity_scores(pdb_path)

                # 将计算得到的特征按照指定顺序合并为一个数组
                features = np.column_stack((shape_index, ddc, hbond, charges, hphobicity, dssp_features))

                # 保存特征为.npy文件
                features_path = os.path.join(output_folder, chain_name + "_input_feat.npy")
                np.save(features_path, features)
                print(f"Saved features to {features_path}")
            except FileNotFoundError as e:
                print(f"Error loading feature files for {chain_name}: {e}")

if __name__ == "__main__":
    pdb_folder = r'/home/wxy/Desktop/se/se3/pdb'
    output_folder = r'/home/wxy/Desktop/se/se3/output'
    compute_and_save_features(pdb_folder, output_folder)