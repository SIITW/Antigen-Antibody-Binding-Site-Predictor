import os
import requests


def extract_pdb_ids(directory):
    pdb_ids = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdb"):
            # 使用os.path.splitext()去除扩展名
            name, _ = os.path.splitext(filename)
            pdb_ids.append(name)
    return pdb_ids


def download_fasta(pdb_id, directory):
    # 调整URL格式
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)

    if response.status_code == 200:
        fasta_content = response.text
        # 确保文件被保存在指定目录下
        with open(os.path.join(directory, f"{pdb_id}.fasta"), "w") as file:
            file.write(fasta_content)
        print(f"FASTA file for {pdb_id} has been downloaded successfully.")
    else:
        print(f"Failed to download FASTA file for {pdb_id}. Status code: {response.status_code}")


# 替换为你的train文件夹的路径
directory_path = r"/home/wxy/Desktop/piston1/sift/train"
pdb_ids = extract_pdb_ids(directory_path)
print(pdb_ids)
# 下载每个PDB ID的FASTA文件/home/wxy/Desktop/piston1/train
for pdb_id in pdb_ids:
    download_fasta(pdb_id, directory_path)