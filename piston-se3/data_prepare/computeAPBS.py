import os
import numpy as np
from subprocess import Popen, PIPE

apbs_bin = r"/home/wxy/Downloads/APBS-3.4.1.Linux/bin/apbs"
pdb2pqr_bin = r"/home/wxy/anaconda3/envs/se31/bin/pdb2pqr"
multivalue_bin = r"/home/wxy/Downloads/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue"
# 库文件所在的目录
lib_path_apbs = r"/home/wxy/Downloads/APBS-3.4.1.Linux/lib"

# 获取当前的LD_LIBRARY_PATH值
current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

# 将新的库路径添加到LD_LIBRARY_PATH中
new_ld_library_path = f"{lib_path_apbs}:{current_ld_library_path}"
os.environ["LD_LIBRARY_PATH"] = new_ld_library_path

def computeAPBS_from_pdb(vertices,pdb_file, output_directory):
    print(pdb_file)
    # 从PDB文件名生成一个独特的基础名
    pdb_basename = os.path.basename(pdb_file)
    tmp_file_base = os.path.splitext(pdb_basename)[0]
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 生成完整路径
    csv_file_path = os.path.join(output_directory, tmp_file_base + ".csv")
    out_file_path = os.path.join(output_directory, tmp_file_base + "_out.csv")

    # 从PDB文件提取坐标
    in_file=os.path.join(output_directory, tmp_file_base + ".in")
    pqr_file = os.path.join(output_directory, tmp_file_base + ".pqr")

    # pdb2pqr处理
    args = [pdb2pqr_bin, "--ff", "PARSE", "--whitespace", "--noopt", "--apbs-input",in_file,pdb_file, pqr_file]
    with Popen(args, stdout=PIPE, stderr=PIPE, cwd=output_directory) as p2:
        stdout, stderr = p2.communicate()
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
    # APBS处理
    args = [apbs_bin, tmp_file_base + ".in"]
    with Popen(args, stdout=PIPE, stderr=PIPE, cwd=output_directory) as p2:
        stdout, stderr = p2.communicate()
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
    # 写入顶点文件
    with open(csv_file_path, "w") as vertfile:
        for vert in vertices:
            vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))

    # multivalue处理
    args = [multivalue_bin, tmp_file_base + ".csv", tmp_file_base + ".pqr.dx", tmp_file_base + "_out.csv"]
    with Popen(args, stdout=PIPE, stderr=PIPE, cwd=output_directory) as p2:
        stdout, stderr = p2.communicate()
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
    # 读取电荷文件
    charges = np.loadtxt(out_file_path, delimiter=',', usecols=[3])

    remove_fn = os.path.join(output_directory, tmp_file_base)
    os.remove(remove_fn+'.csv')
    os.remove(remove_fn+'.pqr.dx')
    os.remove(remove_fn+'.in')
    os.remove(remove_fn+'.pqr')
    os.remove(remove_fn + '.log')
    os.remove(remove_fn+'_out.csv')

    return charges
