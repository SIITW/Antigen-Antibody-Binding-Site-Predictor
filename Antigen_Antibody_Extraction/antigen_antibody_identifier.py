import os
import re
def is_antibody_chain(description):
    # 抗体相关关键词，不区分大小写
    antibody_keywords = [
        r'\bantibody\b', r'\bFv\b', r'\bHeavy chain\b', r'\bLight chain\b',
        r'\bFab\b', r'\bVHH\b', r'\bIMMUNOGLOBULIN\b', r'\bIG\b', r'\bFAB\b',
        r'\bFV\b', r'\bIghg\b', r'\bIc\b', r'\bhc\b', r'\bVH\b', r'\bVL\b',
        r'\bLGH\b', r'\bLGL\b', r'\bIGG\b', r'\bTP7 MAB\b',r'\bFab_heavy_chain\b',r'\bFab_light_chain\b'
    ]
    # 生成正则表达式，用于搜索关键词
    pattern = re.compile('|'.join(antibody_keywords), re.IGNORECASE)
    return pattern.search(description) is not None
def process_fasta(file_path, antibody_file_path, antigen_file_path):
    with open(file_path, 'r') as file:
        records = file.read().split('>')[1:]  # 分割每条FASTA记录

    with open(antibody_file_path, 'a') as antibody_file, open(antigen_file_path, 'a') as antigen_file:
        for record in records:
            lines = record.strip().split('\n')
            header = lines[0]  # 获取头部信息
            if is_antibody_chain(header):
                antibody_file.write(f">{header}\n")  # 写入抗体文件
            else:
                antigen_file.write(f">{header}\n")  # 写入抗原文件

def main(src_directory, output_directory):
    antibody_file_path = os.path.join(output_directory, 'antibody.txt')
    antigen_file_path = os.path.join(output_directory, 'antigen.txt')

    for foldername, subfolders, filenames in os.walk(src_directory):
        for filename in filenames:
            if filename.endswith('.fasta'):
                file_path = os.path.join(foldername, filename)
                process_fasta(file_path, antibody_file_path, antigen_file_path)

src_directory = r'/home/wxy/Desktop/piston1/sift/old'
output_directory = r'/home/wxy/Desktop/piston1/sift'

main(src_directory, output_directory)

