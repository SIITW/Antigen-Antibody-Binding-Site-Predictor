import os
import re


def classify_chain(description):
    # 定义重链和轻链的关键词
    heavy_chain_keywords = [
        r'\bHeavy chain\b', r'\bFab-HC\b', r'\bHC\b', r'\bIghg\b',
        r'\bFab_heavy_chain\b', r'\bIGH\b',r'\bVH\b',r'\bheavy-chain\b',r'\bFab H\b',r'\bH-chain\b',r'\bH chain\b',
    ]
    light_chain_keywords = [
        r'\bLight chain\b', r'\bFab-LC\b', r'\bLC\b', r'\bLGH\b', r'\bLGL\b',
        r'\bFab_light_chain\b', r'\bIGL\b', r'\bIGK\b',r'\bVL\b',r"\blight-chain\b",r'\bFab L\b',r'\bL-chain\b',r'\bL chain\b',
    ]

    # 检测重链关键词
    for keyword in heavy_chain_keywords:
        if re.search(keyword, description, re.IGNORECASE):
            return 'heavy'

    # 检测轻链关键词
    for keyword in light_chain_keywords:
        if re.search(keyword, description, re.IGNORECASE):
            return 'light'

    # 如果不匹配任何关键词，默认为其他类型
    return 'other'


def process_fasta(file_path, output_dir):
    # 在指定目录下创建输出文件
    heavy_file_path = os.path.join(output_dir, 'heavy_chains.txt')
    light_file_path = os.path.join(output_dir, 'light_chains.txt')
    other_file_path = os.path.join(output_dir, 'other_chains.txt')

    with open(file_path, 'r') as file:
        records = file.read().split('>')[1:]  # 分割每条FASTA记录

    with open(heavy_file_path, 'a') as heavy_file, open(light_file_path, 'a') as light_file, open(other_file_path,
                                                                                                  'a') as other_file:
        for record in records:
            lines = record.strip().split('\n')
            header = lines[0]  # 获取头部信息
            chain_type = classify_chain(header)

            if chain_type == 'heavy':
                heavy_file.write(f">{header}\n")  # 写入重链文件
            elif chain_type == 'light':
                light_file.write(f">{header}\n")  # 写入轻链文件
            else:
                other_file.write(f">{header}\n")  # 写入其他链文件


# 示例调用
src_file = r'/home/wxy/Desktop/piston1/sift/antibody.txt'
output_dir = r'/home/wxy/Desktop/piston1/sift'

process_fasta(src_file, output_dir)
