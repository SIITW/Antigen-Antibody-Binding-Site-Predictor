import os

def read_file(file_path):
    """读取文件内容并返回每行的列表。"""
    with open(file_path, 'r') as file:
        return file.readlines()

def find_duplicate_chains(chains):
    """找出在文件中出现两次或以上的链，并返回这些链的完整信息。"""
    chain_counts = {}
    for chain in chains:
        identifier = chain.split('|')[0].split('_')[0][1:].lower()
        if identifier not in chain_counts:
            chain_counts[identifier] = 0
        chain_counts[identifier] += 1
    return [chain for chain in chains if chain_counts[chain.split('|')[0].split('_')[0][1:].lower()] >= 2]

def parse_duplicate_chains(duplicate_chains):
    """解析重复链信息，返回标识符和其对应的所有链ID的列表。"""
    parsed_chains = {}
    for chain in duplicate_chains:
        parts = chain.split('|')[0].split('_')
        identifier = parts[0][1:].lower()  # 如 1E6J
        chain_id = '_'.join(parts)  # 如 1E6J_1
        if identifier not in parsed_chains:
            parsed_chains[identifier] = []
        parsed_chains[identifier].append(chain_id)
    return [(identifier, *chain_ids) for identifier, chain_ids in parsed_chains.items()]
def find_longest_chain_sequence(parsed_chains, new_folder_path):
    """找到并返回每个标识符下最长链的序列。"""
    longest_sequences = []
    for group in parsed_chains:
        identifier = group[0]
        chain_ids = group[1:]
        fasta_file_path = os.path.join(new_folder_path, f"{identifier}.fasta")
        print(f"正在查找 {identifier}.fasta ...")

        if os.path.exists(fasta_file_path):
            print(f"找到文件: {fasta_file_path}")
            sequences = read_file(fasta_file_path)
            longest_sequence = ''
            longest_chain_id = ''
            for chain_id in chain_ids:
                print(f"处理链 {chain_id} ...")
                sequence_start = False
                sequence_buffer = ''
                for line in sequences:
                    # 更新了匹配链ID的逻辑，以确保正确识别
                    if line.strip().startswith(f">{chain_id[1:]}") or line.strip().startswith(f">{chain_id}"):
                        sequence_start = True
                        continue
                    if sequence_start:
                        if line.startswith('>'):
                            break
                        sequence_buffer += line.strip()

                if sequence_buffer:
                    print(f"链 {chain_id} 的序列长度为 {len(sequence_buffer)}")
                else:
                    print(f"链 {chain_id} 的序列未找到或为空")

                if len(sequence_buffer) > len(longest_sequence):
                    longest_sequence = sequence_buffer
                    longest_chain_id = chain_id

            if longest_sequence:
                longest_sequences.append((longest_chain_id, longest_sequence))
                print(f"找到最长序列: {longest_chain_id} 长度 {len(longest_sequence)}")
        else:
            print(f"未找到文件: {fasta_file_path}")

    return longest_sequences

def get_full_chain_info(chain_id, all_chains):
    """从所有链中获取指定链ID的完整信息。"""
    for chain in all_chains:
        if chain.startswith(f">{chain_id}|"):
            return chain
    return "信息未找到"

# 读取 other_chains.txt 文件
other_chains_path = r'/home/wxy/Desktop/piston1/sift/other_chains.txt'
other_chains = read_file(other_chains_path)
# 找出重复链
duplicate_chains = find_duplicate_chains(other_chains)

parsed_chains=parse_duplicate_chains(duplicate_chains)
# new 文件夹路径
new_folder_path = r'/home/wxy/Desktop/piston1/sift/old'

# 找到并写入最长的序列
output_file = r'/home/wxy/Desktop/piston1/sift/longest_chains.txt'

# 执行 find_longest_chain_sequence 函数
longest_sequences = find_longest_chain_sequence(parsed_chains, new_folder_path)

# 读取 duplicates_file 中的所有链信息
all_duplicate_chains = read_file(other_chains_path)

# 定义 heavy_chains.txt 文件路径
heavy_chains_path = r'/home/wxy/Desktop/piston1/sift/heavy_chains.txt'

# 读取现有的 heavy_chains.txt 内容
existing_heavy_chains = read_file(heavy_chains_path)

# 查找最长链的完整信息并添加到 heavy_chains.txt
with open(heavy_chains_path, 'w') as file:
    # 首先写入原有内容
    file.writelines(existing_heavy_chains)
    for chain_id, _ in longest_sequences:
        full_chain_info = get_full_chain_info(chain_id[1:], all_duplicate_chains)
        print(f"链 {chain_id} 的完整信息: {full_chain_info}")
        file.write(full_chain_info )