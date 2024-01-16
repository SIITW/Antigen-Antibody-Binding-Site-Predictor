def read_file(file_path):
    """读取文件内容并返回每行的列表。"""
    with open(file_path, 'r') as file:
        return file.readlines()

def write_file(file_path, lines):
    """将行写入文件。"""
    with open(file_path, 'w') as file:
        file.writelines(lines)

def find_unique_chains(heavy_chains, other_chains):
    """找出在 other_chains 中唯一存在的链条，然后在 heavy_chains 中找出前五个字母不存在的链条。"""
    other_prefixes_count = {}
    for line in other_chains:
        prefix = line.split('|')[0][:5]
        if prefix not in other_prefixes_count:
            other_prefixes_count[prefix] = 0
        other_prefixes_count[prefix] += 1

    unique_chains = [line for line in other_chains if other_prefixes_count[line.split('|')[0][:5]] == 1]

    heavy_prefixes = {line.split('|')[0][:5] for line in heavy_chains}

    chains_to_append = [line for line in unique_chains if line.split('|')[0][:5] not in heavy_prefixes]

    return chains_to_append





# 读取文件内容
heavy_chains_path = r'/home/wxy/Desktop/piston1/sift/heavy_chains.txt'
other_chains_path = r'/home/wxy/Desktop/piston1/sift/other_chains.txt'
heavy_chains = read_file(heavy_chains_path)
other_chains = read_file(other_chains_path)

# 找出唯一链并追加到 heavy_chains 中
unique_chains = find_unique_chains(heavy_chains, other_chains)
print(unique_chains)
heavy_chains.extend(unique_chains)

# 将更新后的内容写回文件
write_file(heavy_chains_path, heavy_chains)
