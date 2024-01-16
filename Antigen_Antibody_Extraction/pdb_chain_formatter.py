def process_line(line):
    # 分割原始行，提取需要的部分
    parts = line.split(',')
    pdb_id = parts[0].split(' ')[1]  # 例如，提取 "5BV7_C"
    closest_chain = parts[1].split(' ')[-1].strip()  # 提取 "B"

    # 重构字符串
    pdb_id_base = pdb_id.split('_')[0]  # 例如，提取 "5BV7"
    return f"{pdb_id_base}_{closest_chain},{pdb_id}"

# 读取和处理文件
input_file_path = r'/home/wxy/Desktop/piston1/sift/1.txt'  # 替换为实际的输入文件路径
output_file_path = r'/home/wxy/Desktop/piston1/sift/result.txt'  # 替换为实际的输出文件路径

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        new_line = process_line(line)
        outfile.write(new_line + '\n')

print("文件处理完成，输出保存至:", output_file_path)

