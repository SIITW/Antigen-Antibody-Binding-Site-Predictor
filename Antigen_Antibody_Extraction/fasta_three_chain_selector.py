import os
import shutil

def count_fasta_records(file_path):
    count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    count += 1
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None
    return count

def move_files_with_three_fasta_records(src_directory, dest_directory):
    for foldername, subfolders, filenames in os.walk(src_directory):
        for filename in filenames:
            if filename.endswith('.fasta') or filename.endswith('.fa'):
                file_path = os.path.join(foldername, filename)
                record_count = count_fasta_records(file_path)
                if record_count == 3:
                    shutil.move(file_path, os.path.join(dest_directory, filename))
                    print(f"移动文件: {file_path} -> {os.path.join(dest_directory, filename)}")

src_directory = r'/home/wxy/Desktop/piston1/sift/train'
dest_directory = r'/home/wxy/Desktop/piston1/sift/new'

# 运行函数
move_files_with_three_fasta_records(src_directory, dest_directory)
