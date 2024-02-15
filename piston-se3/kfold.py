import random

def get_sequence_indices(fasta_file, clusters):
    sequence_indices = []
    frr = open(fasta_file)
    seq = frr.readlines()
    for i in range(len(clusters)):
        count = 1
        for j in range(len(seq)):
            sequence = seq[j].split()[0]
            if sequence[0] == '>':
                if clusters[i] == sequence[1:]:
                    sequence_indices.append(count)
                count += 1
    return sequence_indices


def divide_5fold(cluster_file, fasta_path):
    total_sequences = 1404
    clusters = []
    fr = open(cluster_file,'r')
    list = fr.readlines()
    for line in list:
        cluster = int(line.split()[0])
        clusters.append(cluster)

    fold_total = [[]* n for n in range(5)]
    flag = [0]* 343
    for fold in range(5):
        current_fold = 0
        for i in range(len(flag)):
            if current_fold < total_sequences/5 and flag[i] == 0:
                current_fold += clusters[i]
                flag[i] = 1
                fold_total[fold].append(i)
            elif flag[i] == 1:
                continue
            elif current_fold >= total_sequences/5:
                break
        #print(current_fold)

    seq_total = [[]* n for n in range(5)]
    for fold in range(5):
        index = fold_total[fold]
        for i in range(len(fold_total[fold])):
            seq_total[fold].extend(list[index[i]].split()[2:])

    index_total = [[]* n for n in range(5)]
    train_total = [[]* n for n in range(5)]
    val_total = [[]* n for n in range(5)]
    for fold in range(5):
        sequence = get_sequence_indices(fasta_path, seq_total[fold])
        index_total[fold] = sequence
        #print(len(sequence))
    for fold in range(5):
        #train_total[i].extend(index_total[v])
        #train_total[i].append(index_total[fold])
        val_total[fold] = index_total[fold]
        for i in range(len(index_total[:fold])):
            for j in range(len(index_total[:fold][i])):
                train_total[fold].append(index_total[:fold][i][j])
        for i in range(len(index_total[fold+1:])):
            for j in range(len(index_total[fold+1:][i])):
                train_total[fold].append(index_total[fold+1:][i][j])
    return train_total,val_total

def divide_5fold_bep3(data_size, num_folds):
    """
    将数据划分为指定数量的折叠，并返回折叠的索引
    """
    indices = list(range(data_size))
    fold_size = data_size // num_folds
    extra = data_size % num_folds
    folds = []
    start = 1
    for fold_index in range(num_folds):
        fold_end = start + fold_size
        if fold_index < extra:
            fold_end += 1
        fold_indices = indices[start:fold_end]
        folds.append(fold_indices)
        start = fold_end

    index_total = [[] * n for n in range(5)]
    train_total = [[] * n for n in range(5)]
    val_total = [[] * n for n in range(5)]
    for fold in range(5):
        # train_total[i].extend(index_total[v])
        # train_total[i].append(index_total[fold])
        val_total[fold] = folds[fold]
        for i in range(len(folds[:fold])):
            for j in range(len(folds[:fold][i])):
                train_total[fold].append(folds[:fold][i][j])
        for i in range(len(folds[fold + 1:])):
            for j in range(len(folds[fold + 1:][i])):
                train_total[fold].append(folds[fold + 1:][i][j])

    return train_total,val_total

#path = './Bep3_dataset/trainval_solved/clusterRes_processed.txt'
#fasta = './Bep3_dataset/trainval_solved/trainval_solved.fasta'
#folds = divide_5fold(path,fasta)