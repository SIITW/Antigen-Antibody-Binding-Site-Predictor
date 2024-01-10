import matplotlib.pyplot as plt


fr = open('./dataset/24.fasta','r')
list_fasta = fr.readlines()
len_list = []

for i in range(len(list_fasta)):
    if list_fasta[i].split()[0][0] != '>':
        len_list.append(str(len(list_fasta[i].split()[0])))

for fold in [0, 1, 2, 3, 4]:
#for fold in [0]:
    fr_se3 = open('./result/se3only_(' + str(fold) + ').txt', 'r')
    fr_esm = open('./result/esmonly_(' + str(fold) + ').txt', 'r')
    list_se3 = fr_se3.readlines()
    list_esm = fr_esm.readlines()
    index = 0
    se3_better = []
    esm_better = []
    for seq in len_list:
        count = 0
        count2 = 0
        for i in range(index,index+int(seq)):
            li_se3 = list_se3[i].split()
            li_esm = list_esm[i].split()
            label = int(li_se3[0])
            score_se3 = float(li_se3[1])
            score_esm = float(li_esm[1])
            # both correct
            if label == 0 and score_se3 < 0.5 and score_esm < 0.5:
                if score_se3 < score_esm:
                    count += 1
                else:
                    count2 += 1
            if label == 1 and score_esm > 0.5 and score_se3 > 0.5:
                if score_se3 > score_esm:
                    count += 1
                else:
                    count2 += 1
        se3_better.append(str(count))
        esm_better.append(str(count2))
        index += int(seq)
    #print(len(se3_better),se3_better)
    #print(len(esm_better),esm_better)
    x = range(1, len(se3_better) + 1)

    plt.plot(x, se3_better, label='SE3 Better')
    plt.plot(x, esm_better, label='ESM Better')

    plt.xlabel('Sequence')
    plt.ylabel('Count')
    plt.title('Comparison of SE3 and ESM Predictions')
    plt.legend()
    plt.savefig('./'+str(fold)+'.jpg')
    plt.clf()