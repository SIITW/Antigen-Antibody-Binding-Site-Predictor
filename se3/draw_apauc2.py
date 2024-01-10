import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
# 从文本文件中读取真实值和预测值
def read_data(file_path):
    data = np.loadtxt(file_path)
    true_labels = data[:, 0]
    predicted_scores = data[:, 1]
    return true_labels, predicted_scores

# 读取真实值和预测值数据
true_labels_file1, predicted_scores_file1 = read_data('./result/esmonly_(4).txt')  # 替换为第一个文件的路径
true_labels_file2, predicted_scores_file2 = read_data('./result/se3only_(2).txt')  # 替换为第二个文件的路径

# 计算AP和绘制PR曲线
precision_file1, recall_file1, _ = precision_recall_curve(true_labels_file1, predicted_scores_file1)
average_precision_file1 = auc(recall_file1, precision_file1)
fpr1, tpr1, _ = roc_curve(true_labels_file1, predicted_scores_file1)
roc_auc1 = auc(fpr1, tpr1)


precision_file2, recall_file2, _ = precision_recall_curve(true_labels_file2, predicted_scores_file2)
average_precision_file2 = auc(recall_file2, precision_file2)
fpr2, tpr2, _ = roc_curve(true_labels_file2, predicted_scores_file2)
roc_auc2 = auc(fpr2, tpr2)

# 绘制PR曲线
plt.plot(recall_file1, precision_file1, color='red', label='esm AP = %0.4f' % average_precision_file1)
plt.plot(recall_file2, precision_file2, color='red', label='se3 AP = %0.4f' % average_precision_file2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
#plt.show()
plt.savefig('ap.jpg')
plt.clf()

plt.plot(fpr1, tpr1, color='red', label='esm AUC = %0.4f' % roc_auc1)
plt.plot(fpr2, tpr2, color='red', label='se3 AUC = %0.4f' % roc_auc2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.savefig('auc.jpg')
#plt.show()