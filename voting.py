import os
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


result_path = r'D:/Develop/UBC/Datasets/Run_155_ER/results_classifier_3'
votes = r'D:/Develop/UBC/Datasets/Run_155_ER/labels.csv'


lines = open(votes).readlines()
slides = lines[0][:-1].split(',')[1:]
dict_res = {}
for idx, slide in enumerate(slides):
    dict_res[slide] = {}
    for row in range(1, 14):
        dict_res[slide][str(row)] = lines[row][:-1].split(',')[idx + 1]


true_labels = []
scores = []

for folder in os.listdir(result_path):
    slide_name = folder.split('_')[0]
    if slide_name in dict_res:
        for core in os.listdir(os.path.join(result_path, folder)):
            if os.path.isfile(os.path.join(result_path, folder, core)):
                continue
            core_number = core.split(',')[-1]
            # print(folder)
            data = json.load(open(glob(os.path.join(result_path, folder, core, '*.json'))[0]))
            if core_number not in dict_res[slide_name]:
                continue
            label = dict_res[slide_name][core_number]
            score = data['num_pos'] / data['num_total']
            true_labels.append(1 if label == 'P' else 0)
            scores.append(score)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Annotate the plot with sensitivity and specificity
# for i, threshold in enumerate(thresholds):
#     sensitivity = tpr[i]
#     specificity = 1 - fpr[i]
#     plt.annotate(f'Threshold: {threshold:.2f}\nSensitivity: {sensitivity:.2f}\nSpecificity: {specificity:.2f}',
#                  (fpr[i], tpr[i]),
#                  textcoords="offset points",
#                  xytext=(10, -10),
#                  ha='center',
#                  fontsize=8,
#                  color='blue')

plt.show()

# Print sensitivity and specificity for each threshold
for i, threshold in enumerate(thresholds):
    sensitivity = tpr[i]
    specificity = 1 - fpr[i]
    print(f'Threshold: {threshold:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')