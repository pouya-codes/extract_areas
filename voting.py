import os, cv2
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from shutil import copy as cp


result_path = r'D:/Develop/UBC/Datasets/TNP_Array/results_classifier_dearray'
result_path_2 = r'D:/Develop/UBC/Datasets/TNP_Array/results_classifier_dearray_2'
votes = r'D:/Develop/UBC/Datasets/TNP_Array/labels.csv'
number_of_cores = 80
out_path = r'D:/Develop/UBC/Datasets/TNP_Array/results_classifier_dearray/error_analysis'

lines = open(votes).readlines()
slides = lines[0][:-1].split(',')[1:]
dict_res = {}
for idx, slide in enumerate(slides):
    dict_res[slide] = {}
    for row in range(1, number_of_cores + 1):
        dict_res[slide][str(row)] = lines[row][:-1].split(',')[idx + 1]


true_labels = []
scores = []
cores = []

for folder in os.listdir(result_path):
    slide_name = folder.split('_')[0]
    if slide_name in dict_res:
        for core in os.listdir(os.path.join(result_path, folder)):
            if os.path.isfile(os.path.join(result_path, folder, core)):
                continue
            core_number = core.split(',')[-1]
            files = glob(os.path.join(result_path, folder, core, '*.json'))
            if len(files) == 0:
                continue
            data = json.load(open(files[0]))
            if core_number not in dict_res[slide_name]:
                continue
            label = dict_res[slide_name][core_number]
            if label == 'U':
                continue
            print(slide_name, core_number, label)
            score = data['num_pos'] / data['num_total']
            true_labels.append(1 if label == 'P' else 0)
            scores.append(score)
            cores.append(f'{slide_name}_{core_number}')

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

# plt.show()
# exit()

# Print sensitivity and specificity for each threshold
# for i, threshold in enumerate(thresholds):
#     sensitivity = tpr[i]
#     specificity = 1 - fpr[i]
#     print(f'Threshold: {threshold:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
# Choose a threshold (e.g., 0.5)
threshold = 0.01

# Calculate predicted labels
predicted_labels = [1 if score >= threshold else 0 for score in scores]

# Identify FP and FN casesW
fp_cases = [(core, true_label, score) for core, true_label, score, pred_label in zip(cores, true_labels, scores, predicted_labels) if true_label == 0 and pred_label == 1]
fn_cases = [(core, true_label, score) for core, true_label, score, pred_label in zip(cores, true_labels, scores, predicted_labels) if true_label == 1 and pred_label == 0]

# Print FP and FN cases
print("False Positive (FP) cases:")
os.makedirs(os.path.join(out_path, 'FP'), exist_ok=True)
for core, true_label, score in fp_cases:
    core_path = os.path.join(result_path, core.split('_')[0] + '_ER', 'QuPath,' + core.split('_')[1])
    core_path_2 = os.path.join(result_path_2, core.split('_')[0] + '_ER', 'QuPath,' + core.split('_')[1])
    overlay = glob(os.path.join(core_path, '*overlaid.png'))[0]
    classifier = glob(os.path.join(core_path, '*classifier.png'))[0]
    original_files = glob(os.path.join(core_path, '*.png'))
    deepliiff = glob(os.path.join(core_path_2, '*overlaid.png'))[0]
    original_file = [f for f in original_files if f != overlay and f != classifier][0]
    
    overlay_img = cv2.imread(overlay)
    classifier_img = cv2.imread(classifier)
    original_img = cv2.imread(original_file)
    deepliif_img = cv2.imread(deepliiff)

    concatenated_img = cv2.hconcat([original_img, classifier_img, deepliif_img, overlay_img])
    
    # print(core_file)
    cv2.imwrite(os.path.join(out_path, 'FP', core + "_" + os.path.basename(original_file)), concatenated_img)
    print(f'Core: {core}, True Label: {true_label}, Score: {score}')


# os.makedirs(os.path.join(out_path, 'FN'), exist_ok=True)
# print("\nFalse Negative (FN) cases:")
for core, true_label, score in fn_cases:

    core_path = os.path.join(result_path, core.split('_')[0] + '_ER', 'QuPath,' + core.split('_')[1])
    core_path_2 = os.path.join(result_path_2, core.split('_')[0] + '_ER', 'QuPath,' + core.split('_')[1])
    overlay = glob(os.path.join(core_path, '*overlaid.png'))[0]
    classifier = glob(os.path.join(core_path, '*classifier.png'))[0]
    deepliiff = glob(os.path.join(core_path_2, '*overlaid.png'))[0]
    original_files = glob(os.path.join(core_path, '*.png'))
    original_file = [f for f in original_files if f != overlay and f != classifier][0]
    
    overlay_img = cv2.imread(overlay)
    classifier_img = cv2.imread(classifier)
    original_img = cv2.imread(original_file)
    deepliif_img = cv2.imread(deepliiff)

    concatenated_img = cv2.hconcat([original_img, classifier_img, deepliif_img, overlay_img])
    
    # print(core_file)
    cv2.imwrite(os.path.join(out_path, 'FN', core + "_" + os.path.basename(original_file)), concatenated_img)
    print(f'Core: {core}, True Label: {true_label}, Score: {score}')