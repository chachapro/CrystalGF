import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.symmetry.groups import SpaceGroup
import tqdm
import re
from collections import Counter
import pandas as pd
from get_wyckoff_list import get_test_json
from get_wyckoff_list import wyckoff_list_from_test_json

prediction_list = []
label_list = []
correct_wyckoff = 0

#无法正常提取原子数量和wyckoff位置
error = 0

wyckoff_limit = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

file_name = 'llama_wyckoff_full_form'

def get_atom_wyckoff(string):
    atom_list = []
    wyckoff_list = []
    atom_wyckoff_list = string.split(' ')
    for atom_wyckoff in atom_wyckoff_list:
        atom = atom_wyckoff.split('[')[0]
        wyckoff = atom_wyckoff.split('[')[1].split(']')[0]
        atom_list.append(atom)
        wyckoff_list.append(wyckoff[-1:])
    return atom_list, wyckoff_list

def get_atom_num_wyckoff(string):
    atom_list = []
    wyckoff_list = []
    atom_wyckoff_list = string.split(' ')
    for atom_wyckoff in atom_wyckoff_list:
        atom = atom_wyckoff.split('[')[0]
        wyckoff = atom_wyckoff.split('[')[1].split(']')[0]
        atom_list.append(atom)
        wyckoff_list.append(wyckoff[:])
    return atom_list, wyckoff_list

with open(f'./CrystalGFG/result/wyckoff/{file_name}.jsonl', 'r', encoding='utf-8') as file:
    for lines in tqdm.tqdm(file.readlines()):
        line = json.loads(lines)
        predict = line['predict'].split(':')[1]
        label = line['label'].split(':')[1]

        try:
            atom_predict, wyckoff_predict = get_atom_num_wyckoff(predict)
            atom_label, wyckoff_label = get_atom_num_wyckoff(label)
        except:
            error += 1
            print(lines)
            continue

        try:
            test_json = get_test_json(atom_predict, wyckoff_predict, atom_label, wyckoff_label)
            wyckoff_predict_all, wyckoff_label_all = wyckoff_list_from_test_json(test_json)
        except:
            error += 1
            print(lines)
            continue

        for index in range(len(wyckoff_predict_all)):
            if wyckoff_predict_all[index] == wyckoff_label_all[index]:
                correct_wyckoff += 1
            if wyckoff_predict_all[index] not in wyckoff_limit:
                wyckoff_predict_all[index] = 'other'
            prediction_list.append(wyckoff_predict_all[index])
            if wyckoff_label_all[index] not in wyckoff_limit:
                wyckoff_label_all[index] = 'other'
            label_list.append(wyckoff_label_all[index])


print(f"Accuracy: {correct_wyckoff/len(prediction_list)}")
prediction_count = Counter(prediction_list)
print(prediction_count)
label_count = Counter(label_list)
print(label_count)

#记录标签
prediction_set = set(prediction_list)
unique_labels = set(label_list)

print('error:', error)
print('prediction_set',sorted(prediction_set))
print('len_prediction_set',len(sorted(prediction_set)))
print('unique_labels',sorted(unique_labels))
print('len_unique_labels',len(sorted(unique_labels)))

#统计每个wyckoff位置数量

confusion_matrix = {label: Counter() for label in unique_labels}

# 填充混淆矩阵
for label, prediction in zip(label_list, prediction_list):
    confusion_matrix[label][prediction] += 1

print(confusion_matrix)

label_prediction_ratio = {}
for label, predictions in confusion_matrix.items():
    total_predictions = sum(predictions.values())
    label_prediction_ratio[label] = {pred: (count / total_predictions) for pred, count in predictions.items()}

for label, ratios in label_prediction_ratio.items():
    print(f"Label: {label}")
    for pred, ratio in ratios.items():
        print(f"  Prediction '{pred}': {ratio:.2f}")

confusion_matrix = pd.crosstab(index=prediction_list, columns=label_list, normalize='columns')

confusion_matrix.index.name = 'Predicted'
confusion_matrix.columns.name = 'Actual'

print(confusion_matrix.shape)
print(confusion_matrix)

actual_labels = confusion_matrix.index.tolist()
predicted_labels = confusion_matrix.columns.tolist()

confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

sns.set_style(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap='PuBu', xticklabels=actual_labels , yticklabels=predicted_labels)
# plt.title('Label Heat Map')
# plt.xlabel('Wyckoff Letter')
# plt.ylabel('Validation Predicted')
# plt.xticks(rotation=45, ha='right')
plt.savefig(f'./CrystalGFG/result/wyckoff/{file_name}.png', dpi=400, bbox_inches='tight', pad_inches=0)
plt.show()