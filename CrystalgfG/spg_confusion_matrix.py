import json
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.symmetry.groups import SpaceGroup
import tqdm

spacegroup_limit = ['P1', 'P-1', 'P12_1/m1', 'C12/m1', 'P12_1/c1', 'P2_1/n2_1/m2_1/a', 'C2/m2/c2_1/m', 'P4/m2/m2/m', 'P4/n2_1/m2/m', 'I4/m2/m2/m', 'P3m1', 'P-32/m1', 'R-32/m', 'P-6m2', 'P-62m', 'P6/m2/m2/m', 'P6_3/m2/m2/c', 'P4/m-32/m', 'F4/m-32/m']
int_spg_limit = []
print(len(spacegroup_limit))

for sym in spacegroup_limit:
    int_spg = SpaceGroup(sym).int_number
    int_spg_limit.append(str(int_spg))

print(int_spg_limit)

num = 0
label_list = []
prediction_list = []
max_sample = 3451
file_name = 'llama_spg_full_gap'
prop = 'bandgap' #form bandgap

with open (f"./CrystalGFG/result/{prop}/{file_name}.jsonl", 'r', encoding='utf-8') as file:

    for lines in tqdm.tqdm(file.readlines()):
        line = json.loads(lines)
        # line = json.loads(lines[:-2])
        prediction = line['predict']
        label = line['label']

        '''mid'''
        prediction_start_element_index = prediction.find('空间群类型是') + 6
        label_start_element_index = label.find('空间群类型是') + 6
        # 得到预测值
        prediction_element = prediction[prediction_start_element_index:-1]
        # 得到标签
        label_element = label[label_start_element_index:-1] #qwen -2 其余 -1
        print(f"predict:{prediction_element} ,label:{label_element}")

        #统计other值
        if prediction_element not in int_spg_limit:
            prediction_element = 'other'

        if label_element not in int_spg_limit:
            label_element = 'other'

        if prediction_element == label_element:
            num += 1
        prediction_list.append(prediction_element)
        label_list.append(label_element)

print(f"Accuracy: {num/max_sample}")

unique_labels = list(set(label_list))

confusion_matrix = {label: Counter() for label in unique_labels}

# 填充混淆矩阵
for label, prediction in zip(label_list, prediction_list):
    confusion_matrix[label][prediction] += 1

# print(confusion_matrix['Amm2'])
# # 计算每个标签对应的所有预测结果的占比
label_prediction_ratio = {}
for label, predictions in confusion_matrix.items():
    total_predictions = sum(predictions.values())
    label_prediction_ratio[label] = {pred: (count / total_predictions) for pred, count in predictions.items()}
#
# # 打印结果
for label, ratios in label_prediction_ratio.items():
    print(f"Label: {label}")
    for pred, ratio in ratios.items():
        print(f"  Prediction '{pred}': {ratio:.2f}")

# 创建一个混淆矩阵
confusion_matrix = pd.crosstab(index=prediction_list, columns=label_list, normalize='columns')

# 将索引名称设置为 'Actual'
confusion_matrix.index.name = 'Actual'

# 将列名称设置为 'Predicted'
confusion_matrix.columns.name = 'Predicted'

# 打印百分比分布矩阵
print(confusion_matrix.shape)
print(confusion_matrix)

actual_labels_sym = []
predicted_labels_sym = []
actual_labels = confusion_matrix.index.tolist()
predicted_labels = confusion_matrix.columns.tolist()

for int1,int2 in zip(actual_labels, predicted_labels):
    if int1 != 'other':
        actual_labels_sym.append(SpaceGroup.from_int_number(int(int1)).full_symbol)
    if int2 != 'other':
        predicted_labels_sym.append(SpaceGroup.from_int_number(int(int2)).full_symbol)

actual_labels_sym.append('other')
# print(actual_labels_sym)
actual_labels_sym = ['P1', 'P2_1/m', 'C2/m', 'P4/mmm', 'P4/nmm', 'I4/mmm', 'P2_1/c', 'P3m1', 'P-3m1', 'R-3m', 'P-6m2', 'P-62m', 'P6/mmm', 'P6_3/mmc', 'P-1', 'Pm-3m', 'Fm-3m', 'Pnma', 'Cmcm', 'other']
predicted_labels_sym.append('other')
predicted_labels_sym = ['P1', 'P2_1/m', 'C2/m', 'P4/mmm', 'P4/nmm', 'I4/mmm', 'P2_1/c', 'P3m1', 'P-3m1', 'R-3m', 'P-6m2', 'P-62m', 'P6/mmm', 'P6_3/mmc', 'P-1', 'Pm-3m', 'Fm-3m', 'Pnma', 'Cmcm', 'other']

sns.set_style(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap='OrRd', xticklabels=actual_labels_sym , yticklabels=predicted_labels_sym )
# plt.title('Label Heat Map')
# plt.xlabel('Space Group Symbol')
# plt.ylabel('Validation Predicted')
plt.xticks(rotation=45, ha='right')
plt.savefig(f'./CrystalGFG/result/{prop}/{file_name}.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
plt.show()