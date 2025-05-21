import json
from pymatgen.core.structure import Structure
import xlwt
import tqdm
import xlsxwriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
# from pyxtal.symmetry import Group
# from pyxtal import pyxtal
from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifParser
from collections import Counter

'''得到test_json格式文件'''
def get_test_json(atom_predict, wyckoff_predict, atom_label, wyckoff_label):
    test_json = {}

    '''创建test_json所需格式'''
    atoms = set(atom_predict + atom_label)
    for atom in atoms:
        test_json[atom] = {'predict': [], 'label': []}

    '''往test_json里放值'''
    for index in range(len(atom_predict)):
        nums = wyckoff_predict[index][:-1]
        letter = wyckoff_predict[index][-1:]
        for num in range(int(nums)):
            test_json[atom_predict[index]]['predict'].append(letter)
    for index in range(len(atom_label)):
        nums = wyckoff_label[index][:-1]
        letter = wyckoff_label[index][-1:]
        for num in range(int(nums)):
            test_json[atom_label[index]]['label'].append(letter)

    return test_json

'''Wyckoff位置匹配'''
def wyckoff_list_from_test_json(test_json):

    wyckoff_letter_predict = []
    wyckoff_letter_label = []

    for atom in test_json:
        # print(atom)
        predict = test_json[atom]['predict']
        label = test_json[atom]['label']
        # print('predict:', predict)
        # print('label:', label)

        if len(predict) == len(label):
            '''step1 检测同位置预测正确的wyckoff位置'''
            nums = 0
            for index in range(len(predict)):
                if predict[index] == label[index]:
                    wyckoff_letter_predict.append(predict[index])
                    wyckoff_letter_label.append(label[index])
                    predict[index] = ''
                    label[index] = ''
                    nums += 1
            for num in range(nums):
                predict.remove('')
                label.remove('')

            '''step2 检测错位但预测正确的wyckoff位置'''
            nums = 0
            for index in range(len(predict)):
                '''按照predict中字母的顺序查找label中相同wyckoff位置字母'''

                predict_letter = predict[index]
                '''看看label中有没有这个letter'''
                try:
                    label_letter_index = label.index(predict_letter)
                    wyckoff_letter_predict.append(predict_letter)
                    wyckoff_letter_label.append(predict_letter)
                    predict[index] = ''
                    label[label_letter_index] = ''
                    nums += 1
                except:
                    continue

            for num in range(nums):
                predict.remove('')
                label.remove('')

            '''step3 加上剩余不匹配的wyckoff位置'''
            for index in range(len(predict)):
                wyckoff_letter_predict.append(predict[index])
                wyckoff_letter_label.append(label[index])

        elif len(predict) != len(label):
            '''step1 检测同位置预测正确的wyckoff位置'''
            nums = 0
            for index in range(min(len(predict),len(label))):
                if predict[index] == label[index]:
                    wyckoff_letter_predict.append(predict[index])
                    wyckoff_letter_label.append(label[index])
                    predict[index] = ''
                    label[index] = ''
                    nums += 1
            for num in range(nums):
                predict.remove('')
                label.remove('')

            '''step2 检测错位但预测正确的wyckoff位置'''
            nums = 0
            for index in range(min(len(predict),len(label))):
                '''按照predict中字母的顺序查找label中相同wyckoff位置字母'''

                predict_letter = predict[index]
                '''看看label中有没有这个letter'''
                try:
                    label_letter_index = label.index(predict_letter)
                    wyckoff_letter_predict.append(predict_letter)
                    wyckoff_letter_label.append(predict_letter)
                    predict[index] = ''
                    label[label_letter_index] = ''
                    nums += 1
                except:
                    continue

            for num in range(nums):
                predict.remove('')
                label.remove('')

            '''step3 对剩余长度不匹配的wyckoff位置使用None进行补齐'''
            length = abs(len(predict)-len(label))
            if len(predict) > len(label):
                for num in range(length):
                    label.append('None')
            else:
                for num in range(length):
                    predict.append('None')

            '''step4 加上剩余wyckoff位置'''
            for index in range(len(predict)):
                wyckoff_letter_predict.append(predict[index])
                wyckoff_letter_label.append(label[index])

    return wyckoff_letter_predict, wyckoff_letter_label

if __name__ == '__main__':

    atom_predict = ['Ba', 'Nd', 'Co', 'O', 'O', 'O']
    wyckoff_predict = ['1e', '1a', '2r', '2t', '2s', '1c']
    atom_label = ['Ba', 'Nd', 'Co', 'O', 'O', 'O']
    wyckoff_label = ['1h', '1f', '2q', '2s', '2r', '1c']

    test_json = get_test_json(atom_predict, wyckoff_predict, atom_label, wyckoff_label)
    print(test_json)
    wyckoff_letter_predict, wyckoff_letter_label = wyckoff_list_from_test_json(test_json)
    print(wyckoff_letter_predict, wyckoff_letter_label)
