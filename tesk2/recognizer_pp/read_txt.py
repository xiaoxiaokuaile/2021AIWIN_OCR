# Learner: 王振强
# Learn Time: 2021/11/17 20:22
import os
import numpy as np

"""
    读取txt文件前n行数据并显示1080
"""


if __name__ == '__main__':
    # 需要读取的txt文件路径
    save_txt_path = r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset/H_W_str_sum.txt'
    # save_txt_path = r'F:\JS\2021AIWIN\OCR\tesk2\submit\recognizer\tools/chars.txt'
    with open(save_txt_path, 'r', encoding='utf-8') as out_file:
        lines = out_file.readlines()

        print('start')
        #print(lines)
        for index,line in enumerate(lines):
            print(index)
            # if index<29500:
            #     pass
            # else:
            #     print(line.strip('\n'))
        print('end')












































