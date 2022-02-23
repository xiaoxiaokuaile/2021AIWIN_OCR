# Learner: 王振强
# Learn Time: 2021/11/30 21:05
import argparse
import os
import json
import numpy as np
import cv2
import random

"""
    将若干个模型的bank预测结果列出并保存到txt文件中,用于对比制作bank地址库
"""

# 需要修正的图片数目
global_check_num = 0

# 判断两个字符串是否完全相同,相同则返回一个,不同则返回两个
def string_same(str1,str2):
    if str1 == str2:
        return  1,(str1)
    else:
        return 2,(str1,str2)


# 获取模型预测结果对比列表
def check_bank_answer(answer1, answer2, save_txt_path):
    global global_check_num

    # json标签列表1
    answer_json_1 = open(answer1, 'r', encoding='utf-8')
    answer_json_1_dict = json.load(answer_json_1)

    # json标签列表2
    answer_json_2 = open(answer2, 'r', encoding='utf-8')
    answer_json_2_dict = json.load(answer_json_2)


    f_train = open(save_txt_path+'check_bank.txt', 'w', encoding='utf-8')

    # 遍历answer
    for image_name, label_text in answer_json_1_dict.items():
        # 获取类型
        pre_name = image_name.split('_')[0]

        str1 = answer_json_1_dict[image_name]['result']
        str2 = answer_json_2_dict[image_name]['result']
        num,str_result = string_same(str1, str2)
        if pre_name == 'bank':
            # 判定是否置信度小于0.95 及 两个answer不一致
            if answer_json_1_dict[image_name]['confidence'] < 0.9 or num==2:
                global_check_num += 1
                if num == 2:
                    check = 'check2'
                else:
                    check = 'check1'
                # 保存为txt
                f_train.write('{}\t{}\t{}\n'.format(image_name,check, str_result))
            else:
                check = '     '
                # 预测准确的直接保存
                f_train.write('{}\t{}\t{}\n'.format(image_name, check, str_result))
        else:
            # 不是bank直接pass
            pass

    print('check number:',global_check_num)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # answer地址
    parser.add_argument('--answer1_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2\6_PPOCR_超轻量_0.98683_0.79672/answer.json')
    parser.add_argument('--answer2_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2\8_修正bank与sum+date_0.19855_\bank修正_0.19855/answer.json')
    # 保存txt地址
    parser.add_argument('--train_txt_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2/train.txt')
    opt = parser.parse_args()


    check_bank_answer(opt.answer1_path, opt.answer2_path, opt.train_txt_path)



'''bash
    python recognizer/tools/Fix_extra_train_data.py
'''























































































