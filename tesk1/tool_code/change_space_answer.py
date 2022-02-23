# Learner: 王振强
# Learn Time: 2021/11/17 13:27
import argparse
import json
import numpy as np
import cv2
import os

"""
    1.将answer干扰项置为空对照试验
    
    2.替换两组预测结果的干扰项部分
    
"""

# -------------------- 1.1 将answer干扰项置为空对照试验 -------------------
def answer_to_space(space_list,old_answer_path,new_answer_path):
    # 读取json文件
    test_real_json =  open(old_answer_path, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)
    # 遍历11个干扰项
    for img_name in space_list:
        # 将干扰项预测结果置为空
        test_real_json_dict[img_name]['result'] = ' '

    # 保存替换后的json文件
    json_file = open(new_answer_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(test_real_json_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

# --------------------- 2.替换两组预测结果的干扰项部分 -------------------
def answer_exchange_space(space_list,answer_path_1,answer_path_2,answer_save_path):
    # 读取json文件 --- space部分
    space_json =  open(answer_path_1, 'r', encoding='utf-8')
    space_json_dict = json.load(space_json)

    # 读取json文件 --- 主体部分
    base_json =  open(answer_path_2, 'r', encoding='utf-8')
    base_json_dict = json.load(base_json)

    # 遍历11个干扰项
    for img_name in space_list:
        # 将干扰项预测结果互换
        base_json_dict[img_name]['result'] = space_json_dict[img_name]['result']

    # 保存替换后的json文件
    json_file = open(answer_save_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(base_json_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == '__main__':
    # 干扰项标签列表
    space_list = ['8bb1941d770626de01773cafc1a37e90.jpg',
                  '8bb1941f770626e0017741b6767508c2.jpg',
                  '8bb1942f774ed940017766e5b6681039.jpg',
                  '8bb1942f774ed9400177852222ce4c46.jpg',
                  '8bb1943e774eb21b01778adc439c60bd.jpg',
                  '8bb3942b7657bb8301768289a4634ae4.jpg',
                  '8bb3942b774ed935017784b1886a5b00.jpg',
                  '8bb3942d7706267a017738f24da64839.jpg',
                  '8bb3942d774ed94901778532b78b3f4f.jpg',
                  '8bb39435774ed9380177700034987497.jpg',
                  '8bb3943977062665017728dfa2756fc1.jpg']

    parser = argparse.ArgumentParser()
    # -------------------- 1.将answer干扰项置为空对照试验 -------------------
    # 需要置空的answer路径
    file_path = r'F:\JS\2021AIWIN\OCR\提交过程\tesk1\29_18+15_0.99454'
    parser.add_argument('--old_answer_path', type=str,default=file_path + '/answer.json')
    # 需要保存的answer路径
    parser.add_argument('--new_answer_path', type=str, default=file_path + '/answer_new.json')
    # --------------------- 2.替换两组预测结果的干扰项部分 -------------------
    # 需要替换的answer 1 地址 --- 提取space部分
    answer_path1 = r'F:\JS\2021AIWIN\OCR\提交过程\tesk1\15_DIGIX网络改输入(32,400)_0.98828_0.98698_0.0013\60+60代_space_0.98698'
    # 需要替换的answer 2 地址 --- 提取主体部分
    answer_path2 = r'F:\JS\2021AIWIN\OCR\提交过程\tesk1\18_手工修正标准数据_0.99401_0.99324_0.00077\19_手工修正标准数据_干扰space_0.99324'
    parser.add_argument('--answer_path_1', type=str, default=answer_path1 + '/answer.json')
    parser.add_argument('--answer_path_2', type=str, default=answer_path2 + '/answer.json')
    # 生成的answer保存地址
    answer_out_save = r'F:\JS\2021AIWIN\OCR\提交过程\tesk1\29_18+15_0.99454'
    parser.add_argument('--answer_save_path', type=str, default=answer_out_save + '/answer.json')
    opt = parser.parse_args()

    # -------------------- 1.将answer干扰项置为空对照试验 -------------------
    old_answer_path = opt.old_answer_path
    new_answer_path = opt.new_answer_path
    # 将指定answer置空
    answer_to_space(space_list,old_answer_path,new_answer_path)

    # --------------------- 2.替换两组预测结果的干扰项部分 -------------------
    # answer_path_1 = opt.answer_path_1
    # answer_path_2 = opt.answer_path_2
    # answer_save_path = opt.answer_save_path
    # answer_exchange_space(space_list, answer_path_1, answer_path_2, answer_save_path)

'''bash
    python recognizer/tools/check_space_answer.py
'''



































































