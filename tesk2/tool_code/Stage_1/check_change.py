# Learner: 王振强
# Learn Time: 2021/11/15 14:06
# Learner: 王振强
# Learn Time: 2021/11/11 0:03
# -*- coding=utf-8 -*-
import argparse
import json
import numpy as np
import cv2
import os

"""
    合并生成answer.json准确对照组
"""

global_image_num = 0
# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# 获取任务一真实json文件
# 有误图片路径
def get_real_json(false_image_path,test_real,answer_real,answer_real_save):
    # 从文件夹获取图片并保存图片名称,列表形式保存
    image_filenames = [x for x in os.listdir(false_image_path) if is_image_file(x)]
    # 读取json文件
    test_real_json =  open(test_real, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)
    # 读取需要替换的answer.json
    answer_real_json =  open(answer_real, 'r', encoding='utf-8')
    answer_real_json_dict = json.load(answer_real_json)
    #print(test_real_json_dict)

    # 遍历json文件中的图片键值对
    for image_name in image_filenames:
        _,img_real_name = image_name.split("_",1)
        # print(img1,img2,image_name)
        # 替换对应字符串
        value = test_real_json_dict[image_name]['result']
        if value == ' ':
            answer_real_json_dict[img_real_name]['result'] = value
            print('干扰项')
        else:
            # 更换value
            answer_real_json_dict[img_real_name]['result'] = value
            print(value)

    # 保存替换后的json文件
    json_file = open(answer_real_save, 'w', encoding='utf-8')
    json_file.write(json.dumps(answer_real_json_dict, ensure_ascii=False, indent=4, separators=(',', ':')))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 错误图片路径
    parser.add_argument('--false_image_path', type=str,default='./dataset/test_check/check_false/')
    parser.add_argument('--test_real', type=str,default='./dataset/test_check/label/test_real.json')
    # 需要替换的json
    parser.add_argument('--answer_real', type=str, default=r'./dataset/test_check/answer.json')
    parser.add_argument('--answer_real_save', type=str, default=r'./dataset/test_check/answer_real_have_space.json')
    opt = parser.parse_args()
    # 新数据集
    answer_real = opt.answer_real
    answer_real_save = opt.answer_real_save
    # 原始数据集
    false_image_path = opt.false_image_path
    test_real = opt.test_real

    # 生成标准测试集
    get_real_json(false_image_path,test_real,answer_real,answer_real_save)


'''bash
    python recognizer/tools/check_change.py
'''





















































