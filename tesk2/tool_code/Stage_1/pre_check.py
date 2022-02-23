# Learner: 王振强
# Learn Time: 2021/11/27 16:03
# -*- coding=utf-8 -*-
import argparse
import json
import numpy as np
import cv2
import os

"""
    1.将模型预测的answer.json 中image name 置换为 index_imagename, 便于核对改进模型
    
"""

global_image_num = 0

# 获取任务一真实json文件
# 有误图片路径
def get_real_json(test_real,answer_real):

    # 读取json文件
    test_real_json =  open(test_real, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)

    label_info_dict = {}
    # 遍历json文件中的图片键值对
    for image_name,info in test_real_json_dict.items():
        _,img_real_name = image_name.split("_",1)
        # 替换对应字符串
        value = test_real_json_dict[image_name]['result']
        label_info_dict[img_real_name] = {
            'result': value,
            'confidence': 1
        }

    # 注意编码格式
    with open(answer_real, 'w', encoding='utf-8') as out_file:
        out_file.write(json.dumps(label_info_dict, ensure_ascii=False, indent=4, separators=(',', ':')))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_real', type=str,default='./dataset/test_check/label/answer_check.json')
    # 修改后的json
    parser.add_argument('--answer_real', type=str, default=r'./dataset/test_check/answer.json')
    opt = parser.parse_args()
    # 新数据集
    answer_real = opt.answer_real
    # 原始数据集
    test_real = opt.test_real

    # 生成标准测试集
    get_real_json(test_real,answer_real)


'''bash
    python recognizer/tools/pre_check.py
'''





















































