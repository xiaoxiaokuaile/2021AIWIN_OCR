# Learner: 王振强
# Learn Time: 2021/11/11 0:03
# -*- coding=utf-8 -*-
import argparse
import json
import numpy as np
import cv2
import os

"""
    将测试集图片合并,并制作json模板文件
"""

global_image_num = 0
# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def extract_test_data(src_image_root_path, save_image_path, save_txt_path):
    global global_image_num, char_set
    # 从文件夹获取图片并保存图片名称,列表形式保存
    image_filenames = [x for x in os.listdir(src_image_root_path) if is_image_file(x)]

    # 遍历json文件中的图片键值对
    for image_name in image_filenames:
        # 读取该切片
        img_path = os.path.join(src_image_root_path, image_name)
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        # 生成线下验证test目录
        crop_image_name = '{}_{}'.format(global_image_num,image_name)
        # 统计图片个数
        global_image_num += 1
        if global_image_num%100 == 0:
            print(global_image_num)

        # 将图片保存到新目录下
        save_img_path = os.path.join(save_image_path, crop_image_name)
        cv2.imwrite(save_img_path, src_image)

# 生成json模板文件
def make_json_file(save_test_image_path,save_test_txt_path):
    # 得到所有图片名称列表
    image_filenames = [x for x in os.listdir(save_test_image_path) if is_image_file(x)]
    # 转化为字典的键
    dict_temp = dict.fromkeys(image_filenames)

    json_file = open(save_test_txt_path+'test_temp.json','w')
    # value值
    json_temp = {"result": "", "confidence": 1}
    for img_name in image_filenames:
        dict_temp[img_name] = json_temp

    json_file.write(json.dumps(dict_temp, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新数据集地址
    parser.add_argument('--save_test_image_path', type=str,default='./dataset/test_check/images/')
    parser.add_argument('--save_test_txt_path', type=str,default='./dataset/test_check/label/')
    # 旧数据集地址
    parser.add_argument('--test_data_path', type=str, default=r'./测试集')
    opt = parser.parse_args()
    # 数据种类
    data_list = ['amount','date']

    # 新数据集
    save_test_image_path = opt.save_test_image_path
    save_test_txt_path = opt.save_test_txt_path

    # 目录不存在生成目录
    if not os.path.exists(save_test_image_path):
        os.makedirs(save_test_image_path)
    if not os.path.exists(save_test_txt_path):
        os.makedirs(save_test_txt_path)

    # 原始数据集
    test_data_path = opt.test_data_path

    # 遍历所有种类数据并制作到同一数据集
    for data_name in data_list:
        # 图片文件夹
        test_image_path = test_data_path + '/' + data_name + '/images/'
        # enhance_num 表示离线数据增强数目
        extract_test_data(test_image_path,save_test_image_path,save_test_txt_path)

    print('Image num is {}.'.format(global_image_num))

    # 根据images文件夹下的图片生成.json文件
    make_json_file(save_test_image_path,save_test_txt_path)

'''bash
    python recognizer/tools/make_test_data.py
'''



