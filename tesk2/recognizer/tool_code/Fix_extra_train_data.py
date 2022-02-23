# Learner: 王振强
# Learn Time: 2021/11/13 22:35
import argparse
import os
import json
import numpy as np
import cv2
import random

"""
    将训练集8000张与extra24000张合并到max_dataset
    生成训练集测试集
"""

global_image_num = 0
char_set = set()

# txt文件转化为字典格式
def parse_map_file(txt_path):
    res = list()
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip().replace('\n', '').replace('\r', ''))
    dic = dict()
    for i in res:
        path, values = i.split('\t')
        dic[path] = values
    return dic

# 图片目录,txt文件目录,  保存图片目录，保存txt文件目录
def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path):
    global global_image_num, char_set

    image_to_label = parse_map_file(src_label_json_file)

    f_train = open(save_txt_path+'train.txt', 'a', encoding='utf-8')
    f_test = open(save_txt_path+'test.txt', 'a', encoding='utf-8')

    # 遍历图片集
    for image_name, label_text in image_to_label.items():
        # 读取该切片
        img_path = os.path.join(src_image_root_path, image_name)
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)

        # 切片重命名
        crop_image_name = '{}.jpg'.format(global_image_num)
        global_image_num += 1
        if global_image_num%100 == 0:
            print(global_image_num)

        save_img_path = os.path.join(save_image_path, crop_image_name)
        # 保存图片
        cv2.imwrite(save_img_path, src_image)

        # 保存为训练集或者测试集
        if random.random()<0.9:
            f_train.write('{}\t{}\n'.format(crop_image_name, label_text))
        else:
            f_test.write('{}\t{}\n'.format(crop_image_name, label_text))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 混合数据集地址
    parser.add_argument('--save_train_image_path', type=str,default='./dataset/max_dataset/images/')
    parser.add_argument('--save_train_txt_path', type=str,default='./dataset/max_dataset/label/')
    # 原始数据根目录
    parser.add_argument('--data_root_path', type=str, default=r'./dataset')


    opt = parser.parse_args()
    # 数据种类
    fix_list = ['train','extra']

    # 新数据集
    save_train_image_path = opt.save_train_image_path
    save_train_txt_path = opt.save_train_txt_path
    # 根目录
    data_root_path = opt.data_root_path

    # 遍历所有种类数据并制作到同一数据集
    for data_name in fix_list:
        train_image_path = data_root_path + '/' + data_name + '/images/'
        train_label_txt_file = data_root_path + '/' + data_name + '/label/train.txt'
        # enhance_num 表示离线数据增强数目
        extract_train_data(train_image_path,
                           train_label_txt_file,
                           save_train_image_path,
                           save_train_txt_path)

    print('Image num is {}.'.format(global_image_num))


'''bash
    python recognizer/tools/Fix_extra_train_data.py
'''






























