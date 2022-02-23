# -*- coding=utf-8 -*-
import argparse
import os
import json
import numpy as np
import cv2

"""
    task2
    将源文件数据合并并保存到新目录下
"""

global_image_num = 0
char_set = set()

def extract_train_data(src_image_root_path, save_image_path, save_txt_path):
    global global_image_num, char_set
    gt_path = os.path.join(src_image_root_path, 'gt.json')
    data = open(gt_path, encoding='utf-8')
    # # 读取源数据集json文件
    str = data.read()
    label_info_dict = json.loads(str)  # json文件最后一个值不能是逗号

    with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
        # 遍历json文件中的图片键值对
        for image_name, label_text in label_info_dict.items():
            # 切分出文件夹名称
            img_dir, _ = image_name.split("_", 1)
            # 图片路径
            img_path = os.path.join(src_image_root_path,img_dir, image_name)

            # 可以解决中文路径无法读取的问题
            src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)

            # label
            for char in label_text:
                char_set.add(char)

            # 切片重命名
            # crop_image_name = '{}.jpg'.format(global_image_num)
            global_image_num += 1
            if global_image_num%100 == 0:
                print(global_image_num)

            save_img_path = os.path.join(save_image_path, image_name)
            cv2.imwrite(save_img_path, src_image)
            out_file.write('{}\t{}\n'.format(image_name, label_text))

    # 字符去重
    for image_name, label_text in label_info_dict.items():
        text = label_text
        text = text.replace('\r', '').replace('\n', '')
        for char in text:
            char_set.add(char)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新数据集地址
    parser.add_argument('--save_train_image_path', type=str,default='/aiwin/ocr/dataset/train/images/')
    parser.add_argument('--save_train_txt_path', type=str,default='/aiwin/ocr/dataset/train/label/')
    # 旧数据集地址
    parser.add_argument('--train_data_path', type=str, default=r'/datasets/ocr/task2')
    opt = parser.parse_args()

    # 新数据集
    save_train_image_path = opt.save_train_image_path
    save_train_txt_path = opt.save_train_txt_path

    # 目录不存在生成目录
    if not os.path.exists(save_train_image_path):
        os.makedirs(save_train_image_path)
    if not os.path.exists(save_train_txt_path):
        os.makedirs(save_train_txt_path)

    # 原始数据集
    train_data_path = opt.train_data_path

    # 合并数据集
    extract_train_data(train_data_path,
                       save_train_image_path,
                       save_train_txt_path)

    print('Image num is {}.'.format(global_image_num))

    char_list = list(char_set)
    char_list.sort()

    # chars.txt 制作字符表
    with open('chars.txt', 'w', encoding='utf-8') as out_file:
        for char in char_list:
            out_file.write('{}\n'.format(char))


'''bash
    python recognizer/make_train_data.py
'''

