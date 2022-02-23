# Learner: 王振强
# Learn Time: 2021/11/11 0:03
# -*- coding=utf-8 -*-
import argparse
import json
import numpy as np
import cv2
import os

"""
    遍历原始测试集,制作带顺序标签的测试集,伪标签
"""

global_image_num = 0
# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def extract_test_data(src_image_root_path,old_answer_path, save_image_path, save_txt_path):
    global global_image_num, char_set
    # 从文件夹获取图片并保存图片名称,列表形式保存
    image_filenames = [x for x in os.listdir(src_image_root_path) if is_image_file(x)]

    # 读取json预测文件
    test_real_json =  open(old_answer_path, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)

    # 保存键值对
    label_info_dict = {}
    # 遍历json文件中的图片键值对
    for image_name in image_filenames:
        # 读取该切片
        img_path = os.path.join(src_image_root_path, image_name)
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        # 生成线下验证test目录
        crop_image_name = '{}_{}'.format(global_image_num,image_name)

        # 替换对应字符串
        value = test_real_json_dict[image_name]['result']
        label_info_dict[crop_image_name] = {
            'result': value,
            'confidence': 1
        }
        # 统计图片个数
        global_image_num += 1
        if global_image_num%100 == 0:
            print(global_image_num)

        # 将图片保存到新目录下
        save_img_path = os.path.join(save_image_path, crop_image_name)
        cv2.imwrite(save_img_path, src_image)

    # 注意编码格式
    with open(save_txt_path+'/answer_check.json', 'w', encoding='utf-8') as out_file:
        out_file.write(json.dumps(label_info_dict, ensure_ascii=False, indent=4, separators=(',', ':')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新数据集地址
    parser.add_argument('--save_test_image_path', type=str,default='./dataset/test_check/images/')
    parser.add_argument('--save_test_txt_path', type=str,default='./dataset/test_check/label')
    # 旧数据集地址
    parser.add_argument('--test_data_path', type=str, default=r'./dataset/test/images/')
    # 模型预测answer地址
    parser.add_argument('--old_answer_path', type=str, default=r'./dataset/test/answer_bank_space.json')
    opt = parser.parse_args()

    # 新数据集
    save_test_image_path = opt.save_test_image_path
    save_test_txt_path = opt.save_test_txt_path

    # 目录不存在生成目录
    if not os.path.exists(save_test_image_path):
        os.makedirs(save_test_image_path)


    # 原始数据集
    test_data_path = opt.test_data_path
    # 原始answer
    old_answer_path = opt.old_answer_path

    # 数据集转换
    extract_test_data(test_data_path,old_answer_path,save_test_image_path,save_test_txt_path)

    print('Image num is {}.'.format(global_image_num))

'''bash
    python tool_code/make_test_data.py
'''



