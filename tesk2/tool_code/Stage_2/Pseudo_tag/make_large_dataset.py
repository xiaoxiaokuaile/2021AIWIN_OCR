# Learner: 王振强
# Learn Time: 2021/11/30 19:28
import argparse
import os
import json
import numpy as np
import cv2
import random

"""
    合并tesk2的训练集测试集,为一个新的数据集
"""

global_bank_num = 0
global_day_num = 0
global_year_num = 0
global_month_num = 0
global_sum_num = 0
global_image_num = 0

# 读取训练集label
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
def extract_train_data(train_image_path, train_label_file, test_image_path, test_label_file, save_image_path, save_txt_path):
    global global_image_num, global_bank_num,global_day_num, global_year_num,global_month_num,global_sum_num, crop_image_name

    # 读取训练集标签列表
    image_to_label = parse_map_file(train_label_file)
    # 读取测试集json标签列表
    test_real_json = open(test_label_file, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)

    # 目录不存在生成目录
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)

    f_train = open(save_txt_path+'train.txt', 'w', encoding='utf-8')

    # 遍历训练集
    for image_name, label_text in image_to_label.items():
        # 获取类型
        pre_name = image_name.split('_')[0]
        # 读取该切片
        img_path = os.path.join(train_image_path, image_name)
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)

        if pre_name == 'bank':
            # 切片重命名
            crop_image_name = 'bank_{}.jpg'.format(global_bank_num)
            # bank+1
            global_bank_num += 1
            # global+1
            global_image_num += 1
        elif pre_name == 'day':
            # 切片重命名
            crop_image_name = 'day_{}.jpg'.format(global_day_num)
            # bank+1
            global_day_num += 1
            # global+1
            global_image_num += 1
        elif pre_name == 'month':
            # 切片重命名
            crop_image_name = 'month_{}.jpg'.format(global_month_num)
            # bank+1
            global_month_num += 1
            # global+1
            global_image_num += 1
        elif pre_name == 'year':
            # 切片重命名
            crop_image_name = 'year_{}.jpg'.format(global_year_num)
            # bank+1
            global_year_num += 1
            # global+1
            global_image_num += 1
        elif pre_name == 'sum':
            # 切片重命名
            crop_image_name = 'sum_{}.jpg'.format(global_sum_num)
            # bank+1
            global_sum_num += 1
            # global+1
            global_image_num += 1


        if global_image_num%100 == 0:
            print(global_image_num)

        # 保存图片
        save_img_path = os.path.join(save_image_path, crop_image_name)
        # 保存图片
        cv2.imwrite(save_img_path, src_image)

        # 保存为训练集或者测试集
        f_train.write('{}\t{}\n'.format(crop_image_name, label_text))

    # 遍历测试集
    for image_name, label_text in test_real_json_dict.items():
        # 获取该图片标签名
        label = test_real_json_dict[image_name]['result']
        # 获取类型
        pre_name = image_name.split('_')[0]
        # 读取该切片
        img_path = os.path.join(test_image_path, image_name)
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)

        if pre_name == 'bank':
            # 切片重命名
            crop_image_name = 'bank_{}.jpg'.format(global_bank_num)
            # bank+1
            global_bank_num += 1
            # global+1
            global_image_num += 1
            # 保存图片
            save_img_path = os.path.join(save_image_path, crop_image_name)
            # 保存图片
            cv2.imwrite(save_img_path, src_image)

            # 保存为训练集或者测试集
            f_train.write('{}\t{}\n'.format(crop_image_name, label))
        # elif pre_name == 'day':
        #     # 切片重命名
        #     crop_image_name = 'day_{}.jpg'.format(global_day_num)
        #     # bank+1
        #     global_day_num += 1
        #     # global+1
        #     global_image_num += 1
        # elif pre_name == 'month':
        #     # 切片重命名
        #     crop_image_name = 'month_{}.jpg'.format(global_month_num)
        #     # bank+1
        #     global_month_num += 1
        #     # global+1
        #     global_image_num += 1
        # elif pre_name == 'year':
        #     # 切片重命名
        #     crop_image_name = 'year_{}.jpg'.format(global_year_num)
        #     # bank+1
        #     global_year_num += 1
        #     # global+1
        #     global_image_num += 1
        # elif pre_name == 'sum':
        #     # 切片重命名
        #     crop_image_name = 'sum_{}.jpg'.format(global_sum_num)
        #     # bank+1
        #     global_sum_num += 1
        #     # global+1
        #     global_image_num += 1
        else:
            pass


        if global_image_num%100 == 0:
            print(global_image_num)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 训练集数据集地址
    parser.add_argument('--train_image_path', type=str,default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\train\images')
    parser.add_argument('--train_txt_path', type=str,default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\train\label/train.txt')
    # 测试集数据集地址
    parser.add_argument('--test_image_path', type=str,default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\testB\images')
    parser.add_argument('--test_json_path', type=str,default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2B\1_resnet_train_bank_\纠错_bank/answer.json')
    # 保存
    parser.add_argument('--save_image_path', type=str, default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\train_bank_dataset\images')
    parser.add_argument('--save_txt_path', type=str, default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\train_bank_dataset\label/')

    opt = parser.parse_args()


    extract_train_data(opt.train_image_path,
                       opt.train_txt_path,
                       opt.test_image_path,
                       opt.test_json_path,
                       opt.save_image_path,
                       opt.save_txt_path)

    print('Image num is {}.'.format(global_image_num))


'''bash
    python recognizer/tools/Fix_extra_train_data.py
'''























































































