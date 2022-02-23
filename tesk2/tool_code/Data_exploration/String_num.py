# Learner: 王振强
# Learn Time: 2021/11/11 16:39
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import pandas as pd

"""
    1.统计训练集中字符分信息
        最大长度28个字符
        同时统计图片宽高信息,保存到txt文件中留作生成数据使用
    2.读取txt中宽高,字符数目
"""


def parse_map_file(txt_path):
    res = list()
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip().replace('\n', '').replace('\r', ''))
    dic = dict()
    for i in res:
        path, values = i.split('\t')
        dic[path] = values.split(' ')
    return dic

# 绘制列表元素分布直方图
def draw_hist(myList, bin,Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList,bin)  # bins = 50，顺便可以控制bin宽度
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymin, Ymax)
    plt.xlabel(Xlabel)  # 横轴名
    plt.ylabel(Ylabel)  # 纵轴名

    plt.title(Title)
    plt.show()


# 根据H_W_str_sum.txt统计字符个数分布
def draw_num_str(txt_path):
    res = list()
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip().replace('\n', '').replace('\r', ''))

    list_num = []
    for i in res:
        img_name, H , W, num_str = i.split('\t')
        list_num.append(int(num_str))
    # print(sort_num)
    draw_hist(list_num,25, 'string num', 'num str', 'sum', 0, 30, 0, 8000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 图片所在文件目录
    parser.add_argument('--txt_path', type=str,default=r'/aiwin/ocr/submit/dataset/train/label/real_train.txt')
    # 图片所在目录
    parser.add_argument('--image_path', type=str,default=r'/aiwin/ocr/submit/dataset/train/images')
    # 保存txt文件目录
    parser.add_argument('--save_txt_path', type=str,default=r'/aiwin/ocr/submit/dataset')
    # 生成的统计txt文件目录
    parser.add_argument('--out_txt_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\tool_code\Data_exploration/H_W_str_sum.txt')
    opt = parser.parse_args()

    txt_path = opt.txt_path
    image_path = opt.image_path
    save_txt_path = opt.save_txt_path
    out_txt_path = opt.out_txt_path

    # -------------------------- 1.将图片统计信息写入txt文件 ----------------------------
    # image_to_label = parse_map_file(txt_path)
    # images_name = [image_name for image_name, image_label in image_to_label.items()]
    #
    # # 初始化存储字符长度列表
    # list_str = np.zeros(len(images_name))
    #
    # with open(os.path.join(save_txt_path, 'H_W_str_sum.txt'), 'a', encoding='utf-8') as out_file:
    #     for index,image_name in enumerate(images_name):
    #         # 读取图片
    #         src_image = cv2.imread(os.path.join(image_path, image_name))
    #         H = src_image.shape[0]
    #         W = src_image.shape[1]
    #
    #         label_length = len(image_to_label[image_name])
    #         list_str[index] = label_length
    #         # print(image_name,label_length)
    #         # 图片名称, 高度, 宽度, 字符长度
    #         out_file.write('{}\t{}\t{}\t{}\n'.format(image_name, H, W, label_length))
    #
    # print(list_str)
    # print('最大长度:',max(list_str)) # 28
    # draw_hist(list_str,50,'string num','x','y',0,30,0,2000)
    # ---------------------------------------------------------------------------------

    # 绘制字符个数直方图
    draw_num_str(out_txt_path)


'''bash
    python recognizer/tools/String_num.py
'''























