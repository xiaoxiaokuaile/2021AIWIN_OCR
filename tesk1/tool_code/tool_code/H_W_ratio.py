# Learner: 王振强
# Learn Time: 2021/9/12 20:07
# -*- coding=utf-8 -*-
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
"""
    统计数据集中宽高比信息
                280    420    560  840   ~
  初始(32,280)    1     1.5     2    3   other
    train       6787   1065   132   16   0
      %         84.8   98.15  99.8  1    1
    test        1697   269    32    2    0
      %         84.85  98.3   99.9  1    1
     总%        84.84  98.18  99.82  1    1
   
   绘制分布图可以发现max在600处,除以1.5网络输入宽度(32,400)最佳
   <400 --- 97.375%
"""

# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# 绘制列表元素分布直方图
def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 50)  # bins = 50，顺便可以控制bin宽度
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymin, Ymax)
    plt.xlabel(Xlabel)  # 横轴名
    plt.ylabel(Ylabel)  # 纵轴名

    plt.title(Title)
    plt.show()

if __name__ == '__main__':
    # 图片所在文件目录
    image_path = 'F:/JS/2021AIWIN/OCR/dataset/test/images'
    # 得到所有图片名称列表
    image_filenames = [x for x in os.listdir(image_path) if is_image_file(x)]

    # 统计宽高比小于17的图片数目
    sum_1 = 0
    # 统计1.5倍以内
    sum_15 = 0
    # 统计宽高比大于17-35的图片数目
    sum_2 = 0
    # 统计宽高比大于35-53的数目
    sum_3 = 0
    # 统计宽高比大于53的数目
    sum_4 = 0

    # 初始化存储字符长度列表
    list_str = np.zeros(len(image_filenames))

    for index, img_name in enumerate(image_filenames):
        if index%1000 == 0:
            print(index)

        # 读取图片
        src_image = cv2.imread(os.path.join(image_path, img_name))
        H = src_image.shape[0]
        W = src_image.shape[1]
        # 保存宽高比数据
        list_str[index] = W/H*32

        if W/H*32 < 280:
            sum_1 = sum_1 + 1
        elif W/H*32 < 400:
            sum_15 = sum_15 + 1
        elif W/H*32 < 600:
            sum_2 = sum_2 + 1
        elif W/H*32 < 800:
            sum_3 = sum_3 + 1
        else:
            sum_4 = sum_4 + 1

    draw_hist(list_str, 'resize width num', 'x', 'y', 0, 800, 0, 1200)

    print(sum_1,sum_15,sum_2,sum_3,sum_4)


'''bash
    python recognizer/tools/from_text_to_label.py
'''