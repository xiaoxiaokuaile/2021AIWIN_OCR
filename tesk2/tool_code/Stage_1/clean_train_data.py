# Learner: 王振强
# Learn Time: 2021/11/13 13:05
# -*- coding=utf-8 -*-
import argparse
import os
import json
import cv2
import sys
import math
import numpy as np
from itertools import groupby
from enum import Enum
# 计算编辑距离
import Levenshtein
"""
    清洗训练集中的脏数据，训练集中预测标签与真实标签完全对应不上的剔除出来
    取出模型拟合不充分的数据
"""

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

# ==================================== 模型选择 ==========================================
# 调用改造densenet模型
from recognizer.models.myself_net.crnn_model import CNN_CTC_MODLE
# 调用resnet模型
# from recognizer.models.resnet.crnn_resnet_model import CNN_CTC_MODLE
# 调用resnet模型
#from recognizer.models.densenet.crnn_densenet_model import CNN_CTC_MODLE  # 上一级目录import
# =======================================================================================

from recognizer.tools.config import config
from recognizer.tools.utils import get_chinese_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ModelType(Enum):
    DENSENET_CRNN_TIME_SOFTMAX = 0


def load_model(model_type, weight):
    if model_type == ModelType.DENSENET_CRNN_TIME_SOFTMAX:
        base_model, _ = CNN_CTC_MODLE()
        base_model.load_weights(weight)
    else:
        raise ValueError('parameter model_type error.')
    return base_model

def img_predict(img,base_model,input_width, input_height, input_channel):
    img_height, img_width = img.shape[0:2]
    new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
    new_image[:] = 255
    # 若为灰度图，则扩充为彩色图
    if input_channel == 1:
        img = np.expand_dims(img, axis=2)
    new_image[:, :img_width, :] = img
    image_clip = new_image

    text_image = np.array(image_clip, 'f') / 127.5 - 1.0
    text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    y_pred = base_model.predict(text_image)

    return y_pred

# 通过图片预测文本
def predict(image, input_shape, base_model, image_resize=None):
    # 32,280,3
    input_height, input_width, input_channel = input_shape
    # 缩放比例
    scale = image.shape[0] * 1.0 / input_height

    # 判定图片是否为平面而不是一条线,若不是图片返回空
    if scale==0:
        scale = -1
    image_width = int(image.shape[1] // scale)
    if image_width <= 0:
        return ''

    # resize大小
    image = cv2.resize(image, (image_width, input_height))
    image_height, image_width = image.shape[0:2]

    """
    图片预测识别:
        1.图片宽高比在280/32以内填充后预测
        2.因训练时候是resize训练的,所以1.5倍以内resize到(32,280)也可以预测
        3.图片两倍280/32时候切两张图且不resize预测
        4.图片大于两倍280/32时候切两张图resize预测
    """

    if image_width <= input_width:
        y_pred = img_predict(image, base_model, input_width, input_height, input_channel)
        y_pred = y_pred[:, :, :]
    elif image_width <= input_width*1.5:
        # 若图片宽高比不超过1.5倍则resize为(32,280)
        image_res = cv2.resize(image, (input_width, input_height))
        text_image = np.array(image_res, 'f') / 127.5 - 1.0
        text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
        y_pred = base_model.predict(text_image)
        y_pred = y_pred[:, :, :]
        #print('图片1.5倍!')
    elif image_width <= input_width*2:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 2), :]
        y_pred1 = img_predict(imgg1, base_model, input_width, input_height, input_channel)
        y_pred1 = y_pred1[:, :, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 2):, :]
        y_pred2 = img_predict(imgg2, base_model, input_width, input_height, input_channel)
        y_pred2 = y_pred2[:, :, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2), axis=1)
        #print('图片2.0倍!')
    else:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 2), :]
        image_clip1 = cv2.resize(imgg1, (input_width, input_height))
        text_image1 = np.array(image_clip1, 'f') / 127.5 - 1.0
        text_image1 = np.reshape(text_image1, [1, input_height, input_width, input_channel])
        y_pred1 = base_model.predict(text_image1)
        y_pred1 = y_pred1[:, :, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 2):, :]
        image_clip2 = cv2.resize(imgg2, (input_width, input_height))
        text_image2 = np.array(image_clip2, 'f') / 127.5 - 1.0
        text_image2 = np.reshape(text_image2, [1, input_height, input_width, input_channel])
        y_pred2 = base_model.predict(text_image2)
        y_pred2 = y_pred2[:, :, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2), axis=1)
        #print('图片3.0倍!')

    char_list = list()
    # 得到最大值索引,即得到大小280字符索引列表
    pred_text = list(y_pred.argmax(axis=2)[0])
    # 将预测的数字转换为字符
    for index in groupby(pred_text):
        # 若index不为282,空键
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])

    return u''.join(char_list)

# 读取并分割txt文件
def parse_map_file(txt_path):
    res = list()
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip().replace('\n', '').replace('\r', ''))
    dic = dict()
    for i in res:
        path, values = i.split('\t')
        # dic[path] = values.split(' ')  # 值以空格分隔的
        dic[path] = values
    return dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--char_path', type=str, default='./recognizer/tools/chars.txt')
    parser.add_argument('--model_path', type=str,default='./modle_save/CNN_CTC_extra_my_fintune.h5')
    # 训练集图片路径
    parser.add_argument('--train_image_path', type=str, default='./dataset/train/images/')
    # 训练集真实标签路径
    parser.add_argument('--train_label_path', type=str,default='./dataset/train/label/train.txt')

    # 最终结果图片路径
    parser.add_argument('--clean_image_path', type=str, default='./dataset/train_clean/images/')
    # 最终结果标签路径
    parser.add_argument('--clean_label_path', type=str, default='./dataset/train_clean/label')

    opt = parser.parse_args()

    character_map_table = get_chinese_dict(opt.char_path)
    # 图片 shape
    input_shape = (32, 400, 3)
    # 加载模型
    model = load_model(ModelType.DENSENET_CRNN_TIME_SOFTMAX, opt.model_path)
    print('load model done.')

    # 训练集
    train_labe_file = opt.train_label_path
    train_image_path = opt.train_image_path
    # clean结果
    clean_train_label_path = opt.clean_label_path
    clean_train_images_path = opt.clean_image_path

    # 生成目录
    if not os.path.exists(clean_train_label_path):
        os.makedirs(clean_train_label_path)
    if not os.path.exists(clean_train_images_path):
        os.makedirs(clean_train_images_path)

    # {'1002.jpg': '柒仟元整', '1003.jpg': '壹仟元整', ...}
    image_to_label = parse_map_file(train_labe_file)

    idx = 0
    with open(os.path.join(clean_train_label_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
        # 遍历训练集8000张图片
        for image_name,image_label in image_to_label.items():
            # 读取图片
            src_image = cv2.imread(os.path.join(train_image_path,image_name))

            if idx%100 == 0:
                print('process: {:3d}/{:3d}. image: {}'.format(idx + 1, len(image_to_label), image_name))
            idx += 1

            # 预测图片标签
            rec_result = predict(src_image, input_shape, model)
            # print(rec_result)
            # 计算预测结果与真实值编辑距离
            dis_str = Levenshtein.distance(image_label, rec_result)

            if dis_str>0:
                print(dis_str)
                # 图片预测label,以预测结果命名图片
                crop_image_name = '{}.jpg'.format(os.path.splitext(image_name)[0]+'_'+str(dis_str)+'_'+rec_result)
                save_img_path = os.path.join(clean_train_images_path, crop_image_name)
                # 中文路径图片保存
                cv2.imencode('.jpg',src_image)[1].tofile(save_img_path)
                out_file.write('{}\t{}\n'.format(image_name, dis_str))


'''bash
    python recognizer/clean_train_data.py
'''























