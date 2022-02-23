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

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

# ==================================== 模型选择 ==========================================
# 调用改造densenet模型
from recognizer.models.myself_net.crnn_model import CNN_CTC_MODLE
# 调用resnet模型
# from recognizer.models.resnet.crnn_resnet_model import CNN_CTC_MODLE
# 调用densenet模型
# from recognizer.models.densenet.crnn_densenet_model import CNN_CTC_MODLE
# =======================================================================================

from recognizer.tools.config import config
from recognizer.tools.utils import get_chinese_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ModelType(Enum):
    DENSENET_CRNN_TIME_SOFTMAX = 0

# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


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
        print('图片1.5倍!')
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
        print('图片2.0倍!')
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
        print('图片3.0倍!')

    char_list = list()
    # 得到最大值索引,即得到大小280字符索引列表
    pred_text = list(y_pred.argmax(axis=2)[0])
    # 将预测的数字转换为字符
    for index in groupby(pred_text):
        # 若index不为282,空键
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])

    return u''.join(char_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--char_path', type=str, default='./recognizer/tools/chars.txt')
    parser.add_argument('--model_path', type=str,default='./modle_save/CNN_CTC_extra_my_fintune.h5')
    # 最终结果输出路径
    parser.add_argument('--submission_path', type=str, default='answer.json')
    # 测试集图片路径/datasets/ocr-A
    parser.add_argument('--test_image_path', type=str, default='./dataset/test/images')
    opt = parser.parse_args()

    character_map_table = get_chinese_dict(opt.char_path)
    # 图片 shape
    input_shape = (32, 400, 3)
    # 加载模型
    model = load_model(ModelType.DENSENET_CRNN_TIME_SOFTMAX, opt.model_path)
    print('load model done.')

    # 测试集图片路径
    test_image_root_path = opt.test_image_path
    # 得到所有图片名称列表
    image_filenames = [x for x in os.listdir(test_image_root_path) if is_image_file(x)]
    label_info_dict = {}

    # 遍历测试集
    for idx, image_name in enumerate(image_filenames):
        # 读取图片
        src_image = cv2.imread(os.path.join(test_image_root_path, image_name))

        if idx%100 == 0:
            print('process: {:3d}/{:3d}. image: {}'.format(idx + 1, len(image_filenames), image_name))

        # 预测图片标签
        rec_result = predict(src_image, input_shape, model)
        label_info_dict[image_name] = {
            'result': rec_result,
            'confidence': 1
        }

    # 保存文件路径
    save_label_json_file = opt.submission_path
    # 注意编码格式
    with open(save_label_json_file, 'w', encoding='utf-8') as out_file:
        out_file.write(json.dumps(label_info_dict, ensure_ascii=False, indent=4, separators=(',', ':')))


'''bash
    python recognizer/predict.py
'''
