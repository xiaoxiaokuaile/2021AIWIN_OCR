import os
import argparse
import numpy as np
import paddle
import cv2
import json
import sys

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)
# ==================================== 模型选择 ==========================================
# 调用自己模型
# from recognizer_pp.models.myself_net.crnn import my_net
# 调用resnet模型
from recognizer_pp.models.resnet.crnn_resnet import Resnet50
# 调用densenet模型
# from recognizer_pp.models.densenet.crnn_densenet import densenet121
# =======================================================================================
from itertools import groupby

# 判定文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# 解码预测输出结果
def ctc_greedy_decoder(probs_seq, vocabulary, blank=0):

    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(vocabulary):
            raise ValueError("probs_seq 尺寸与词汇不匹配")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = paddle.argmax(probs_seq, -1).numpy()
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    index_list = [index for index in index_list if index != blank]
    # 将索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])

# 单张图片预测
def img_predict(img,base_model,input_width, input_height, input_channel):
    img_height, img_width = img.shape[0:2]
    new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
    new_image[:] = 255
    # 若为灰度图，则扩充为彩色图
    if input_channel == 1:
        img = np.expand_dims(img, axis=2)

    new_image[:, :img_width, :] = img
    image_clip = new_image

    # 转化为CHW
    image_clip = np.transpose(image_clip, (2, 0, 1))

    text_image = np.array(image_clip, 'f') / 127.5 - 1.0  # [3,,32,400]
    # batch size维度添加
    text_image = text_image[np.newaxis, :]
    text_image = paddle.to_tensor(text_image, dtype='float32')
    # 图像变为[1,3,32,400]输入预测
    y_pred = base_model(text_image) # [1,100,1082]
    out = paddle.transpose(y_pred, perm=[1, 0, 2]) # [1, 100, 1082]
    out = paddle.nn.functional.softmax(out)[0] # [100,1082]

    return out

# 通过图片预测文本
def predict(image, input_shape, base_model):
    # 32,400,3
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
        1.图片宽高比在400/32以内填充后预测
        2.因训练时候是resize训练的,所以1.5倍以内resize到(32,400)也可以预测
        3.图片两倍400/32时候切两张图且不resize预测
        4.图片大于两倍400/32时候切两张图resize预测
    """
    if image_width <= input_width:
        y_pred = img_predict(image, base_model, input_width, input_height, input_channel)
        y_pred = y_pred[:, :]
    elif image_width <= input_width*1.5:
        # 若图片宽高比不超过1.5倍则resize为(32,280)
        image_res = cv2.resize(image, (input_width, input_height))
        y_pred = img_predict(image_res, base_model, input_width, input_height, input_channel)
        y_pred = y_pred[:, :]
        print('图片1.5倍!')
    elif image_width <= input_width*2:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 2), :]
        y_pred1 = img_predict(imgg1, base_model, input_width, input_height, input_channel)
        y_pred1 = y_pred1[:, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 2):, :]
        y_pred2 = img_predict(imgg2, base_model, input_width, input_height, input_channel)
        y_pred2 = y_pred2[:, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2), axis=0)
        y_pred = paddle.to_tensor(y_pred)
        print('图片2.0倍!')
    else:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 2), :]
        image_clip1 = cv2.resize(imgg1, (input_width, input_height))
        y_pred1 = img_predict(image_clip1, base_model, input_width, input_height, input_channel)
        y_pred1 = y_pred1[:, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 2):, :]
        image_clip2 = cv2.resize(imgg2, (input_width, input_height))
        y_pred2 = img_predict(image_clip2, base_model, input_width, input_height, input_channel)
        y_pred2 = y_pred2[:, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2), axis=0)
        y_pred = paddle.to_tensor(y_pred)
        print('图片3.0倍!')

    # 解码获取识别结果
    out_string = ctc_greedy_decoder(y_pred, vocabulary)

    return out_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 字符表路径
    parser.add_argument('--char_path', type=str, default='./recognizer_pp/tools/chars.txt')
    # 模型路径
    parser.add_argument('--model_path', type=str,default='./modle_save')
    # 最终结果输出路径
    parser.add_argument('--submission_path', type=str, default='answer.json')
    # 测试集图片路径
    parser.add_argument('--test_image_path', type=str, default='./dataset/test/images/')
    opt = parser.parse_args()


    with open(opt.char_path, 'r', encoding='utf-8') as f:
        vocabulary = f.readlines()
    vocabulary = [v.replace('\n', '') for v in vocabulary]
    # print(vocabulary)

    # 类别数目
    n_class = len(vocabulary)
    input_shape = (32,400,3)
    # 加载模型
    model = Resnet50(n_class)
    model.set_state_dict(paddle.load(os.path.join(opt.model_path, 'model_resnet1.pdparams')))
    model.eval()
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

        # 预测图片标签
        rec_result = predict(src_image, input_shape, model)
        label_info_dict[image_name] = {
            'result': rec_result,
            'confidence': 1
        }

        if idx%10 == 0:
            print('process: {:3d}/{:3d}. image: {} result:{}'.format(idx + 1, len(image_filenames), image_name,rec_result))

    # 保存文件路径
    save_label_json_file = opt.submission_path
    # 注意编码格式
    with open(save_label_json_file, 'w', encoding='utf-8') as out_file:
        out_file.write(json.dumps(label_info_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

'''bash
    python recognizer_pp/predict.py
'''










