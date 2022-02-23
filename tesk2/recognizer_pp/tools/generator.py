# -*- coding=utf-8 -*-
import paddle
from paddle.io import Dataset
import os
import random
import traceback
import time
import cv2
import numpy as np
import imutils
from recognizer_pp.tools.warp_mls import WarpMLS


# 随机裁剪
def random_crop(img):
    img_h, img_w, img_d = img.shape[:]
    crop_shape = [int(img_h / 2), int(img_h / 2)]

    # 最少要切一个字的1/4边长
    nh = random.randint(int(crop_shape[0] / 2), crop_shape[0])
    nw = random.randint(int(crop_shape[0] / 2), crop_shape[1])

    if random.random() > 0.5:
        # 高度方向切分
        if random.random() > 0.5:
            image_crop = img[nh:, :]
        else:
            image_crop = img[:img_h - nh, :]
    else:
        # 宽度方向切分
        if random.random() > 0.5:
            image_crop = img[:, nw:]
        else:
            image_crop = img[:, :img_w - nw]

    return image_crop


# 随机添加干扰线
def random_padding(img):
    img_h, img_w, img_d = img.shape[:]
    image_crop = img.copy()
    padding = random.randint(int(img_h / 12), int(img_h / 6))

    # 随机加干扰横线
    start_index = random.randint(0, img_h - padding)
    image_crop[start_index:start_index + padding, :, 0:img_d] = 0

    if random.random() < 0.3:
        # 随机加干扰竖线
        start_index = random.randint(0, img_w - padding)
        image_crop[:, start_index:start_index + padding, 0:img_d] = 0

    return image_crop


# 定义缩放resize函数
def resize(img, inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    # 缩放比例(1,1.5)
    ratio = random.random() / 2 + 1
    width = int(w / ratio)
    # 缩放图像
    resized = cv2.resize(img, (width, h), interpolation=inter)

    # 返回缩放后的图像
    return resized


# 随机调整对比度
def contrast(img):
    alpha = random.random()
    beta = random.random()
    brightness = random.randint(1, 100)
    resized = cv2.addWeighted(img, alpha, img, beta, brightness)
    return resized


# 2.RGB空间做颜色随机扰动
def PCA_Jittering(img):
    img = np.asanyarray(img, dtype='float32')
    img = img / 255.0
    img_size = img.size // 3  # 转换为单通道
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)  # 转置
    img_cov = np.cov([img1[0], img1[1], img1[2]])  # 协方差矩阵
    lamda, p = np.linalg.eig(img_cov)  # 得到上述协方差矩阵的特征向量和特征值
    # p是协方差矩阵的特征向量
    p = np.transpose(p)  # 转置回去
    # 生成高斯随机数********可以修改
    alpha1 = random.gauss(0, 2)
    alpha2 = random.gauss(0, 2)
    alpha3 = random.gauss(0, 2)
    # lamda是协方差矩阵的特征值
    v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))  # 转置
    # 得到主成分
    add_num = np.dot(p, v)
    # 在原图像的基础上加上主成分
    img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
    # 现在是BGR，要转成RBG再进行保存
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    img2 = img2 * 255.0
    img2 = img2.astype(np.uint8)
    return img2


# 3.gauss模糊
def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [3, 5]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize >= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img


# 4.norm模糊
def apply_norm_blur(img, ks=None):
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img


# 5.锐化
def apply_emboss(img):
    emboss_kernal = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])
    return cv2.filter2D(img, -1, emboss_kernal)


# 6.滤波,字体更细
def apply_sharp(img):
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(img, -1, sharp_kernel)


# 7.噪点增加
def add_noise(img):
    for i in range(100):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


# 8.扭曲
def distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        img_h + np.random.randint(thresh) - half_thresh])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


# 9.伸展
def stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


# 10.透镜
def perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


# 数据增强部分
def data_enhance(image):
    if random.random() < 0.5:
        mode = random.randint(0, 6)
        if mode == 0:
            # 随机裁剪
            image = random_crop(image)  # 因为有水平和垂直切割，所以概率两倍
        elif mode == 1:
            # 随机裁剪
            image = random_crop(image)
        elif mode == 2:
            # 随机加干扰线
            image = random_padding(image)
        elif mode == 2:
            # 随机缩放
            image = resize(image)
        elif mode == 3:
            # 扭曲
            image = distort(image)
        elif mode == 4:
            # 伸展
            image = stretch(image)
        elif mode == 5:
            # 透镜
            image = perspective(image)
        else:
            pass
    else:
        pass
    #  --------------- 色彩方面增强 -------------------
    if random.random() < 0.5:
        mode = random.randint(0, 7)
        if mode == 0:
            # 随机调整对比度
            image == contrast(image)
        elif mode == 1:
            # 色彩抖动
            image = PCA_Jittering(image)
        elif mode == 2:
            # 高斯模糊
            image = apply_gauss_blur(image)
        elif mode == 3:
            # 均值模糊
            image = apply_norm_blur(image)
        elif mode == 4:
            # 锐化
            image = apply_emboss(image)
        elif mode == 5:
            # 滤波
            image = apply_sharp(image)
        elif mode == 6:
            # 噪点增加
            image = add_noise(image)
        else:
            pass
    else:
        pass
    return image


# 图片预处理
def img_process(root_path, image_name, input_shape, is_enhance=True):
    # 32, 400, 3
    input_height, input_width, input_channel = input_shape
    if input_channel == 1:
        image = cv2.imread(os.path.join(root_path, image_name), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(os.path.join(root_path, image_name), cv2.IMREAD_COLOR)

    # ------------ 数据增强 -----------------
    if is_enhance:
        image = data_enhance(image)
    # ---------------------------------------

    # 缩放图片
    scale = image.shape[0] * 1.0 / input_height
    image_width = int(image.shape[1] // scale)
    # 图片缩放到(32,X)
    image = cv2.resize(image, (image_width, input_height))

    # 如果图片宽度小于400则进行填充操作
    if image_width <= input_width:
        new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
        new_image[:] = 255
        new_image[:, :image_width, :] = image
        image = new_image
    else:
        # 若图片宽度大于400则直接resize训练
        image = cv2.resize(image, (input_width, input_height))

    # 转化为CHW
    image = np.transpose(image, (2, 0, 1))
    # 归一化
    image = np.array(image, 'f') / 127.5 - 1.0

    return image


# 单张图片数据加载器
class Generator(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    # 图片地址, real_train.txt地址 , input_shape(32,400,3),是否是训练模式,是否数据增强
    def __init__(self, root_path, input_map_file, input_shape, is_enhance=True):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(Generator, self).__init__()
        # 输入图片大小(32,400,3)
        self.input_shape = input_shape
        self.root_path = root_path
        # 获取图片名称及Label集合
        with open(input_map_file, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.is_enhance = is_enhance

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        # 获取index索引图片名称及label
        image_name, label_temp = self.lines[index].replace('\n', '').split('\t')
        label = label_temp.split(' ')
        # 获取处理后的图片
        img = img_process(self.root_path, image_name, self.input_shape, self.is_enhance)

        label = np.array(label, dtype='int32')

        return img, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.lines)


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[2], reverse=True)
    channel_size = batch[0][0].shape[0]  # 3
    height_size = batch[0][0].shape[1]  # 32
    max_width_length = 400
    # 最后一层FC长度100
    sequence_length = 100
    batch_size = len(batch)  # 64
    # 标签最大长度
    max_label_length = 50
    # 以最大的长度创建0张量 (64, 3, 32, 400)
    inputs = np.zeros((batch_size, channel_size, height_size, max_width_length), dtype='float32')
    # 补齐标签表,大于num_class-1的数视作空
    labels = np.zeros((batch_size, max_label_length), dtype='int32')
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        # 获取图片
        tensor = sample[0]  # (3, 32, 400)
        # 获取标签
        target = sample[1]
        label_length = target.shape[0]  # 标签长度
        # 前面图片已经resize+padding过了
        inputs[x, :, :, :] = tensor[:, :, :]
        labels[x, :label_length] = target[:]
        input_lens.append(sequence_length)
        label_lens.append(len(target))
    input_lens = np.array(input_lens, dtype='int64')
    label_lens = np.array(label_lens, dtype='int64')
    return inputs, labels, input_lens, label_lens



