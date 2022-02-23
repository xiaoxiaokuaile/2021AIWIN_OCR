# Learner: 王振强
# Learn Time: 2021/11/12 22:05
import os
import time
from random import choice, randint, randrange
import random
import re
from PIL import Image, ImageDraw, ImageFont
import cv2
import imutils
from warp_mls import WarpMLS
import numpy as np

"""
    生成合成数据集，数据集分布符合原始数据集, 图片宽度、高度、字数
"""

global_image_num =0
# 读取chars.txt生成字符串
def read_char(char_path):
    with open(char_path, 'r', encoding='utf-8') as out_file:
        lines = out_file.readlines()
        char_str = ''
        for index,line in enumerate(lines):
            char_str = char_str + line.strip('\n')

    return char_str


def selectedCharacters(length):
    result = ''.join(choice(characters) for _ in range(length))
    return result

# 字符色,深色
def getColor():
    if random.random()<0.9:
        r = 0
        g = 0
        b = 0
    else:
        r = randint(0, 100)
        g = randint(0, 100)
        b = randint(0, 100)
    return (r, g, b)
# 背景色,浅色
def getColor_back():
    r = randint(200, 255)
    g = randint(200, 255)
    b = randint(200, 255)
    return (r, g, b)

# =============================== 数据增强 ========================================
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

# ===============================================================================

# 输入要生成的图片尺寸,字符个数,背景图像颜色
def main(size=(300, 48), characterNumber=7, bgcolor=(255, 255, 255),f_train=None,font_path=None):
    global global_image_num
    # 创建空白图像和绘图对象
    imageTemp = Image.new('RGB', size, bgcolor)
    draw01 = ImageDraw.Draw(imageTemp)

    # 生成并计算随机字符串的宽度和高度
    text = selectedCharacters(characterNumber)
    #print(text)
    # 每个字体占用尺寸宽度
    widthEachCharater = size[0] // characterNumber
    # 设置字体
    font = ImageFont.truetype(font_path, widthEachCharater-2)
    width, height = draw01.textsize(text, font)
    #print(width,height)

    # 绘制随机字符串中的字符
    startX = 0

    for i in range(characterNumber):
        position = (startX, (size[1] - height) // 2)
        draw01.text(xy=position, text=text[i], font=font, fill=getColor())
        startX += widthEachCharater

    # 对像素位置进行微调，实现扭曲的效果
    imageFinal = Image.new('RGB', size, bgcolor)
    pixelsFinal = imageFinal.load()
    pixelsTemp = imageTemp.load()
    for y in range(size[1]):
        offset = randint(-1, 0)
        for x in range(size[0]):
            newx = x + offset
            if newx >= size[0]:
                newx = size[0] - 1
            elif newx < 0:
                newx = 0
            pixelsFinal[newx, y] = pixelsTemp[x, y]

    # 绘制随机颜色随机位置的干扰像素
    draw02 = ImageDraw.Draw(imageFinal)
    for i in range(int(size[0] * size[1] * 0.07)):
        draw02.point((randrange(0, size[0]), randrange(0, size[1])), fill=getColor())

    crop_image_name = '{}.jpg'.format(global_image_num)
    global_image_num += 1
    if global_image_num % 100 == 0:
        print(global_image_num)

    image = cv2.cvtColor(np.asarray(imageFinal), cv2.COLOR_RGB2BGR)
    # --------------- 空间方面增强 -------------------
    if random.random() < 0.6:
        mode = randint(0, 6)
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
    if random.random() < 0.6:
        mode = randint(0, 7)
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

    # 保存并显示图片
    # imageFinal.save("dataset/extra/images/"+crop_image_name)
    cv2.imwrite("dataset/extra/images/"+crop_image_name, image)
    # 保存标签
    f_train.write('%s\t%s\n' % (crop_image_name, text))



if __name__ == '__main__':

    if not os.path.exists('dataset/extra/images'):
        os.makedirs('dataset/extra/images')
        os.makedirs('dataset/extra/label')

    # 宋体
    font_path_song1 = "recognizer/tools/fonts/全字库宋体_全.ttf"
    font_path_song2 = "recognizer/tools/fonts/新华宋体_全.ttf"
    font_path_song3 = "recognizer/tools/fonts/华文仿宋_全.ttf"
    font_path_song4 = "recognizer/tools/fonts/华文宋体_全.ttf"
    # 手写体
    font_path_shou = "recognizer/tools/fonts/中国式手写风体_全.ttf"
    font_path_shou1 = "recognizer/tools/fonts/小箱子手写体_最像.ttf"
    font_path_shou2 = "recognizer/tools/fonts/手写体2.ttf"
    font_path_shou3 = "recognizer/tools/fonts/上手迎风手写体.ttf"

    # 切片信息统计txt地址
    txt_path = "dataset/H_W_str_sum.txt"
    # 字符集地址
    dictionary_file_path = 'recognizer/tools/chars.txt'
    characters = read_char(dictionary_file_path)
    # print(characters)
    # ------------ 训练集 ---------------
    # 转换后标签txt
    dst_train_file_path = 'dataset/extra/label/real_train.txt'
    # 转换前txt
    src_train_file_path = 'dataset/extra/label/train.txt'
    # ------------ 测试集 ---------------
    # 转换后标签txt
    dst_test_file_path = 'dataset/extra/label/real_test.txt'
    # 转换前txt
    src_test_file_path = 'dataset/extra/label/test.txt'

    f_train = open('dataset/extra/label/train.txt', 'a', encoding='utf-8')

    # 生成数据集
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        # 3W张图片迭代35轮生成105W张图片数据集
        for epoch in range(35):
            for i in lines:
                # imgname,h,w, str_len = i.split('\t')
                imgname, h, w, str_len = re.split(r"[ ]+", i)
                # print(h,w,str_len)
                # 读取原有数据分布,图片大小及字符个数并生成
                rand = random.random()
                if rand<0.2:
                    # 小箱子手写体
                    main((int(w), int(h)), int(str_len), getColor_back(), f_train=f_train,font_path=font_path_shou1)
                elif rand<0.25:
                    # 手写体2
                    main((int(w), int(h)), int(str_len), getColor_back(), f_train=f_train,font_path=font_path_shou2)
                elif rand<0.3:
                    # 上手迎风手写体
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train, font_path=font_path_shou3)
                elif rand<0.5:
                    # 中国式手写风体
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train, font_path=font_path_shou)
                elif rand<0.6:
                    # 华文宋体_全
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train, font_path=font_path_song4)
                elif rand<0.7:
                    # 华文仿宋_全
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train, font_path=font_path_song3)
                elif rand<0.8:
                    # 新华宋体_全
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train, font_path=font_path_song2)
                else:
                    # 全字库宋体_全
                    main((int(w), int(h)), int(str_len), getColor_back(),f_train=f_train,font_path=font_path_song1)

    f_train.close()

    # 给字符标记序号
    char_to_index = dict()
    with open(dictionary_file_path, 'r', encoding='utf-8') as in_file:
        lines = in_file.readlines()
        for index, line in enumerate(lines):
            line = line.strip('\r').strip('\n')
            char_to_index[line] = index

    # 训练集转换
    with open(dst_train_file_path, 'a', encoding='utf-8') as out_file:
        with open(src_train_file_path, 'r', encoding='utf-8') as in_file:
            lines = in_file.readlines()
            for index, line in enumerate(lines):
                line = line.strip('\r').strip('\n')
                line_list = line.split('\t')
                if '#' in line_list[1]:
                    continue
                if line_list[0].split('.')[1] != 'jpg':
                    print(index, line)
                if len(line_list[-1]) <= 0:
                    continue
                out_file.write('{}\t'.format(line_list[0]))
                for char in line_list[-1][:len(line_list[-1]) - 1]:
                    out_file.write('{} '.format(char_to_index[char]))
                out_file.write('{}\n'.format(char_to_index[line_list[-1][-1]]))


"""
    python recognizer/tools/make_extra_data.py
"""


































