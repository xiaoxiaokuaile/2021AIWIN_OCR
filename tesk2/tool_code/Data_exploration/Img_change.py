# Learner: 王振强
# Learn Time: 2021/11/13 16:54
import cv2
import numpy as np
import random
from warp_mls import WarpMLS


# 提取红色印章
def remove_red_seal(input_img):
    # 分离图片的通道
    blue_c, green_c, red_c = cv2.split(input_img)
    #利用大津法自动选择阈值
    thresh, ret = cv2.threshold(red_c, 0, 255,cv2.THRESH_OTSU)
    #对阈值进行调整
    filter_condition = int(thresh * 0.90)
    #移除红色的印章
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
    # 把图片转回3通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

    return result_img

# 添加噪点
def add_noise(img):
    img_temp = img.copy()
    for i in range(100):  # 添加点噪声
        temp_x = np.random.randint(0, img_temp.shape[0])
        temp_y = np.random.randint(0, img_temp.shape[1])
        img_temp[temp_x][temp_y] = 255
    return img_temp

# 定义缩放resize函数
def resize(image,inter=cv2.INTER_AREA):

    (h, w) = image.shape[:2]
    print(h,w)
    # 缩放比例(1,1.5)
    ratio = random.random()/2 + 1
    width = int(w/ratio)
    print(width)
    # 缩放图像
    resized = cv2.resize(image, (width,h), interpolation=inter)

    # 返回缩放后的图像
    return resized

# 随机调整对比度
def contrast(image):
    alpha = random.random()
    beta = random.random()
    brightness = random.randint(1,100)
    return cv2.addWeighted(image,alpha,image,beta,brightness)

# 随机裁剪
def random_crop(image):
    img_h,img_w,img_d = image.shape
    crop_shape = [int(img_h / 2), int(img_h / 2)]

    # 最少要切一个字的1/4边长
    nh = random.randint(int(crop_shape[0] / 2), crop_shape[0])
    nw = random.randint(int(crop_shape[0] / 2), crop_shape[1])

    if random.random() > 0.5:
        # 高度方向切分
        if random.random() > 0.5:
            image_crop = image[nh:, :]
        else:
            image_crop = image[:img_h - nh, :]
    else:
        # 宽度方向切分
        if random.random() > 0.5:
            image_crop = image[:, nw:]
        else:
            image_crop = image[:, :img_w - nw]

    return image_crop

# 随机添加干扰线
def random_padding(image):
    img_h, img_w, img_d = image.shape
    image_crop = image.copy()
    padding = random.randint(int(img_h/12),int(img_h/6))

    # 随机加干扰横线
    start_index = random.randint(0,img_h-padding)
    image_crop[start_index:start_index+padding,:,0:img_d] = 0

    if random.random() < 0.3:
        # 随机加干扰竖线
        start_index = random.randint(0,img_w-padding)
        image_crop[:,start_index:start_index+padding,0:img_d] = 0

    return image_crop

# 随机拼接
def random_concatenate(image):
    img_h, img_w, img_d = image.shape
    # 随机截取图片切片大小
    padding = random.randint(int(img_h / 12), int(img_h / 2))
    # 截取图片的开始Index
    start_index = random.randint(0,img_h-padding)
    # 用于padding的切片
    image_crop = image[start_index:start_index + padding, :, 0:img_d]

    img_black = np.zeros((img_h + padding,img_w,img_d))
    # print(img_h, padding,img_black.shape)
    # 0.5概率上下随机拼接
    if random.random()<0.5:
        img_black[:padding, :, 0:img_d] = image_crop
        img_black[padding:, :, 0:img_d] = image
    else:
        img_black[:img_h, :, 0:img_d] = image
        img_black[img_h:, :, 0:img_d] = image_crop

    return img_black


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




if __name__ == '__main__':
    image_path = r"F:\JS\2021AIWIN\OCR\tesk2\submit\tool_code\Data_exploration\img_ch/"
    # 读取图片
    input_img = cv2.imread(image_path + 'test.jpg')

    # 随机添加噪声
    # ch_img = add_noise(input_img)
    # 随机缩放
    # ch_img = resize(input_img)
    # 随机调整对比度
    # ch_img = contrast(input_img)
    # 随机裁剪
    # ch_img = random_crop(input_img)
    # 随机添加干扰线
    # ch_img = random_padding(input_img)
    # RGB空间做颜色随机扰动
    # ch_img = PCA_Jittering(input_img)
    # gauss模糊
    # ch_img = apply_gauss_blur(input_img)
    # norm模糊
    # ch_img = apply_norm_blur(input_img)
    # 锐化
    # ch_img = apply_emboss(input_img)
    # 滤波,字体更细
    # ch_img = apply_sharp(input_img)
    # 扭曲
    # ch_img = distort(input_img)
    # 伸展
    # ch_img = stretch(input_img)
    # 透镜
    # ch_img = perspective(input_img)
    # 透镜
    ch_img = random_concatenate(input_img)


    cv2.imshow("remove_seal image", input_img)
    cv2.imshow("add_noise image", ch_img)

    cv2.imwrite(image_path + 'ch_img.jpg', ch_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



































