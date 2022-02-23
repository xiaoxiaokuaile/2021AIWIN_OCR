# Learner: 王振强
# Learn Time: 2021/11/13 16:54
import cv2
import numpy as np
import random

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



if __name__ == '__main__':
    input_img = cv2.imread("./test/img/111.jpg")
    remove_seal = remove_red_seal(input_img)
    add_noise_img = random_padding(input_img)

    cv2.imshow("remove_seal image", input_img)
    cv2.imshow("add_noise image", add_noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



































