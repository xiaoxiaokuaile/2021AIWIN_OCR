# Learner: 王振强
# Learn Time: 2021/11/13 15:27
import numpy as np
import random
import cv2
"""
    随机裁剪
"""
image_path = r"F:\JS\2021AIWIN\OCR\tesk2\submit\tool_code\test_try\test\img_ch/test.jpg"


def random_crop(image, padding=None):
    img_h = image.shape[0]
    img_w = image.shape[1]
    img_d = image.shape[2]
    crop_shape = [int(img_h/2), int(img_h/2)]
    if padding:
        oshape_h = img_h + 2 * padding
        oshape_w = img_w + 2 * padding
        img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
        img_pad[padding:padding + img_h, padding:padding + img_w, 0:img_d] = image

        # 最少要切一个字的1/4边长
        nh = random.randint(crop_shape[0]/2, crop_shape[0])+padding
        nw = random.randint(crop_shape[0]/2, crop_shape[1])+padding

        if random.random() > 0.5:
            # 高度方向切分
            if random.random()>0.5:
                image_crop = img_pad[nh:, :]
            else:
                image_crop = img_pad[:oshape_h-nh, :]
        else:
            # 宽度方向切分
            if random.random()>0.5:
                image_crop = img_pad[:, nw:]
            else:
                image_crop = img_pad[:, :oshape_w-nw]

        return image_crop
    else:
        # 最少要切一个字的1/4边长
        nh = random.randint(int(crop_shape[0]/2), crop_shape[0])
        nw = random.randint(int(crop_shape[0]/2), crop_shape[1])

        if random.random() > 0.5:
            # 高度方向切分
            if random.random()>0.5:
                image_crop = image[nh:, :]
            else:
                image_crop = image[:img_h-nh, :]
        else:
            # 宽度方向切分
            if random.random()>0.5:
                image_crop = image[:, nw:]
            else:
                image_crop = image[:, :img_w-nw]

        return image_crop


if __name__ == "__main__":
    image_src = cv2.imread(image_path)

    image_dst_crop = random_crop(image_src, padding=None)

    cv2.imshow("oringin image", image_src)
    cv2.imshow("crop image", image_dst_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

























