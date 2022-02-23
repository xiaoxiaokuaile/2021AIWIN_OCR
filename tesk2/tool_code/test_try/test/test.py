# Learner: 王振强
# Learn Time: 2021/11/10 21:54
import cv2
import numpy as np
import os

# 查询图片
imgname = '3.jpg'
img_path = r'F:\JS\2021AIWIN\OCR\tesk2\submit\tool_code\tool_code\tool_code\test\bank/'

# img = cv2.imread(r'8bb1941c760a2c1d01762c6bc2e52188.jpg')
img = cv2.imdecode(np.fromfile(img_path+imgname,dtype=np.uint8),-1)
h, w, c = img.shape
print(h, w, c)

# ------------- 压缩图片 --------------
resize_ratio = 0.5
print(int(h*resize_ratio),int(w*resize_ratio))
img_ratio = cv2.resize(img,(int(w*resize_ratio),int(h*resize_ratio)),interpolation=cv2.INTER_AREA)
# --------------图像灰度化-------------
grayimg = cv2.cvtColor(img_ratio, cv2.COLOR_BGR2GRAY)
h_ratio = int(h*resize_ratio)
w_ratio = int(w*resize_ratio)
print(grayimg.shape)

# 保存图片为txt格式
with open(img_path + '/bank_dict.txt', 'a', encoding='utf-8') as out_file:
    out_file.write('{}\n'.format(imgname))
    for i in range(h_ratio):
        for j in range(w_ratio):
            out_file.write('{} '.format(grayimg[i][j]))
        out_file.write('\n'.format(grayimg[i][j]))



# 保存图片到answer文件夹
save_img_path = os.path.join(img_path, 'test1.jpg')
cv2.imwrite(save_img_path, grayimg)

# ------------------- 压缩保存可以压缩一倍 --------------------
cv2.imwrite(save_img_path,grayimg,[cv2.IMWRITE_JPEG_QUALITY,50])
# -----------------------------------
# 原图
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
# 压缩后的图片
cv2.namedWindow('img_resize', cv2.WINDOW_NORMAL)
cv2.imshow('img_resize', img_ratio)
# 灰度图
cv2.namedWindow('img_gray', cv2.WINDOW_NORMAL)
cv2.imshow('img_gray', grayimg)

cv2.waitKey(0)
cv2.destroyAllWindows()

























