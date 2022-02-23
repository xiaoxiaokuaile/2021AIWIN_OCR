# Learner: 王振强
# Learn Time: 2021/11/10 21:54
import cv2
import numpy as np

# 查询图片
imgname = '0_8bb3941e7a248697017a2c0628eb02c9.jpg'
img_path = 'F:/JS/2021AIWIN/OCR/dataset/test/images/'

# img = cv2.imread(r'8bb1941c760a2c1d01762c6bc2e52188.jpg')
img = cv2.imdecode(np.fromfile(img_path+imgname,dtype=np.uint8),-1)
h, w, c = img.shape

cv2.namedWindow('add', cv2.WINDOW_NORMAL)
cv2.imshow('add', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

























