# Learner: 王振强
# Learn Time: 2021/11/13 14:08
import Levenshtein
import os
# 计算两个中文字符串的编辑距离

if __name__ == '__main__':
    str1 = "二"
    str2 = "二三四"
    dic = {"img1":[1,2,3],"img2":[2,3],"img3":[1,3],"img4":[3]}
    dis = Levenshtein.distance(str1,str2)
    print(dic[1])
    # str3 = '111.jpg'
    # print(os.path.splitext(str3)[0])