# Learner: 王振强
# Learn Time: 2021/11/15 15:12
import argparse
import json
import os
# 计算编辑距离
import Levenshtein

"""
    获取线下分数
"""
# 获取任务一真实json文件
# 有误图片路径
def get_real_json(answer,answer_real):
    # 读取手动修正的文件
    answer_real_json =  open(answer_real, 'r', encoding='utf-8')
    answer_real_json_dict = json.load(answer_real_json)
    # 读取预测的json文件
    answer_json =  open(answer, 'r', encoding='utf-8')
    answer_json_dict = json.load(answer_json)

    # 得到预测图片数目
    num_img = len(answer_real_json_dict)
    # 初始得分
    score = 1
    # 遍历json文件中的图片键值对
    for idx, info in enumerate(answer_json_dict.items()):
        image_name, text_info_list = info
        # 读取真实文件字符串
        str_real = answer_real_json_dict[image_name]['result']
        # 读取预测文件字符串
        str_predict = answer_json_dict[image_name]['result']

        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(str_real, str_predict)
        # 计算两个字符串最大长度
        max_len = max(len(str_real),len(str_predict))

        # 如果为干扰项直接判0
        if str_real == ' ':
            score = score - 1/num_img
            print('干扰项:', score)
        else:
            if dis_str>0:
                score = score - (dis_str/max_len)/num_img
                print(num_img,dis_str,max_len,score)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 预测json路径
    parser.add_argument('--answer', type=str,default='./temp/answer.json')
    # 真实json路径
    parser.add_argument('--answer_real', type=str, default='./temp/answer_real.json')
    opt = parser.parse_args()
    # 真实数据
    answer_real = opt.answer_real
    # 预测数据
    answer = opt.answer

    # 生成标准测试集
    get_real_json(answer,answer_real)


'''bash
    python recognizer/score_out.py
'''













































