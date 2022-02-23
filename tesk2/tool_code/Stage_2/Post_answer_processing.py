# Learner: 王振强
# Learn Time: 2021/11/30 21:36
import json
import argparse
import os
# 计算编辑距离
import Levenshtein

"""
    对answer预测结果进行后处理
"""


# ------------ 判定字符串包含关系, KMP --------------
def getnext(a,needle):
    next=['' for i in range(a)]
    k=-1
    next[0]=k
    for i in range(1,len(needle)):
        while (k>-1 and needle[k+1]!=needle[i]):
            k=next[k]
        if needle[k+1]==needle[i]:
            k+=1
        next[i]=k
    return next


# haystack='hello' , needle='ll',若存在包含关系返回值>=0,否则返回值小于0
def strStr(haystack, needle):
    a=len(needle)
    b=len(haystack)
    if a==0:
        return 0
    next= getnext(a,needle)
    p=-1
    for j in range(b):
        while p>=0 and needle[p+1]!=haystack[j]:
            p=next[p]
        if needle[p+1]==haystack[j]:
            p+=1
        if p==a-1:
            return j-a+1
    return -1


def str_Str(label_day,day_temp):
    KMP = -1
    result_value = label_day
    for day_value in day_temp:
        # 判定预测值长度是否大于标签表长度，大于的才可进行匹配
        if len(day_value)<len(label_day):
            KMP = strStr(label_day, day_value)
            # 若匹配成功则得到结果直接返回
            if KMP>=0:
                result_value = day_value
            else:
                pass
        else:
            pass

    # 若没有可以匹配的,直接在列表中找第一个
    if KMP<0:
        result_value = day_temp[0]
    else:
        pass

    return result_value
# ----------------------------------------------


# bank后处理
def bank_processing(label_bank,bank_list):
    # 初始化真实标签
    real_label = label_bank
    # 初始化编辑距离
    distance = len(label_bank)+1
    for bank in bank_list:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_bank, bank)
        if dis_str<distance:
            distance = dis_str
            real_label = bank
        else:
            pass

    return real_label


# day后处理
def day_processing(label_day):
    day_dict = ['壹','贰','叁','肆','伍','陆','柒','捌','玖',
                '零壹','零贰','零叁','零肆','零伍','零陆','零柒','零捌','零玖',
                '零壹拾','壹拾','壹拾壹','壹拾贰','壹拾叁','壹拾肆','壹拾伍','壹拾陆','壹拾柒','壹拾捌','壹拾玖',
                '零贰拾','贰拾','贰拾壹','贰拾贰','贰拾叁','贰拾肆','贰拾伍','贰拾陆','贰拾柒','贰拾捌','贰拾玖',
                '零叁拾','叁拾','叁拾壹']

    # 初始化真实标签
    real_label = label_day
    # 初始化编辑距离
    distance = len(label_day)+1

    # 找出最小编辑距离
    for day in day_dict:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_day, day)
        if dis_str<distance:
            distance = dis_str
        else:
            pass

    # 记录符合最小编辑距离的所有样本
    day_temp = []
    # 记录字符串长度符合要求的样本
    long_str = []
    # 最小编辑距离的模板有多个
    for day2 in day_dict:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_day, day2)
        # 若有样本等于最小距离
        if dis_str == distance:
            day_temp.append(day2)
        else:
            pass

    # 输出结果
    if len(day_temp)==1:
        # 若只有一个最小编辑距离则直接赋值
        real_label =  day_temp[0]
    else:
        # 若有多个最小编辑距离则判定字符串长度是否一致
        for mon in day_temp:
            # 判断字符串长度
            if len(mon)==len(label_day):
                long_str.append(mon)
            else:
                pass
        # 若没有字符串相等则看包含关系
        if len(long_str)==0:
            # 若字符串不相等
            real_label = str_Str(label_day,day_temp)
        else:
            # 若存在字符串长度相同,则直接返回第一个就行
            real_label = long_str[0]

    return real_label


# month后处理
def month_processing(label_month):
    month_dict = ['壹','贰','叁','肆','伍','陆','柒','捌','玖',
                  '零壹','零贰','零叁','零肆','零伍','零陆','零柒','零捌','零玖',
                  '壹拾','零壹拾','壹拾壹','拾壹','壹拾贰','拾贰']
    # 初始化真实标签
    real_label = label_month
    # 初始化编辑距离
    distance = len(label_month)+1

    # 找出最小编辑距离
    for month in month_dict:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_month, month)
        if dis_str<distance:
            distance = dis_str
        else:
            pass

    # 记录符合最小编辑距离的所有样本
    month_temp = []
    # 记录字符串长度符合要求的样本
    long_str = []
    # 最小编辑距离的模板有多个
    for month2 in month_dict:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_month, month2)
        # 若有样本等于最小距离
        if dis_str == distance:
            month_temp.append(month2)
        else:
            pass

    # 输出结果
    if len(month_temp)==1:
        # 若只有一个最小编辑距离则直接赋值
        real_label =  month_temp[0]
    else:
        # 若有多个最小编辑距离则判定字符串长度是否一致
        for mon in month_temp:
            # 判断字符串长度
            if len(mon)==len(label_month):
                long_str.append(mon)
            else:
                pass
        # 若没有字符串相等则看包含关系
        if len(long_str)==0:
            # 若字符串不相等
            # real_label = month_temp[0]
            # 包含关系
            real_label = str_Str(label_month, month_temp)

        else:
            # 若存在字符串长度相同,则直接返回第一个就行
            real_label = long_str[0]

    return real_label


# year后处理
def year_processing(label_year):
    year_dict = ['贰零壹玖','贰零壹柒','贰零壹捌','贰零贰零']
    # 初始化真实标签
    real_label = label_year
    # 初始化编辑距离
    distance = len(label_year)+1
    for year in year_dict:
        # 计算两个字符串编辑距离
        dis_str = Levenshtein.distance(label_year, year)
        if dis_str<distance:
            distance = dis_str
            real_label = year
        else:
            pass

    return real_label


# sum后处理
def sum_processing(label_sum):
    """
        sum具有较强的序列关系,网络中RNN部分已经可以较好的处理纠正了,故暂不处理
    """
    return label_sum


# 所有后处理
def all_processing(answer,bank_dict,new_answer_save_path):

    # 读取json文件
    global real_label, confidence
    answer_json =  open(answer, 'r', encoding='utf-8')
    answer_json_dict = json.load(answer_json)
    # 初始化存储Bank的列表
    bank_list = []
    # 读取bank dict的文档
    with open(bank_dict, 'r', encoding='utf-8') as in_file:
        bank_lines = in_file.readlines()
        for line in bank_lines:
            line = line.strip('\r').strip('\n')
            bank_list.append(line)

    label_info_dict = {}
    # 遍历json预测文件处理
    for img_name,info in answer_json_dict.items():
        pre_name = img_name.split('_')[0]
        if pre_name == 'bank':
            # 处理bank
            label_bank = answer_json_dict[img_name]['result']
            confidence = answer_json_dict[img_name]['confidence']
            real_label = bank_processing(label_bank, bank_list)
        elif pre_name == 'day':
            # 处理day
            label_day = answer_json_dict[img_name]['result']
            confidence = answer_json_dict[img_name]['confidence']
            real_label = day_processing(label_day)
        elif pre_name == 'month':
            # 处理month
            label_month = answer_json_dict[img_name]['result']
            confidence = answer_json_dict[img_name]['confidence']
            real_label = month_processing(label_month)
        elif pre_name == 'year':
            # 处理year
            label_year = answer_json_dict[img_name]['result']
            confidence = answer_json_dict[img_name]['confidence']
            real_label = year_processing(label_year)
        elif pre_name == 'sum':
            # 处理sum
            label_sum = answer_json_dict[img_name]['result']
            confidence = answer_json_dict[img_name]['confidence']
            real_label = sum_processing(label_sum)
        else:
            pass

        # 保存修正过的label
        label_info_dict[img_name] = {
            'result': real_label,
            'confidence': confidence
        }
    with open(new_answer_save_path, "w", encoding='utf-8') as fout:
        fout.write(json.dumps(label_info_dict, ensure_ascii=False, indent=4, separators=(',', ':')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # answer地址
    parser.add_argument('--answer_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2B\4_mobile_train_/answer.json')
    # bank地址库地址
    parser.add_argument('--bank_dict_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset/bank_dict_testB.txt')
    # 新的answer地址
    parser.add_argument('--new_answer_path', type=str,
                        default=r'F:\JS\2021AIWIN\OCR\提交过程\tesk2B\4_mobile_train_\后处理/answer.json')
    opt = parser.parse_args()

    all_processing(opt.answer_path,opt.bank_dict_path,opt.new_answer_path)

