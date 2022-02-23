# Learner: 王振强
# Learn Time: 2021/11/27 13:01
import argparse
import json
"""
    处理预测结果的bank和其它类
    1.1. 将answer的bank置为空
    1.2.抽取某一属性
    2.将带标签的answer还原
    3.融合两个answer的bank和date部分
"""

# -------------------- 1.1. 将answer的bank置为空 -------------------
def answer_to_space(old_answer_path,new_answer_path):

    # 读取json文件
    test_real_json =  open(old_answer_path, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)
    # 遍历json预测文件
    for img_name,info in test_real_json_dict.items():
        pre_name = img_name.split('_')[0]
        if pre_name == 'bank':
            # 处理bank
            test_real_json_dict[img_name]['result'] = ' '
        else:
            pass

    # 保存替换后的json文件
    json_file = open(new_answer_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(test_real_json_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

# -------------------- 1.2.抽取某一属性 -------------------
def date_sum_to_space(old_answer_path,new_answer_path):

    # 读取json文件
    test_real_json =  open(old_answer_path, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)
    # 遍历json预测文件
    for img_name,info in test_real_json_dict.items():
        pre_name = img_name.split('_')[0]
        if pre_name == 'bank':
            pass
        else:
            # 处理bank
            test_real_json_dict[img_name]['result'] = ' '

    # 保存替换后的json文件
    json_file = open(new_answer_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(test_real_json_dict, ensure_ascii=False, indent=4, separators=(',', ':')))


# -------------------- 2.将带标签的answer还原 -------------------
def answer_to_noindex(old_answer_path,new_answer_path):

    # 读取json文件
    test_real_json =  open(old_answer_path, 'r', encoding='utf-8')
    test_real_json_dict = json.load(test_real_json)
    # 创建保存新键值对的字典
    answer_dict = {}
    # 遍历json预测文件
    for img_name,info in test_real_json_dict.items():

        # 获取图片真实名称
        pre_name = img_name.split('_',1)[1]
        # print(pre_name)
        # 替换对应字符串
        value = test_real_json_dict[img_name]['result']
        answer_dict[pre_name] = {
            'result': value,
            'confidence': 1
        }
    # 保存替换后的json文件
    json_file = open(new_answer_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(answer_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

# -------------------- 3.融合两个answer的bank和date部分 -------------------
def bank_change_date_sum(answer_bank_path,answer_date_sum_path,answer_add_save_path):

    # 读取 bank json文件
    bank_json = open(answer_bank_path, 'r', encoding='utf-8')
    bank_json_dict = json.load(bank_json)
    # 读取 date_sum json文件
    date_sum_json = open(answer_date_sum_path, 'r', encoding='utf-8')
    date_sum_json_dict = json.load(date_sum_json)
    # 记录的字典
    result_dict = {}
    # 遍历json预测文件
    for img_name,info in bank_json_dict.items():
        pre_name = img_name.split('_')[0]
        if pre_name == 'bank':
            value = bank_json_dict[img_name]['result']
            # 处理bank
            result_dict[img_name] = {
                'result': value,
                'confidence': 1
            }
        else:
            value = date_sum_json_dict[img_name]['result']
            # 处理date sum
            result_dict[img_name] = {
                'result': value,
                'confidence': 1
            }

    # 保存替换后的json文件
    json_file = open(answer_add_save_path, 'w', encoding='utf-8')
    json_file.write(json.dumps(result_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- 1.将answer的bank置为空 -------------------
    # 年、月、日、金额answer路径
    file_path = r'F:\JS\2021AIWIN\OCR\提交过程\tesk2\15_resnet34vd_train_500_\last\后处理'
    parser.add_argument('--answer_1_path', type=str,default=file_path + '/answer.json')
    # 处理后json保存路径
    parser.add_argument('--new_answer_path', type=str, default=file_path + '/answer_new.json')
    # -------------------- 2.将带标签的answer还原 -------------------
    # 需要还原的answer_check.json
    parser.add_argument('--answer_check_path', type=str,default='./dataset/test_check/label/answer_check.json')
    # 处理后json保存路径
    parser.add_argument('--noindex_answer_path', type=str, default='./dataset/test_check/label/answer.json')
    # -------------------- 3.融合两个answer的bank和date部分 -------------------
    # 提取bank
    parser.add_argument('--answer_bank_path', type=str,default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\test\bank_old0.18926_new0.19855\new/answer_bank_real.json')
    # 提取date_sum
    parser.add_argument('--answer_date_sum_path', type=str, default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\test/answer_sum_date_real.json')
    # 新合并的answer保存地址
    parser.add_argument('--answer_add_save_path', type=str, default=r'F:\JS\2021AIWIN\OCR\tesk2\submit\dataset\test/answer.json')
    opt = parser.parse_args()

    # -------------------- 1.1.将answer的bank置为空 -------------------
    # answer_to_space(opt.answer_1_path,opt.new_answer_path)
    # -------------------- 1.2.将answer的 date sum 置为空 -------------------
    date_sum_to_space(opt.answer_1_path,opt.new_answer_path)
    # -------------------- 2.将带标签的answer还原 -------------------
    # answer_to_noindex(opt.answer_check_path,opt.noindex_answer_path)
    # -------------------- 3.融合两个answer的bank和date部分 -------------------
    # bank_change_date_sum(opt.answer_bank_path,opt.answer_date_sum_path,opt.answer_add_save_path)

'''bash
    python tool_code/mix_bank_date.py
'''


















