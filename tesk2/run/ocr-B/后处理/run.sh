#!/bin/bash
echo "file name: $0"
echo "test image parh: $1"
cd PaddleOCR
# echo "train!"
# python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml
echo "predict!"
# mobilenet
#python tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml \
#                          -o Global.checkpoints=output/mobilenet_train/best_accuracy \
#                          Global.load_static_weights=false Global.infer_img=$1
# resnet
 python tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml \
                           -o Global.checkpoints=output/resnet_testB_train/best_accuracy \
                           Global.load_static_weights=false Global.infer_img=$1
# 将PaddleOCR预测结果放到 /tool_code/Stage_2
mv answer.json ../tool_code/Stage_2
# 切换路径到 tool_code
cd ../tool_code
# 后处理
python Post_answer_processing.py
# 将预测结果放到上级目录
mv answer.json ..

