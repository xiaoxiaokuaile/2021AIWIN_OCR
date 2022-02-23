#!/bin/bash
echo "file name: $0"
echo "test image parh: $1"
cd PaddleOCR
# echo "train!"
# python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml
echo "predict!"
python tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model=output/mobilenet_train/latest Global.load_static_weights=false Global.infer_img=$1
mv answer.json ../
