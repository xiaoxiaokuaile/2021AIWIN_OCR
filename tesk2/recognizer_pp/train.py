# -*- coding=utf-8 -*-
import argparse
import os
import sys

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)
# ==================================== 模型选择 ==========================================
# 调用自己模型
# from recognizer_pp.models.myself_net.crnn import my_net
# 调用resnet模型
from recognizer_pp.models.resnet.crnn_resnet import Resnet50
# 调用densenet模型
# from recognizer_pp.models.densenet.crnn_densenet import densenet121
# =======================================================================================
import paddle
from recognizer_pp.tools.generator import Generator, collate_fn
from paddle.io import DataLoader
from datetime import datetime
import numpy as np
from visualdl import LogWriter
from recognizer_pp.tools.decoder import ctc_greedy_decoder, label_to_string, cer


# 评估模型
def evaluate(model, test_loader, vocabulary):
    cer_result = []
    for batch_id, (inputs, labels, _, _) in enumerate(test_loader()):
        # 执行识别
        outs = model(inputs)
        outs = paddle.transpose(outs, perm=[1, 0, 2])
        outs = paddle.nn.functional.softmax(outs)
        # 解码获取识别结果
        labelss = []
        out_strings = []
        for out in outs:
            out_string = ctc_greedy_decoder(out, vocabulary)
            out_strings.append(out_string)
        for i, label in enumerate(labels):
            # 标签转换为文字  50->字符串长度
            label_str = label_to_string(label, vocabulary)
            labelss.append(label_str)
        for out_string, label in zip(out_strings, labelss):
            # 计算准确率
            max_len = max(len(out_string), len(label))

            score = 1 - cer(out_string, label) / max_len

            cer_result.append(score)

    cer_out = float(np.mean(cer_result))
    return cer_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 词汇表路径
    parser.add_argument('--char_path', type=str, default='./recognizer_pp/tools/chars.txt')
    # 模型保存路径
    parser.add_argument('--model_save_dir', default='./modle_save/')
    # 设置训练数据集
    parser.add_argument('--image_dir', default='/home/aistudio/work/dataset/extra_train/images/')
    parser.add_argument('--txt_dir', default='/home/aistudio/work/dataset/extra_train/label/fix_train.txt')
    # 设置测试数据集
    parser.add_argument('--image_test', default='/home/aistudio/work/dataset/extra_test/images/')
    parser.add_argument('--txt_test', default='/home/aistudio/work/dataset/extra_test/label/fix_train.txt')
    parser = parser.parse_args()

    with open(parser.char_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    vocabulary = [labels[i].replace('\n', '') for i in range(len(labels))]

    # 每一批数据大小
    batch_size = 128
    # 预训练模型路径
    pretrained_model = parser.model_save_dir
    # pretrained_model = None
    # 训练轮数
    num_epoch = 60
    # 类别数目
    n_class = len(vocabulary)
    # 最大字符长度
    max_label_length = 50
    # 日志记录噐
    writer = LogWriter(logdir='log')

    # 获取训练数据(返回每张图片及 数值化标签)
    train_dataset = Generator(parser.image_dir, parser.txt_dir, (32, 400, 3), is_enhance=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # # 获取测试数据
    test_dataset = Generator(parser.image_test, parser.txt_test, (32, 400, 3), is_enhance=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 获取模型
    model = Resnet50(n_class)
    # model = densenet121(n_class)
    # model = my_net(n_class)
    paddle.summary(model, input_size=(batch_size, 3, 32, 400))

    # 设置学习策略
    # scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[20, 40], values=[0.001, 0.0001, 0.00001], verbose=False)
    # # 自适应学习策略
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.001, factor=0.3, patience=5, verbose=True)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(1e-4))

    # 获取损失函数,空白键为最后一位,默认为0
    ctc_loss = paddle.nn.CTCLoss(reduction='mean')
    # 加载预训练模型
    if pretrained_model is not None:
        model.set_state_dict(paddle.load(os.path.join(parser.model_save_dir, 'model.pdparams')))
        # optimizer.set_state_dict(paddle.load(os.path.join(parser.model_save_dir, 'optimizer.pdopt')))
        print('load model successful!!!')

    train_step = 0
    test_step = 0
    max_score = 0
    # 计算训练每个epoch的step
    step_num = len(train_loader())
    # 遍历所有轮次
    for epoch in range(num_epoch):
        # 输出每个epoch的lr
        print('lr:', scheduler.last_lr)
        # 记录loss平均值
        loss_list = []
        # [64, 3, 32, 400]; [64, 21]; [100, 100, 100, 100,...]; [3 , 3 , 13, 15, 2 , ...]
        for batch_id, (inputs, labels, input_lengths, label_lengths) in enumerate(train_loader()):

            out = model(inputs)
            # print(out.shape)
            # 计算损失
            loss = ctc_loss(out, labels, input_lengths, label_lengths)
            loss_list.append(loss)
            # print(loss)
            loss.backward()
            # 模型更新
            optimizer.step()
            optimizer.clear_grad()
            # 1000 个轮次显示一次
            if batch_id % 500 == 0:
                print('[%s] Train epoch [%d/%d], batch [%d/%d], loss: %f' % (
                datetime.now(), epoch, num_epoch, batch_id, step_num, loss))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1

        # 禁用动态图梯度计算,可以在验证评估时候不额外占用显存资源
        with paddle.no_grad():
            # 执行评估
            if epoch % 1 == 0:
                model.eval()
                cer_score = evaluate(model, test_loader, vocabulary)
                print('[%s] Test epoch %d, cer: %f' % (datetime.now(), epoch, cer_score))
                writer.add_scalar('Test cer', cer_score, test_step)
                test_step += 1
                model.train()
        # 记录学习率
        writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
        # 计算每个epoch平均Loss
        loss_mean = np.mean(loss_list)
        print('[%s] Train epoch [%d/%d],mean loss: %f' % (datetime.now(),epoch, num_epoch,loss_mean ))
        # 调整学习率
        scheduler.step(loss_mean)

        if cer_score > max_score:
            max_score = cer_score
            # 若该epoch模型效果好于之前的,就更新模型
            paddle.save(model.state_dict(), os.path.join(parser.model_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(parser.model_save_dir, 'optimizer.pdopt'))

        # 更新最后一个模型
        paddle.save(model.state_dict(), os.path.join(parser.model_save_dir, 'model_last.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(parser.model_save_dir, 'optimizer_last.pdopt'))

        # break

'''bash
    python recognizer_pp/train.py
'''

