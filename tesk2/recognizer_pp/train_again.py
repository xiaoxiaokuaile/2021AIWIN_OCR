# -*- coding=utf-8 -*-
import argparse
import os
import sys

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

import tensorflow as tf


# ==================================== 模型选择 ==========================================
# 调用自己模型
from recognizer.models.myself_net.crnn_model import CNN_CTC_MODLE
# 调用resnet模型
# from recognizer.models.resnet.crnn_resnet_model import CNN_CTC_MODLE
# 调用densenet模型
# from recognizer.models.densenet.crnn_densenet_model import CNN_CTC_MODLE
# =======================================================================================
from recognizer.tools.config import config
from recognizer.tools.generator import Generator

# 或import tensorflow.keras.backend as K
K = tf.keras.backend

# ============================= 自定义学习策略所需 =============================
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    print('lr',K.get_value(model.optimizer.lr))
    # 每隔50个epoch，学习率减小为原来的1/10
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir',default='./modle_save')
    # 训练集
    parser.add_argument('--image_dir',default='./dataset/train/images')
    parser.add_argument('--txt_dir',default='./dataset/train/label')
    # 加载预训练模型pre train model path
    parser.add_argument('--pre_train_model_path', type=str,default=r'./modle_save/CNN_CTC_base_modle.h5')

    parser = parser.parse_args()

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)

    K.set_session(sess)
    batch_size = 32
    max_label_length = 50
    epochs = 60
    base_model, model = CNN_CTC_MODLE(initial_learning_rate=0.0001,mode='train_again')

    # ============================ 加载预训练模型 =================================
    model.load_weights(parser.pre_train_model_path)
    print('load pre_train model successfull!')
    # ==========================================================================

    train_loader = Generator(root_path=parser.image_dir,
                             input_map_file=os.path.join(parser.txt_dir, 'real_train.txt'),
                             batch_size=batch_size,
                             max_label_length=max_label_length,
                             input_shape=(32, 400, 3),
                             is_training=True,
                             is_enhance=True)
    # 保存模型文件文件
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(parser.model_save_dir, 'CNN_CTC_Final.h5'),
        # filepath=os.path.join(parser.model_save_dir, 'weights_crnn_pre_train-{epoch:03d}-{loss:.3f}.h5'),
        monitor='loss', save_best_only=True, save_weights_only=True
    )
    # 调整学习率：
    change_learning_rate = LearningRateScheduler(scheduler)
    # 训练微调模型
    model.fit_generator(generator=train_loader.__next__(),
                        steps_per_epoch=train_loader.num_samples() // batch_size,
                        epochs=epochs,
                        initial_epoch=0,
                        workers = 4,
                        verbose=1,
                        use_multiprocessing=True,
                        callbacks=[checkpoint, change_learning_rate])


'''bash
    python recognizer/train_again.py
'''
