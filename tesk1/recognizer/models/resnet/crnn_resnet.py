# -*- coding=utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from recognizer.tools.config import config
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
LeakyRelu = tf.keras.layers.LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import TimeDistributed


# 直接相加
def identity_block(input_tensor, kernel_size, filters, stage, block,activate=True):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # -------------------------------#
    #   利用1x1卷积进行通道数的下降
    # -------------------------------#
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # -------------------------------#
    #   利用3x3卷积进行特征提取
    # -------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # -------------------------------#
    #   利用1x1卷积进行通道数的上升
    # -------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])

    # -------------------------------#
    #   最后一层是否使用激活函数
    # -------------------------------#
    if activate:
        x = Activation('relu')(x)
    else:
        pass

    return x

# 卷积一层+卷积3层
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),activate=True):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # -------------------------------#
    #   利用1x1卷积进行通道数的下降
    # -------------------------------#
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # -------------------------------#
    #   利用3x3卷积进行特征提取
    # -------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # -------------------------------#
    #   利用1x1卷积进行通道数的上升
    # -------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # -------------------------------#
    #   将残差边也进行调整
    #   才可以进行连接
    # -------------------------------#
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])

    # -------------------------------#
    #   最后一层是否使用激活函数
    # -------------------------------#
    if activate:
        x = Activation('relu')(x)
    else:
        pass
    return x

def ResNet50(img_input, nclass):

    x = ZeroPadding2D((3, 3))(img_input)

    # (32,400)->(16,200)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    #  (16,200)-(8,100)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # (8,100)-(8,100)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # (8,100)-(4,100)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 1))  # 默认(2,2)stride
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # (4,100)-(2,50)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # (2,50)-(2,50)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1))
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',activate=True)

    # 展开
    x = Reshape((100, 1, 2048), name='reshape')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)

    # x = Dense(512, name='FC',activation='sigmoid')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred

def resnet_crnn_time(inputs, activation=None,mode='train', include_top=False):
    x = ResNet50(inputs, config.num_class)
    return x