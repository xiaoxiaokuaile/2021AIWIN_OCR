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
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import TimeDistributed


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

# 过渡块
def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    # 作1×1卷积操作调整通道数
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,kernel_regularizer=l2(weight_decay))(x)
    # dropout
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    # 均值池化
    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1), padding='same')(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1), padding='same')(x)
    elif (pooltype == 4):
        pass
    return x, nb_filter

def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # ========================== 第一层 ===============================(2,2) (32,400)->(16,200)
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(1, 1), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # ========================== 第二层 ===============================(2,2) (16,200)->(8,100)
    # 64 + 6 * 32 = 256
    x, _nb_filter = dense_block(x, 6, _nb_filter, 32, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    # ======================= 第三层+第四层 =============================(2,1) (8,100)->(4,100)
    # 128 + 8 * 16 = 256
    x, _nb_filter = dense_block(x, 8, _nb_filter, 16, None, _weight_decay)
    # 只压缩通道不maxpooling
    x, _nb_filter = transition_block(x, 256, _dropout_rate, 4, _weight_decay)
    # 128 + 8 * 16 = 256
    x, _nb_filter = dense_block(x, 8, _nb_filter, 16, None, _weight_decay)
    # 256 -> 128
    x, _nb_filter = transition_block(x, 256, _dropout_rate, 3, _weight_decay)
    # ========================== 第五层 ===============================(pass) (4,100)
    # 256 + 8 * 32 = 512
    x, _nb_filter = dense_block(x, 8, _nb_filter, 32, None, _weight_decay)
    # 512 -> 256
    x, _nb_filter = transition_block(x, 256, _dropout_rate, 4, _weight_decay)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    # ========================== 第六层 ===============================(2,1) (4,100)->(2,50)
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = LeakyRelu(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # ========================== 第七层 ================================(pass) (2,50)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Reshape((100, 1, 512), name='reshape')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    # =================== 学PPOCR加FC ======================================
    x = Dense(512, name='FC',activation='sigmoid')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred


def densenet_crnn_time(inputs, activation=None,mode='train', include_top=False):
    x = dense_cnn(inputs, config.num_class)
    return x