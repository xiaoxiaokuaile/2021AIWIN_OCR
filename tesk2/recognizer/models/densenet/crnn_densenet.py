# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers import LeakyReLU

from recognizer.tools.config import config
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import TimeDistributed


def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, dropout_rate=None):
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2):
    for i in range(nb_layers):
        cb = DenseLayer(x, growth_rate, dropout_rate=droput_rate)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

# 过渡块
def transition_block(input, nb_filter,alpha=0.0, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    # 作1×1卷积操作调整通道数
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = LeakyReLU(alpha=alpha)(x)
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
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    elif (pooltype == 5):
        x = MaxPooling2D((2, 2), strides=(2, 1), padding='same')(x)
    elif (pooltype == 6):
        pass
    else:
        pass

    return x, nb_filter

def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # (2,2) (32,400)->(8,100)
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (7, 7), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (pass) (8,100)->(8,100)
    # 64 + 6 * 32 = 256
    x, _nb_filter = dense_block(x, 6, _nb_filter, 32)
    x, _nb_filter = transition_block(x, 128, alpha = 0.0,dropout_rate = _dropout_rate, pooltype=6, weight_decay=_weight_decay)
    # (2,1) (8,100)->(4,100)
    # 128 + 12 * 32 = 256
    x, _nb_filter = dense_block(x, 12, _nb_filter, 32)
    x, _nb_filter = transition_block(x, 256, alpha = 0.0,dropout_rate = _dropout_rate, pooltype=5, weight_decay=_weight_decay)
    # (2,1) (4,100)-(2,50)
    x, _nb_filter = dense_block(x, 24, _nb_filter, 32)
    x, _nb_filter = transition_block(x, 512, alpha = 0.0,dropout_rate = _dropout_rate, pooltype=4, weight_decay=_weight_decay)
    # (pass) (2,50)
    # 256 + 8 * 32 = 512
    x, _nb_filter = dense_block(x, 16, _nb_filter, 32)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Reshape((100, 1, 1024), name='reshape')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    # =================== 学PPOCR加FC ======================
    # x = Dense(512, name='FC',activation='sigmoid')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred


def densenet_crnn_time(inputs, activation=None,mode='train', include_top=False):
    x = dense_cnn(inputs, config.num_class)
    return x