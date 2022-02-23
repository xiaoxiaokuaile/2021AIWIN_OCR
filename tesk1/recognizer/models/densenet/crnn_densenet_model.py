# -*- coding=utf-8 -*-
import tensorflow as tf
from recognizer.models.densenet.crnn_densenet import densenet_crnn_time

K = tf.keras.backend
Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input


def ctc_lambda_func(args):
    prediction, label, input_length, label_length = args
    return K.ctc_batch_cost(label, prediction, input_length, label_length)
# ================================ densenet ================================
def CNN_CTC_MODLE(initial_learning_rate=0.0005,mode='train'):
    shape = (32, 400, 3)
    input = tf.keras.layers.Input(shape=shape, name='input_data')
    y_pred = densenet_crnn_time(inputs=input, activation='softmax',mode=mode)
    basemodel = tf.keras.models.Model(inputs=input, outputs=y_pred)
    # basemodel.summary()
    labels = Input(name='label', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    if mode == 'train':
        model = tf.keras.models.Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    elif mode == 'train_again':
        model = tf.keras.models.Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
        # fintune时候冻结网络层数
        # for layer in model.layers[:50]:
        #     layer.trainable = False

    model.summary()
    model.compile(loss={'ctc': lambda y_true, prediction: prediction},
                optimizer=tf.keras.optimizers.Adam(initial_learning_rate), metrics=['accuracy'])

    return basemodel, model

