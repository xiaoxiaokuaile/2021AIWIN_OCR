# -*- coding=utf-8 -*-
import paddle
import paddle.nn as nn


class ConvBlock(nn.Layer):
    def __init__(self, in_channel, kernel_size, filters, strides=(2, 2), activate=True):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.stage = nn.Sequential(
            # -------------------------------#
            #   利用1x1卷积进行通道数的下降
            # -------------------------------#
            nn.Conv2D(in_channel, filters1, (1, 1), stride=strides, padding=0),
            nn.BatchNorm2D(filters1),
            nn.ReLU(True),
            # -------------------------------#
            #   利用3x3卷积进行特征提取
            # -------------------------------#
            nn.Conv2D(filters1, filters2, kernel_size, stride=(1, 1), padding=1),
            nn.BatchNorm2D(filters2),
            nn.ReLU(True),
            # -------------------------------#
            #   利用1x1卷积进行通道数的上升
            # -------------------------------#
            nn.Conv2D(filters2, filters3, (1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2D(filters3),
        )
        # -------------------------------#
        #   将残差边也进行调整
        #   才可以进行连接
        # -------------------------------#
        self.shortcut_1 = nn.Conv2D(in_channel, filters3, (1, 1), stride=strides, padding=0)
        self.batch_1 = nn.BatchNorm2D(filters3)

        self.relu_1 = nn.ReLU(True)
        self.activate = activate

    def forward(self, X):
        # 残差边
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        # 主干网
        X = self.stage(X)

        X = X + X_shortcut
        # -------------------------------#
        #   最后一层是否使用激活函数
        # -------------------------------#
        if self.activate:
            X = self.relu_1(X)
        else:
            pass

        return X


class IndentityBlock(nn.Layer):
    def __init__(self, in_channel, kernel_size, filters, activate=True):
        super(IndentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.stage = nn.Sequential(
            # -------------------------------#
            #   利用1x1卷积进行通道数的下降
            # -------------------------------#
            nn.Conv2D(in_channel, filters1, (1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2D(filters1),
            nn.ReLU(True),
            # -------------------------------#
            #   利用3x3卷积进行特征提取
            # -------------------------------#
            nn.Conv2D(filters1, filters2, kernel_size, stride=1, padding=1),
            nn.BatchNorm2D(filters2),
            nn.ReLU(True),
            # -------------------------------#
            #   利用1x1卷积进行通道数的上升
            # -------------------------------#
            nn.Conv2D(filters2, filters3, (1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2D(filters3),
        )
        self.relu_1 = nn.ReLU(True)
        self.activate = activate

    def forward(self, X):
        # 残差边
        X_shortcut = X
        # 主干网
        X = self.stage(X)
        X = X + X_shortcut
        # -------------------------------#
        #   最后一层是否使用激活函数
        # -------------------------------#
        if self.activate:
            X = self.relu_1(X)
        else:
            pass

        return X


class Resnet50(nn.Layer):
    def __init__(self, n_class):
        super(Resnet50, self).__init__()
        # (32,400)->(16,200)  (16,200)-(8,100)
        self.stage1 = nn.Sequential(
            nn.Conv2D(3, 64, (7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(True),
            nn.MaxPool2D((3, 3), stride=(2, 2), padding=1),
        )
        # (8,100)-(8,100)
        self.stage2 = nn.Sequential(
            ConvBlock(64, kernel_size=(3, 3), filters=[64, 64, 256], strides=(1, 1)),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )
        # (8,100)-(4,100)
        self.stage3 = nn.Sequential(
            ConvBlock(256, kernel_size=(3, 3), filters=[128, 128, 512], strides=(2, 1)),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )
        # (4,100)-(2,50)
        self.stage4 = nn.Sequential(
            ConvBlock(512, kernel_size=(3, 3), filters=[256, 256, 1024], strides=(2, 2)),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )
        # (2,50)-(2,50)
        self.stage5 = nn.Sequential(
            ConvBlock(1024, kernel_size=(3, 3), filters=[512, 512, 2048], strides=(1, 1)),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048], activate=True),
        )
        # FC
        self.output = nn.Linear(in_features=2048, out_features=n_class)

    def forward(self, X):
        # (32,400)->(16,200)  (16,200)-(8,100)
        out = self.stage1(X)
        # (8,100)-(8,100)
        out = self.stage2(out)
        # (8,100)-(4,100)
        out = self.stage3(out)
        # (4,100)-(2,50)
        out = self.stage4(out)
        # (2,50)-(2,50)
        out = self.stage5(out)  # [64, 2048, 2, 50]
        batch, channel, height, width = out.shape
        # shape变为[64,2048,1,100]
        out = paddle.reshape(out, shape=(batch, channel, 1, 100))
        # 将height==1的维度去掉-->BCW [64,2048,100]
        conv = paddle.squeeze(out, axis=2)
        # 将时间通道放在前面 [100,64,2048]
        conv = paddle.transpose(conv, perm=[2, 0, 1])
        # 展开
        time, batch_size, chalnel_2 = conv.shape
        conv = paddle.reshape(conv, shape=(time * batch_size, chalnel_2))
        # FC层输出
        conv = self.output(conv)
        # # paddle训练时候最后一层不能加softmax激活函数,需要预测时候加
        # conv = nn.Softmax()(conv)
        conv = paddle.reshape(conv, shape=(time, batch_size, -1))

        return conv
















