# -*- coding=utf-8 -*-
import paddle
import paddle.nn as nn
import paddle.fluid as fluid

# 卷积增长块，去除1×1卷积
class conv_block(nn.Layer):
    def __init__(self,in_channels, growth_rate,dropout_rate=0):
        super(conv_block, self).__init__()
        self.stage = nn.Sequential(
            nn.BatchNorm2D(in_channels),
            nn.ReLU(True),
            nn.Conv2D(in_channels, growth_rate, (3,3), stride=(1, 1), padding=1),
        )
        self.dropout_rate = dropout_rate

    def forward(self,X):
        X = self.stage(X)
        if self.dropout_rate > 0:
            X = fluid.layers.dropout(X, self.dropout_rate)
        return X


class dense_block(nn.Layer):
    def __init__(self,in_channels,nb_layers, growth_rate,droput_rate=0.2):
        super(dense_block, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.nb_layers = nb_layers
        self.dropout_rate = droput_rate

    def forward(self,X):
        # 必须使用个中间变量,不然无法更新通道数
        in_channels_temp = self.in_channels
        for i in range(self.nb_layers):
            # 卷积卷出来 growth_rate 个通道特征图
            cb = conv_block(in_channels_temp, self.growth_rate,self.dropout_rate)(X)
            # 在通道channel维度进行拼接
            X = paddle.concat(x=[X,cb], axis=1)
            in_channels_temp += self.growth_rate

        # 返回输出值及最终通道数
        return X

# 过渡块
class transition_block(nn.Layer):
    def __init__(self,in_channels,out_channels, dropout_rate=0, pooltype=1):
        super(transition_block, self).__init__()
        self.stage = nn.Sequential(
            nn.BatchNorm2D(in_channels),
            nn.ReLU(True),
            nn.Conv2D(in_channels, out_channels, (1,1), stride=(1, 1), padding=0),
        )
        self.dropout_rate = dropout_rate
        self.pooltype = pooltype
        self.out_channels = out_channels

    def forward(self,X):
        X = self.stage(X)
        # dropout
        if self.dropout_rate > 0:
            X = fluid.layers.dropout(X, self.dropout_rate)
        # 均值池化
        if (self.pooltype == 2):
            X = nn.AvgPool2D((2, 2), stride=(2, 2), padding=0)(X)
        elif (self.pooltype == 1):
            # padding [height,width] [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            X = nn.AvgPool2D((2, 2), stride=(2, 1), padding=(0,0,0,1))(X)
        elif (self.pooltype == 3):
            X = nn.MaxPool2D((2, 2), stride=(2, 2), padding=0)(X)
        elif (self.pooltype == 4):
            X = nn.MaxPool2D((2, 2), stride=(2, 1), padding=(0,0,0,1))(X)
        elif (self.pooltype == 5):
            pass
        else:
            pass

        return X

# 组网
class my_net(nn.Layer):
    def __init__(self,nclass):
        super(my_net, self).__init__()
        # ================== 第一层 ================== (2,2) (32,400)->(16,200)
        self.stage1 = nn.Sequential(
            nn.Conv2D(3, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2D((2, 2), stride=(2, 2), padding=0),
        )
        # ================== 第二层 ================== (2,2) (16,200)->(8,100)
        self.stage2 = nn.Sequential(
            # 64 + 6*32 = 256
            dense_block(64, 6, 32,droput_rate=0),
            transition_block(256, 128,  dropout_rate=0.2, pooltype=2),
        )
        # ================== 第三层 ================== (1,1) (8,100)->(8,100)
        self.stage3 = nn.Sequential(
            # 128 + 8*16 = 256
            dense_block(128, 8, 16,droput_rate=0),
            transition_block(256, 256,  dropout_rate=0.2, pooltype=5),
        )
        # ================== 第四层 ================== (2,1) (8,100)->(4,100)
        self.stage4 = nn.Sequential(
            # 256 + 8*16 = 384
            dense_block(256, 8, 16,droput_rate=0),
            transition_block(384, 256,  dropout_rate=0.2, pooltype=1),
        )
        # ================== 第五层 ================== (pass) (4,100)
        self.stage5 = nn.Sequential(
            # 256 + 8*32 = 512
            dense_block(256, 8, 32,droput_rate=0),
            transition_block(512, 256,  dropout_rate=0.2, pooltype=5),
        )
        # ================== 第六层 ================== (2,1) (4,100)->(2,50)
        self.stage6 = nn.Sequential(
            nn.Conv2D(256, 256, (5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2D(256),
            # -------------- 激活函数 ---------------
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Swish(),
            # ---------------------------------------
            nn.MaxPool2D((2, 2), stride=(2, 2), padding=0),
        )
        # ================== 第七层 ================== (pass) (2,50)
        self.stage7 = nn.Sequential(
            nn.Conv2D(256, 512, (3, 3), stride=(1, 1), padding=1),
        )
        self.FC = nn.Linear(in_features=512, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=nclass)

    def forward(self,X):
        # (2,2) (32,400)->(16,200) [64,3,32,400]
        out = self.stage1(X)
        # (2,2) (16,200)->(8,100)
        out = self.stage2(out) # [64, 128, 8, 100]
        # (1,1) (8,100)->(8,100)
        out = self.stage3(out)
        # (2,1) (8,100)->(4,100)
        out = self.stage4(out)
        # (pass) (4,100)
        out = self.stage5(out)
        # (2,1) (4,100)->(2,50)
        out = self.stage6(out)
        # (pass) (2,50)
        out = self.stage7(out)  # [64,512,2,50]
        batch, channel, height, width = out.shape
        # shape变为[64,512,1,100]
        out = paddle.reshape(out, shape=(batch, channel, 1, 100))
        # 将height==1的维度去掉-->BCW [64,512,100]
        conv = paddle.squeeze(out, axis=2)
        # 将时间通道放在前面 [100,64,512]
        conv = paddle.transpose(conv, perm=[2, 0, 1])
        # 展开 [6400,512]
        time, batch_size, chalnel_2 = conv.shape
        conv = paddle.reshape(conv, shape=(time * batch_size, chalnel_2))
        # FC层输出
        conv = self.FC(conv)
        conv = nn.Sigmoid()(conv)
        conv = self.output(conv)
        # conv = nn.Softmax()(conv)
        conv = paddle.reshape(conv, shape=(time, batch_size, -1)) # [100, 64, 1082]

        return conv
























