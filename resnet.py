from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from config import config

# 训练参数
batch_size = config.batch_size  # 原论文按照 batch_size=128 训练所有的网络
epochs = 200
data_augmentation = True
num_classes = 6

# 减去像素均值可提高准确度
subtract_pixel_mean = True

# 模型参数
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

def lr_schedule(epoch):
    """学习率调度

    学习率将在 80, 120, 160, 180 轮后依次下降。
    他作为训练期间回调的一部分，在每个时期自动调用。

    # 参数
        epoch (int): 轮次

    # 返回
        lr (float32): 学习率
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D 卷积批量标准化 - 激活栈构建器

    # 参数
        inputs (tensor): 从输入图像或前一层来的输入张量
        num_filters (int): Conv2D 过滤器数量
        kernel_size (int): Conv2D 方形核维度
        strides (int): Conv2D 方形步幅维度
        activation (string): 激活函数名
        batch_normalization (bool): 是否包含批标准化
        conv_first (bool): conv-bn-activation (True) 或
            bn-activation-conv (False)

    # 返回
        x (tensor): 作为下一层输入的张量
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=num_classes):
    """ResNet 版本 2 模型构建器 [b]

    (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D 的堆栈，也被称为瓶颈层。
    每一层的第一个快捷连接是一个 1 x 1 Conv2D。
    第二个及以后的快捷连接是 identity。
    在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
    而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
    特征图尺寸:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # 参数
        input_shape (tensor): 输入图像张量的尺寸
        depth (int): 核心卷积层的数量
        num_classes (int): 类别数 (CIFAR10 为 10)

    # 返回
        model (Model): Keras 模型实例
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # 开始模型定义。
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 在将输入分离为两个路径前执行带 BN-ReLU 的 Conv2D 操作。
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # 实例化残差单元的栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # 瓶颈残差单元
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            # if(res_block == 0):
            #     y = MaxPool2D(pool_size=2)(y)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # 线性投影残差快捷键连接，以匹配更改的 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            # if(res_block == 0):
            #     x = MaxPool2D(pool_size=2)(x)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # 在顶层添加分类器
    # v2 has BN-ReLU before Pooling
    # x = MaxPool2D(pool_size=2)(x) # add
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=7)(x)
    y = Flatten()(x)
    # y = Dense(128)(y) # add
    # y = Dropout(0.4)(y) # add

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 实例化模型。
    model = Model(inputs=inputs, outputs=outputs)
    return model


def res_model(input_shape=config.input_shape, train_generator=None, validation_generator=None,
              model_save_path=config.model_path, log_dir=config.logs_path):

    model = resnet_v2(input_shape=input_shape, depth=20)

    model.summary()
    model.compile(
        # 是优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer=Adam(lr=0.001),
        # 损失函数,多分类采用 categorical_crossentropy
        loss='categorical_crossentropy',
        # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
        metrics=['accuracy'])

    # 可视化，TensorBoard 是由 Tensorflow 提供的一个可视化工具。
    tensorboard = TensorBoard(log_dir)

    # 训练模型, fit_generator函数:https://keras.io/models/model/#fit_generator
    # 利用Python的生成器，逐个生成数据的batch并进行训练。
    # callbacks: 实例列表。在训练时调用的一系列回调。详见 https://keras.io/callbacks/。
    d = model.fit_generator(
        # 一个生成器或 Sequence 对象的实例
        generator=train_generator,
        # epochs: 整数，数据的迭代总轮数。
        epochs=epochs,
        # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        steps_per_epoch=1845 // batch_size,
        # 验证集
        validation_data=validation_generator,
        # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        validation_steps=230 // batch_size,
        callbacks=[tensorboard])
    # 模型保存
    model.save(model_save_path)

    return d, model

