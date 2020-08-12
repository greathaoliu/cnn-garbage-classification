# 导入相关包
import keras
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Conv2D, ReLU, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard
import time
from config import config


def cnn_model(input_shape=config.input_shape, train_generator=None, validation_generator=None,
              model_save_path=config.model_path, log_dir=config.logs_path):

    inputs = Input(shape=input_shape)

    cnn = Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape)(inputs)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(64, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Flatten())(cnn)
    cnn = (Dense(128))(cnn)
    cnn = (Dropout(0.4))(cnn)
    cnn = (Dense(6))(cnn)

    outputs = cnn

    model = Model(inputs=inputs, outputs=outputs)

    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
        # 是优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer='Adam',
        # 损失函数,多分类采用 categorical_crossentropy
        loss='categorical_crossentropy',
        # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
        metrics=['accuracy'])

    # 可视化，TensorBoard 是由 Tensorflow 提供的一个可视化工具。
    tensorboard = TensorBoard(config.logs_path)

    # 训练模型, fit_generator函数:https://keras.io/models/model/#fit_generator
    # 利用Python的生成器，逐个生成数据的batch并进行训练。
    # callbacks: 实例列表。在训练时调用的一系列回调。详见 https://keras.io/callbacks/。
    d = model.fit_generator(
        # 一个生成器或 Sequence 对象的实例
        generator=train_generator,
        # epochs: 整数，数据的迭代总轮数。
        epochs=5,
        # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        steps_per_epoch=2259 // 32,
        # 验证集
        validation_data=validation_generator,
        # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        validation_steps=248 // 32,
        callbacks=[tensorboard])
    # 模型保存
    model.save(model_save_path)

    return d, model


def dnn_model(input_shape, train_generator, validation_generator, model_save_path='results/cnn.h5',
              log_dir="results/logs/"):
    """
    该函数实现 Keras 创建深度学习模型的过程
    :param input_shape: 模型数据形状大小，比如:input_shape=(384, 512, 3)
    :param train_generator: 训练集
    :param validation_generator: 验证集
    :param model_save_path: 保存模型的路径
    :param log_dir: 保存模型日志路径
    :return: 返回已经训练好的模型
    """
    # Input 用于实例化 Keras 张量。
    # shape: 一个尺寸元组（整数），不包含批量大小。 例如，shape=(32,) 表明期望的输入是按批次的 32 维向量。
    inputs = Input(shape=input_shape)

    # 将输入展平
    dnn = Flatten()(inputs)

    # Dense 全连接层  实现以下操作：output = activation(dot(input, kernel) + bias)
    # 其中 activation 是按逐个元素计算的激活函数，kernel 是由网络层创建的权值矩阵，
    # 以及 bias 是其创建的偏置向量 (只在 use_bias 为 True 时才有用)。
    dnn = Dense(6)(dnn)
    # 批量标准化层: 在每一个批次的数据中标准化前一层的激活项， 即应用一个维持激活项平均值接近 0，标准差接近 1 的转换。
    # axis: 整数，需要标准化的轴 （通常是特征轴）。默认值是 -1
    dnn = BatchNormalization(axis=-1)(dnn)
    # 将激活函数,输出尺寸与输入尺寸一样，激活函数可以是'softmax'、'sigmoid'等
    dnn = Activation('sigmoid')(dnn)
    # Dropout 包括在训练中每次更新时，将输入单元的按比率随机设置为 0, 这有助于防止过拟合。
    # rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
    dnn = Dropout(0.25)(dnn)

    dnn = Dense(12)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('relu')(dnn)
    dnn = Dropout(0.5)(dnn)

    dnn = Dense(6)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('softmax')(dnn)

    outputs = dnn

    # 生成一个函数型模型
    model = Model(inputs=inputs, outputs=outputs)

    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
        # 是优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer='Adam',
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
        epochs=5,
        # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        steps_per_epoch=2259 // 32,
        # 验证集
        validation_data=validation_generator,
        # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        validation_steps=248 // 32,
        callbacks=[tensorboard])
    # 模型保存
    model.save(model_save_path)

    return d, model
