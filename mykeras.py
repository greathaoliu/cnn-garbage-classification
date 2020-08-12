from config import config
import cnn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from resnet import res_model
import matplotlib.pyplot as plt
import glob, os, cv2, random, time


def standlize(img):
    img = (img - 0.5) / 0.5
    return img


def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    # -------------------------- 实现数据处理部分代码 ----------------------------
    validation_split = 0.1
    height = 384
    width = 512
    batch_size = config.batch_size

    train_data = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1. / 255,
        preprocessing_function=standlize,
        # # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        # shear_range=0.1,
        # # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        # zoom_range=0.1,
        # # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        # width_shift_range=0.1,
        # # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        # height_shift_range=0.1,
        # # 布尔值，进行随机水平翻转
        # horizontal_flip=True,
        # # 布尔值，进行随机竖直翻转
        # vertical_flip=True,
        # # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=validation_split
    )

    # 接下来生成测试集，可以参考训练集的写法
    validation_data = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1. / 255,
        preprocessing_function=standlize,
        # # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        # shear_range=0.1,x
        # # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        # zoom_range=0.1,
        # # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        # width_shift_range=0.1,
        # # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        # height_shift_range=0.1,
        # # 布尔值，进行随机水平翻转
        # horizontal_flip=True,
        # # 布尔值，进行随机竖直翻转
        # vertical_flip=True,
        # # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=validation_split
    )

    train_generator = train_data.flow_from_directory(
        # 提供的路径下面需要有子目录 directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹
        data_path,
        # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
        target_size=(height, width),
        # 一批数据的大小
        batch_size=batch_size,
        # "categorical", "binary", "sparse", "input" 或 None 之一。
        # 默认："categorical",返回one-hot 编码标签。
        class_mode='categorical',
        # 数据子集 ("training" 或 "validation")
        subset='training',
        seed=0)
    validation_generator = validation_data.flow_from_directory(
        data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=0)
    return train_generator, validation_generator
    # ------------------------------------------------------------------------
    # train_data, test_data = None, None
    # return train_data, test_data


def model(train_data, test_data, save_model_path):
    """
    创建、训练和保存深度学习模型
    :param train_data: 训练集数据
    :param test_data: 测试集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------

    # 选择模型，选择序贯模型（Sequential())
    # 通过将网络层实例的列表传递给 Sequential 的构造器，来创建一个 Sequential 模型
    # d, model = kerasmodel.cnn_model(input_shape=config.input_shape, train_generator=train_data,
    #                                 validation_generator=test_data,
    #                                 model_save_path=config.model_path, log_dir=config.logs_path)
    d, model = res_model(input_shape=config.input_shape, train_generator=train_data,
                                    validation_generator=test_data,
                                    model_save_path=config.model_path, log_dir=config.logs_path)
    # 打印模型概况
    model.summary()
    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------

    return model


def evaluate_mode(test_data, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_data: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    model = load_model(save_model_path)
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate_generator(test_data,steps=10)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    # ---------------------------------------------------------------------------


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = config.data_path  # 数据集路径
    save_model_path = config.model_path  # 保存模型路径和名称

    # 获取数据
    train_data, test_data = processing_data(data_path)

    # 创建、训练和保存模型
    model(train_data, test_data, save_model_path)

    # 评估模型
    evaluate_mode(test_data, save_model_path)


if __name__ == '__main__':
    main()
